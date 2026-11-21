# Adapted from Kyutai pocket-tts v4.0.1 (https://github.com/kyutai-labs/pocket-tts), MIT License.
# Copyright (c) 2025 Kyutai. Adapted for the Ava project; see NOTICE.md for attribution.
# Changes vs. upstream: local/self-contained loading only (no hf:// downloads, no Kyutai
# voice catalog), sharded safetensors support, voices shipped with the repo, and the
# `audio_prompt` voice-file format used by the browser demo.
"""The TTS model: flow language model + Mimi codec, streaming inference."""

import copy
import logging
import math
import os
import queue
import statistics
import threading
import time
from pathlib import Path

import safetensors
import scipy.io.wavfile
import torch
from torch import nn
from torch.nn import functional as F

from ava.data.audio import audio_read
from ava.data.audio_utils import convert_audio, end_on_pause
from ava.models.flow_lm import FlowLMModel
from ava.models.mimi import build_mimi
from ava.models.model_state import _import_model_state, _is_safetensors_source
from ava.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from ava.modules.stateful_module import (
    ModelState,
    StatefulModule,
    increment_steps,
    init_states,
)
from ava.utils.config import Config, config_base_dir, load_config, resolve_path
from ava.utils.utils import display_execution_time, size_of_dict
from ava.utils.weights_io import load_file
from ava.utils.weights_loading import get_training_checkpoint_state_dicts

torch.set_num_threads(int(os.environ.get("AVA_TORCH_THREADS", "1")))
logger = logging.getLogger(__name__)

# Generated latents handed to the Mimi decoder thread; None closes the stream.
LatentQueue = queue.Queue[torch.Tensor | None]
# Decoder thread -> generator: ("chunk", audio), ("done", None) or ("error", exception).
ResultQueue = queue.Queue[tuple[str, object]]

MAX_TOKEN_PER_CHUNK = 50
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SAMPLER_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = None
DEFAULT_EOS_THRESHOLD = -2.0

DEBUG_MIMI = os.environ.get("AVA_DEBUG_MIMI", "0") == "1"


def stamp_state_names(tts_model: "TTSModel"):
    """StatefulModules find their slice of the state dict by absolute name."""
    for top_module in (tts_model.flow_lm, tts_model.mimi):
        for module_name, module in top_module.named_modules():
            if isinstance(module, StatefulModule):
                module._module_absolute_name = module_name


class TTSModel(nn.Module):
    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0
    # EOS is ignored on the first frames: before speech starts, the EOS logit of some voices can
    # cross the threshold, and a short text then ends before the word is spoken.
    _MIN_FRAMES_BEFORE_EOS = 6

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        config: Config,
        origin: Path | None = None,
        pad_with_spaces_for_short_inputs: bool = False,
        model_recommended_frames_after_eos: int | None = None,
        remove_semicolons: bool = False,
        append_terminal_punctuation: bool = True,
        capitalize_first_letter: bool = True,
        replace_characters: dict[str, str] | None = None,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        # 0 = decode every queued frame in one call; 1 = one frame per call
        self.max_decoder_frames_per_call = 0
        self.temp = temp
        self.sampler_decode_steps = sampler_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True
        self.origin = origin
        self.pad_with_spaces_for_short_inputs: bool = pad_with_spaces_for_short_inputs
        self.model_recommended_frames_after_eos = model_recommended_frames_after_eos
        self.remove_semicolons = remove_semicolons
        self.append_terminal_punctuation = append_terminal_punctuation
        self.capitalize_first_letter = capitalize_first_letter
        self.replace_characters = replace_characters or {}
        self._base = config_base_dir(origin) if origin is not None else None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls,
        config: Config,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        origin: Path | None,
    ) -> "TTSModel":
        # Resolve the tokenizer path against the config's repository root so the
        # model loads from any working directory.
        if origin is not None:
            base = config_base_dir(origin)
            config.flow_lm.lookup_table.tokenizer_path = str(
                resolve_path(config.flow_lm.lookup_table.tokenizer_path, base)
            )
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm,
            latent_dim=config.mimi.quantizer.dimension,
            insert_bos_before_voice=config.flow_lm.insert_bos_before_voice,
        )
        return cls(
            flow_lm,
            temp,
            sampler_decode_steps,
            noise_clamp,
            eos_threshold,
            config,
            origin=origin,
            pad_with_spaces_for_short_inputs=config.pad_with_spaces_for_short_inputs,
            model_recommended_frames_after_eos=config.model_recommended_frames_after_eos,
            remove_semicolons=config.remove_semicolons,
            append_terminal_punctuation=config.append_terminal_punctuation,
            capitalize_first_letter=config.capitalize_first_letter,
            replace_characters=config.replace_characters,
        )

    has_custom_weights: bool = False

    def load_training_checkpoint(self, path: str | Path, use_ema: bool = True):
        """Load weights from a training checkpoint (.pt) into a config-built model."""
        self.has_custom_weights = True
        flow_lm_state, mimi_state = get_training_checkpoint_state_dicts(Path(path), use_ema)
        config = self.config
        self.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros(
                (
                    config.flow_lm.transformer.d_model,
                    config.mimi.inner_dim or config.mimi.seanet.dimension,
                ),
                dtype=torch.float32,
            )
        )
        self.flow_lm.load_state_dict(flow_lm_state, strict=True)
        self.mimi = build_mimi(config.mimi).to(device="cpu")
        # Training freezes Mimi by default, so its weights come from the config's bundle.
        if mimi_state:
            self.mimi.load_state_dict(mimi_state, strict=False)
        elif config.weights_path is not None:
            bundled = load_file(resolve_path(config.weights_path, self._base))
            self.mimi.load_state_dict(
                {k.removeprefix("mimi."): v for k, v in bundled.items() if k.startswith("mimi.")},
                strict=True,
            )
        else:
            raise ValueError("no Mimi weights: set weights_path in the config")
        self.mimi.eval()
        stamp_state_names(self)

    @classmethod
    def _from_pydantic_config_with_weights(
        cls,
        config: Config,
        temp: float,
        sampler_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        origin: Path | None = None,
    ) -> "TTSModel":
        tts_model = cls._from_pydantic_config(
            config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
        )
        tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros(
                (
                    config.flow_lm.transformer.d_model,
                    config.mimi.inner_dim or config.mimi.seanet.dimension,
                ),
                dtype=torch.float32,
            )
        )
        tts_model.mimi = build_mimi(config.mimi).to(device="cpu")

        if config.weights_path is not None:
            weights_file = resolve_path(config.weights_path, tts_model._base)
            logger.info("Loading TTSModel weights from %s", weights_file)
            state_dict = load_file(weights_file)
            tts_model.load_state_dict(state_dict, strict=True)
        else:
            logger.warning("No weights_path specified, model is uninitialized!")

        tts_model.mimi.eval()
        stamp_state_names(tts_model)
        return tts_model

    @classmethod
    def load_model(
        cls,
        config: str | Path,
        temp: float | None = None,
        sampler_decode_steps: int = DEFAULT_SAMPLER_DECODE_STEPS,
        noise_clamp: float | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float | None = None,
        checkpoint: str | Path | None = None,
    ) -> "TTSModel":
        """Load the Ava TTS model from a local YAML config.

        Args:
            config: Path to a YAML config file (e.g. ``config/fa.yaml`` inside the repo).
            temp: Sampling temperature for generation. If None, defaults to the config's
                ``default_temperature`` (0.3).
            sampler_decode_steps: Number of flow decode steps (1 for the drifting head).
            noise_clamp: Maximum absolute value for the sampled noise (None = no clamp).
            eos_threshold: Generation stops when the EOS logit exceeds it. If None, defaults
                to the config's ``default_eos_threshold`` (-2.0).
            checkpoint: Optional training checkpoint (.pt) to load instead of the config's
                safetensors weights, so any step of a training run can be played directly.
                EMA weights are used when the checkpoint carries them.

        Returns:
            TTSModel: Fully initialized model on CPU, ready for inference.
        """
        model_config = load_config(config)
        if temp is None:
            temp = model_config.default_temperature
        if eos_threshold is None:
            eos_threshold = model_config.default_eos_threshold
        logger.info("Loading model from config at %s...", config)

        origin = Path(config)
        if checkpoint is not None:
            tts_model = cls._from_pydantic_config(
                model_config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            )
            tts_model.load_training_checkpoint(checkpoint)
        else:
            tts_model = cls._from_pydantic_config_with_weights(
                model_config, temp, sampler_decode_steps, noise_clamp, eos_threshold, origin=origin
            )
        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First one is the backbone output, second one is the audio decoding output."""
        if text_tokens is None:
            text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=self.flow_lm.device)
        if backbone_input_latents is None:
            backbone_input_latents = torch.empty(
                (1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )
        if audio_conditioning is None:
            audio_conditioning = torch.empty(
                (1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm(
        self,
        model_state: ModelState,
        text_tokens: torch.Tensor,
        backbone_input_latents: torch.Tensor,
        audio_conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = self.flow_lm.conditioner(text_tokens)
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            sampler_decode_steps=self.sampler_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _decode_and_dump(self, encoded: torch.Tensor, filename: str):
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=10000)
        restored_audio = self.mimi.decode_from_latent(encoded, mimi_state)
        scipy.io.wavfile.write(filename, self.sample_rate, restored_audio.numpy())
        logger.info("Saved restored audio from Mimi encoding to %s for debugging", filename)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        encoded = self.mimi.encode_to_latent(audio)

        if DEBUG_MIMI:
            # sanity check
            self._decode_and_dump(encoded, "debug_encoded_latent_decoded.wav")

        latents = encoded.to(torch.float32)
        # Voice conditioning = speaker projection over codec latents. The default
        # matches upstream pocket-tts / xn-ptts exactly (raw latents). Some
        # checkpoints trained outside Kyutai expect *normalized* latents here —
        # set `voice_cond_normalized: true` in the config for those; see
        # docs/ARCHITECTURE.md ("Voice conditioning") for the evidence.
        if getattr(self.config, "voice_cond_normalized", False):
            latents = (latents - self.flow_lm.emb_mean) / self.flow_lm.emb_std
        conditioning = F.linear(latents, self.flow_lm.speaker_proj_weight)
        return conditioning

    def _expand_kv_cache(self, model_state: ModelState, sequence_length: int):
        """Expand KV cache back to full sequence_length for generation.

        When a model state is retrieved from cache with sliced KV caches,
        this method expands them back to the full size needed for generation.
        """
        for module_name, module_state in model_state.items():
            if "cache" in module_state:
                cache = module_state["cache"]
                # KV cache has shape [2, batch_size, current_length, num_kv_heads, dim_per_head]
                current_length = cache.shape[2]
                if current_length < sequence_length:
                    expanded_cache = torch.full(
                        (
                            cache.shape[0],
                            cache.shape[1],
                            sequence_length,
                            cache.shape[3],
                            cache.shape[4],
                        ),
                        float("NaN"),
                        device=cache.device,
                        dtype=cache.dtype,
                    )
                    expanded_cache[:, :, :current_length, :, :] = cache
                    module_state["cache"] = expanded_cache

    def _flow_lm_current_end(self, model_state: ModelState) -> int:
        for module_state in model_state.values():
            offset = module_state.get("offset")
            if offset is not None:
                return int(offset.view(-1)[0].item())
        raise ValueError(
            "Could not find offset in model state; the state was probably built by another model."
        )

    @torch.no_grad
    def _decode_audio_worker(
        self,
        latents_queue: LatentQueue,
        result_queue: ResultQueue,
        mimi_sequence_length: int,
        mimi_steps_per_latent: int,
    ):
        """Worker thread function for decoding audio latents from queue with immediate streaming."""
        try:
            mimi_state = init_states(self.mimi, batch_size=1, sequence_length=mimi_sequence_length)
            # A fresh Mimi decoder state puts a small step (about -41 dBFS) in its first samples,
            # heard as a click at the start of every chunk: fade the first 5 ms in.
            fade_in: torch.Tensor | None = torch.linspace(0, 1, self.config.mimi.sample_rate // 200)
            while True:
                latent = latents_queue.get()
                if latent is None:
                    break
                # Decode every latent frame the generator has queued in one call. The first
                # frame never waits. max_decoder_frames_per_call=1 is frame-by-frame decoding.
                latents = [latent]
                finished = False
                while not finished and (
                    self.max_decoder_frames_per_call <= 0
                    or len(latents) < self.max_decoder_frames_per_call
                ):
                    try:
                        nxt = latents_queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        finished = True
                    else:
                        latents.append(nxt)
                mimi_decoding_input = (
                    torch.cat(latents, dim=1) * self.flow_lm.emb_std + self.flow_lm.emb_mean
                )

                t = time.monotonic()
                audio_frame = self.mimi.decode_from_latent(mimi_decoding_input, mimi_state)
                if fade_in is not None:
                    n = min(fade_in.numel(), audio_frame.shape[-1])
                    audio_frame[..., :n] *= fade_in[:n].to(audio_frame)
                    fade_in = None
                increment_steps(
                    self.mimi, mimi_state, increment=mimi_steps_per_latent * len(latents)
                )
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate
                logger.debug(
                    "Decoded %d ms of audio (%d frames) with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    len(latents),
                    int((time.monotonic() - t) * 1000),
                )
                result_queue.put(("chunk", audio_frame))

                for _ in latents:
                    latents_queue.task_done()
                if finished:
                    break

            result_queue.put(("done", None))

        except Exception as e:
            result_queue.put(("error", e))

    @torch.no_grad
    def generate_audio(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> torch.Tensor:
        """Generate complete audio from text; returns a float32 tensor [samples].

        The text is split into <= `max_tokens` chunks, each generated to EOS. The
        model state may come from `get_state_for_audio_prompt()` (voice cloning).
        """
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
        ):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad
    def generate_audio_stream(
        self,
        model_state: ModelState,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        stop: threading.Event | None = None,
    ):
        """Yield audio chunks ([samples] tensors) as they are decoded.

        Uses a worker thread so that Mimi decoding overlaps latent generation.
        """
        if frames_after_eos is None:
            frames_after_eos = self.model_recommended_frames_after_eos
        if stop is None:
            stop = threading.Event()

        # Long inputs are split on sentence boundaries into chunks <= max_tokens.
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text_to_generate,
            max_tokens,
            self.pad_with_spaces_for_short_inputs,
            remove_semicolons=self.remove_semicolons,
            append_terminal_punctuation=self.append_terminal_punctuation,
            capitalize_first_letter=self.capitalize_first_letter,
            replace_characters=self.replace_characters,
        )

        for chunk in chunks:
            if stop.is_set():
                break
            text_to_generate, frames_after_eos_guess = prepare_text_prompt(
                chunk,
                self.pad_with_spaces_for_short_inputs,
                self.remove_semicolons,
                self.append_terminal_punctuation,
                self.capitalize_first_letter,
                self.replace_characters,
            )
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=text_to_generate,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
                stop=stop,
            )

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self,
        model_state: ModelState,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        stop: threading.Event,
    ):
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.shape[1]
        max_gen_len = self._estimate_max_gen_len(token_count)
        mimi_steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        mimi_sequence_length = max_gen_len * mimi_steps_per_latent

        latents_queue: LatentQueue = queue.Queue()
        result_queue: ResultQueue = queue.Queue()

        decoder_thread = threading.Thread(
            target=self._decode_audio_worker,
            args=(latents_queue, result_queue, mimi_sequence_length, mimi_steps_per_latent),
            daemon=True,
        )
        t_generating = time.monotonic()
        decoder_thread.start()

        self._generate(
            model_state=model_state,
            prepared=prepared,
            max_gen_len=max_gen_len,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
            stop=stop,
        )

        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]  # Remove batch, channel
            elif result[0] == "done":
                break
            elif result[0] == "error":
                decoder_thread.join()
                raise result[1]

        decoder_thread.join()

        duration_generated_audio = int(total_generated_samples * 1000 / self.config.mimi.sample_rate)
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / max(generation_time, 1)
        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    @torch.no_grad
    def _generate(
        self,
        model_state: ModelState,
        prepared: torch.Tensor,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: LatentQueue,
        result_queue: ResultQueue,
        stop: threading.Event,
    ):
        token_count = prepared.shape[1]
        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)

        with display_execution_time("Prompting text", print_output=False):
            self._run_flow_lm_and_increment_step(model_state=model_state, text_tokens=prepared)

        def run_generation():
            try:
                self._autoregressive_generation(
                    model_state, max_gen_len, frames_after_eos, latents_queue, stop
                )
            except Exception as e:
                logger.error("Error in autoregressive generation: %s", e)
                if result_queue is not None:
                    result_queue.put(("error", e))
                if latents_queue is not None:
                    latents_queue.put(None)

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    @torch.no_grad
    def _autoregressive_generation(
        self,
        model_state: ModelState,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: LatentQueue,
        stop: threading.Event,
    ):
        backbone_input = torch.full(
            (1, 1, self.flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(self.flow_lm.parameters())).device,
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        for generation_step in range(max_gen_len):
            if stop.is_set():
                break
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )
                if (
                    bool(is_eos.item())
                    and eos_step is None
                    and generation_step >= self._MIN_FRAMES_BEFORE_EOS
                ):
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                latents_queue.put(next_latent)
                backbone_input = next_latent
            assert timer.elapsed_time_ms is not None
            steps_times.append(timer.elapsed_time_ms)
        else:
            if os.environ.get("AVA_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, this very often indicates an error."
            )

        latents_queue.put(None)
        if steps_times:
            logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> ModelState:
        """Create model state conditioned on a voice prompt (voice cloning).

        Accepts:
            - a path to a WAV/FLAC audio file (3-10 s recommended),
            - a path to a ``.safetensors`` voice file: either a raw ``audio_prompt``
              waveform tensor (the format shipped in ``weights/voices/``) or an
              exported per-module streaming state (from ``export_model_state``),
            - a float tensor [channels, samples] at the model's sample rate.
        """
        if isinstance(audio_conditioning, torch.Tensor):
            audio = audio_conditioning
            conditioning_sample_rate = self.config.mimi.sample_rate
        elif isinstance(audio_conditioning, (str, Path)) and _is_safetensors_source(
            audio_conditioning
        ):
            voice_path = Path(audio_conditioning)
            if not voice_path.exists() and self._base is not None:
                voice_path = resolve_path(voice_path, self._base)
            state = load_file(voice_path)
            if set(state.keys()) == {"audio_prompt"}:
                tensor = state["audio_prompt"].to(torch.float32)
                if tensor.ndim == 2 and tensor.shape[-1] == self.flow_lm.dim:
                    # Precomputed voice conditioning [T, d_model] (the browser-demo format):
                    # feed it straight into the transformer as the audio prompt.
                    prompt = tensor[None, ...].to(self.device)
                    if self.flow_lm.insert_bos_before_voice:
                        prompt = torch.cat([self.flow_lm.bos_before_voice, prompt], dim=1)
                    model_state = init_states(
                        self.flow_lm, batch_size=1, sequence_length=prompt.shape[1]
                    )
                    self._run_flow_lm_and_increment_step(
                        model_state=model_state, audio_conditioning=prompt
                    )
                    return model_state
                # Otherwise treat it as a raw waveform voice file.
                audio = tensor
                if audio.dim() == 1:
                    audio = audio[None, :]
                conditioning_sample_rate = self.config.mimi.sample_rate
            else:
                return _import_model_state(voice_path, self.device)
        else:
            voice_path = Path(audio_conditioning)
            if not voice_path.exists() and self._base is not None and not voice_path.is_absolute():
                # A bare name ("nava") resolves inside the config's voices dir.
                voices_dir = (
                    resolve_path(self.config.voices_dir, self._base)
                    if self.config.voices_dir
                    else None
                )
                for candidate in (
                    voices_dir / f"{voice_path}.safetensors" if voices_dir else None,
                    voices_dir / f"{voice_path}.wav" if voices_dir else None,
                ):
                    if candidate is not None and candidate.exists():
                        voice_path = candidate
                        break
                else:
                    if not voice_path.exists():
                        raise FileNotFoundError(
                            f"Voice '{audio_conditioning}' not found: give a WAV file path or a "
                            f"name shipped in {self.config.voices_dir}."
                        )
            audio, conditioning_sample_rate = audio_read(voice_path)

        if audio.dim() == 1:
            audio = audio[None, :]
        if truncate:
            max_samples = int(30 * conditioning_sample_rate)
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
                logger.info("Audio truncated to first 30 seconds (%d samples)", max_samples)

        audio = convert_audio(audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1)

        audio = end_on_pause(audio, self.config.mimi.sample_rate)
        with display_execution_time("Encoding audio prompt", print_output=False):
            prompt = self._encode_audio(audio.unsqueeze(0).to(self.device))

        if self.flow_lm.insert_bos_before_voice:
            prompt = torch.cat([self.flow_lm.bos_before_voice, prompt], dim=1)

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=prompt.shape[1])

        with display_execution_time("Prompting audio", print_output=False):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        logger.info(
            "Size of the model state for audio prompt: %d MB", size_of_dict(model_state) // 1e6
        )

        return model_state

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        frame_rate = self.config.mimi.frame_rate
        return math.ceil(gen_len_sec * frame_rate)
