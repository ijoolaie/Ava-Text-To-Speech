# Architecture

This document describes the model exactly as it is implemented here, how the
implementation was verified against the released checkpoint, and the one place
where this repository had to make a judgment call (voice conditioning).

## 1. Overview

Ava is a Pocket-TTS-format model: an autoregressive **flow language
model** over continuous codec latents, plus a **Mimi codec** that turns 12.5 Hz
latents into 24 kHz audio. One checkpoint holds both:

```
audio (24 kHz) ──Mimi encoder──> 200 Hz, 512-d ──encoder transformer──> ──downsample (k32, s16)──> 12.5 Hz, 32-d latents
                                                                                        │
voice prompt ──────────────── speaker projection (32 → 512) ──────────────┐             │
text ──tokenizer──> LUT embedding (4005+1 bins → 512) ────────────────────┤             ▼
                                                                        Flow LM ── next 32-d latent ──┐
                                                                                                      │
                              audio <──Mimi decoder──<──decoder transformer──<──output_proj (32→128)──┘
```

Parameter audit (must match the model card's numbers, and does, exactly):

| Component | Params | Claimed |
|---|---|---|
| flow_lm.* | 15.559M | 15.6M |
| Mimi decoder (decoder SeaNet + decoder transformer + quantizer + upsample) | 2.617M | 2.6M |
| Mimi encoder (encoder SeaNet + encoder transformer + downsample) | 9.749M | 9.7M |
| Total | 27.93M | 27.9M |

## 2. Flow LM (the trainable part)

- **Backbone**: 8 pre-norm transformer layers, d_model 512, 8 heads, FFN 512
  (hidden_scale 1), GELU-tanh FFN, RoPE (max_period 10000), full causal
  attention with a 250-position KV cache window in streaming mode. No
  layer_scale, no biases in attention/FFN.
- **Inputs**: the sequence is the concatenation of *conditioning positions*
  (text LUT embeddings + voice-conditioning projections) followed by *latent
  positions* (input_linear: 32 → 512). The first latent slot is a NaN sentinel
  replaced by the learned `bos_emb`.
- **Output head per latent position**: LayerNorm → `z_proj` (512 → 64).
  - `out_eos`: linear 64 → 1; generation stops when the logit exceeds
    `eos_threshold` (-2.0), ignoring the first 6 frames.
  - Sampler head (`flow_net`, a 4-block DiT-style AdaLN MLP, width 192,
    `flow.type: drifting`): takes the 64-d conditioning and a 32-d noise
    x₀ ~ N(0, temp²) and **directly outputs the next latent in one step**
    (no time conditioning — `num_time_conds = 0`).
- **Latent normalization**: generation produces latents in a normalized space;
  `emb_mean`/`emb_std` (registered buffers in the checkpoint) denormalize them
  for the Mimi decoder (`x·std + mean`).

## 3. Mimi codec (frozen in the released recipe)

- **Encoder** (speaker encoder): SeaNet, 1→64→128→256→512 channels, strides
  [4, 5, 6] (200 Hz), kernel 7, ELU activations, then a 2-layer/512-d
  transformer (context 250, layer_scale 0.01) and `downsample` (512 → 32,
  kernel 32, stride 16, replicate pad).
- **Decoder**: `output_proj` 32 → 128 (12.5 Hz), 12-layer/128-d transformer
  (context 50), `upsample` (grouped ConvTranspose, stride 16), then a SeaNet
  decoder 128→256→128→64→32→1, strides [6, 5, 4] → 24 kHz.
- The codec is **continuous** (no RVQ): the "quantizer" is just the 32 → 128
  output projection (`DummyQuantizer`), so the flow LM models unquantized
  latents directly.

## 4. Streaming

Every conv and attention module is `StatefulModule`. Generation keeps per-layer
KV caches (with a RoPE offset and padding slots) and per-conv overlap buffers,
so a 1-frame step is O(1). A decoder thread overlaps Mimi decoding with latent
generation; the first 5 ms of each stream is faded in to mask a startup click.

## 5. How this implementation was verified

1. **Strict weight load**: `TTSModel.load_state_dict(state_dict, strict=True)`
   accepts the released 302-tensor checkpoint through the shard index — every
   tensor name and shape matches the vendored architecture exactly.
2. **Parameter audit**: 15.56M / 2.62M / 9.75M / 27.93M (§1).
3. **Generation**: real Persian speech from the shipped voice files at ≈ 4×
   real time on 2 CPU cores; an ASR system transcribes the output back to the
   input text.
4. **Round-trip**: encode → decode of generated audio is consistent (11-14 dB
   latent/waveform SNR, no frame shift, no per-dimension bias), which rules out
   alignment or padding bugs in the encoder/decoder pair.
5. **Training**: the smoke pipeline trains a flow LM on real frozen-codec
   latents and the loss measurably falls (see docs/TRAINING.md).

## 6. Voice conditioning — an investigated deviation

Upstream Kyutai code (pocket-tts 4.0.1 Python and xn-ptts Rust) computes the
voice conditioning as `speaker_proj(raw encoder latents)`. The voice files
shipped with Ava (`weights/voices/*.safetensors`, tensors `[T, 512]`)
were produced by the model author's own export pipeline, which we could not
inspect (the source repo is gated).

What we established empirically:

- A least-squares span test proves the shipped tensors are (up to a 3.6%
  residual) in the column space of the checkpoint's `speaker_proj_weight` —
  i.e. they are `speaker_proj(z)` for some 32-d latents `z`, **not** projections
  through `input_linear` (74% residual) or any other module. `input_linear` is
  ruled out; `speaker_proj` is confirmed.
- The recovered `z` statistics are consistent with raw latents of real speech
  (per-dim std ≈ 0.6-0.9 × `emb_std`, non-zero per-clip means), but the files
  may have been produced with a slightly different `speaker_proj` value (3.6%
  off-space residual ≈ weight drift between export time and release) or with a
  speaker-encoder state not present in the released checkpoint.
- Feeding *our* re-encoded conditionings (from espeak audio, real speech, or
  model-generated audio) through `speaker_proj(raw latents)` breaks the EOS
  head (logits stuck at +2…+8 → generation stops after the 6-frame minimum).
  Normalizing the latents first fixes some cases (espeak) but not all.

Conclusion: the shipped voice files always work; **cloning from arbitrary WAVs
is upstream-compatible but may stop early on some prompts with this
checkpoint**. `config.voice_cond_normalized` (default `false` = upstream
behavior) switches `_encode_audio` to project normalized latents; the training
pipeline takes the same setting (`training/flow_lm_training.py`), so a model
you train yourself is self-consistent under either choice.

## 7. File provenance

Code marked "Vendored from Kyutai pocket-tts v4.0.1" is MIT-licensed Kyutai code
adapted for this repo (import renames, local-only loading, sharded weights,
repository voices). Everything else — training pipeline, sharded weights I/O,
CLI scripts, Persian normalizer, tests, docs — is original to this repository.
Full table in NOTICE.md.
