# -*- coding: utf-8 -*-
# Ava — Persian text-to-speech engine.
# History: the legacy model stack (ava/models/tts_model.py, ava/models/flow_lm.py,
# ava/models/mimi.py, ava/modules/) is inference code adapted from Kyutai
# pocket-tts v4.0.1 (MIT). It is kept ONLY for comparison/evaluation and is NOT
# a mandatory dependency: the product path (AVA Native) never imports it.
# See NOTICE.md and docs/CORE_INDEPENDENCE_AUDIT.md.
"""Ava: Persian text-to-speech engine.

The product path is AVA Native — ``ava.language`` + ``ava.models.native`` +
``ava.runtime`` + ``ava.audio`` — fully independent: importing it does NOT
load any third-party model code or weights.

The legacy Pocket-TTS-format model remains available for comparison and
evaluation, and is loaded lazily (PEP 562) only when explicitly requested:

    from ava import TTSModel            # lazy; pulls the legacy stack

    model = TTSModel.load_model(config="config/fa.yaml")
    voice_state = model.get_state_for_audio_prompt("weights/voices/nava.safetensors")
    audio = model.generate_audio(voice_state, "سلام دنیا")
    # audio: float32 tensor [samples] at model.sample_rate (24000)
"""

__all__ = ["TTSModel", "export_model_state"]


def __getattr__(name: str):  # PEP 562 — legacy symbols load lazily, on demand
    if name == "TTSModel":
        from ava.models.tts_model import TTSModel
        return TTSModel
    if name == "export_model_state":
        from ava.models.model_state import export_model_state
        return export_model_state
    raise AttributeError(f"module 'ava' has no attribute {name!r}")
