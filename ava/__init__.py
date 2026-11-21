# Ava — Persian text-to-speech, Pocket TTS compatible.
# Inference code adapted from Kyutai pocket-tts v4.0.1 (MIT); training pipeline is original.
"""Ava: a 18.2M-parameter Persian TTS model in the Pocket TTS format.

Public API mirrors ``pocket_tts``:

    from ava import TTSModel

    model = TTSModel.load_model(config="config/fa.yaml")
    voice_state = model.get_state_for_audio_prompt("weights/voices/nava.safetensors")
    audio = model.generate_audio(voice_state, "سلام دنیا")
    # audio: float32 tensor [samples] at model.sample_rate (24000)
"""

from ava.models.model_state import export_model_state
from ava.models.tts_model import TTSModel

__all__ = ["TTSModel", "export_model_state"]
