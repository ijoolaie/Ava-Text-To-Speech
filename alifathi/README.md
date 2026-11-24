# Ali Fathi Persian TTS backend

AVA includes an optional, inference-only adapter for the pretrained Persian VITS ONNX model published by Ali Fathi:

- Model card and weights: https://huggingface.co/alifthi/persian-TTS
- Model license: CC BY 4.0. Commercial use and redistribution are permitted under that license with the required attribution.
- Sample rate: 22050 Hz; seven speaker IDs are exposed by the published model configuration.

## Install

```powershell
pip install ".[alifathi]"
```

The adapter requires an external `espeak-ng` installation with the Persian (`fa`) voice. eSpeak NG is GPL-3.0-or-later and is not bundled or redistributed by AVA. ONNX Runtime is installed through the optional extra above and is not vendored.

## Add the model files

The ONNX weight file is intentionally **not stored in this Git repository**. Download the official `onnx/model.onnx` file from the model repository and place it here as `alifathi/model.onnx`. Keep the matching `model.onnx.json` alongside it; the configuration and symbol reference files in this directory are included for convenience. No model download occurs at runtime.

Alternatively, set `AVA_ALIFATHI_MODEL_DIR` to a local directory containing `model.onnx` and `model.onnx.json`.

## Attribution

Include this credit in a user-visible place in any product that uses the model:

> Persian VITS TTS - created by Ali Fathi (https://github.com/alifthi)

The model card also requires attribution for LibriTTS-R and, if used, FLEURS. See `NOTICE.md` and the official model card for details.