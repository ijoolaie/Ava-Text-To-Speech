# Ali Fathi Persian TTS - notices

The optional ONNX backend uses the pretrained model published at https://huggingface.co/alifthi/persian-TTS. The model weights are licensed CC BY 4.0. The model file (`model.onnx`) is not included in this Git repository and must be obtained from the official model repository. Keep the official license and model card with any separately distributed weights.

Required model credit:

> Persian VITS TTS - created by Ali Fathi (https://github.com/alifthi)

The model card identifies LibriTTS-R (CC BY 4.0), FLEURS (CC BY 4.0, if used), Mana-TTS (CC0), GPTInformal (CC0), and Persian-Speech (MIT) as training data, and the Piper `_base_model` (MIT) trained on LibriTTS-R as initialization. Preserve the model card's required attribution to LibriTTS-R and, if applicable, FLEURS.

The AVA adapter code is original repository code under the repository's MIT license; it does not copy code from the model's training repository, which has no license file.

External runtime components:

- eSpeak NG is GPL-3.0-or-later. It is an optional external executable, not bundled or redistributed by AVA.
- ONNX Runtime is MIT-licensed and installed as an optional dependency; it is not vendored by AVA.

The legacy PocketTTS weights and their CC BY-NC 4.0 terms are separate and are not used by this backend.