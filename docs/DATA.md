# Data

What the training pipeline expects, how to get it, and how to keep the text
side consistent with the model.

## Manifest format

A CSV (or TSV) with two columns — no header required:

```csv
clips/clip0001.wav,"امروز هوا آفتابی است و من به پارک می‌روم."
clips/clip0002.wav,"قیمت این کتاب دویست و پنجاه هزار تومان است."
```

- Relative paths resolve against the manifest's directory.
- Transcripts go through `ava/text_norm.py` (the same normalizer inference
  uses) unless `--no-normalize` is passed. **Keep normalization on** — training
  and inference must see the same text distribution: Arabic-script variants
  mapped to Persian, digits spelled out, only Persian letters, ZWNJ, spaces and
  `،؛:.!؟` survive.
- Audio: any sample rate/channel count readable as WAV/FLAC/OGG (non-WAV and
  non-16-bit WAV need `pip install soundfile`). Clips are resampled to 24 kHz
  mono. Practical bounds: ≥ 0.8 s and ≤ 20 s per clip (`min_frames` /
  `max_frames` in the training config cap this at the dataset level, 12.5 Hz
  frames).

## Where to get Persian speech

The training recipe is data-hungry in the same way Kyutai's recipe is: real,
single-speaker-per-clip, transcribed speech. Options, roughly in order of
practicality:

| Dataset | Notes |
|---|---|
| **Mozilla Common Voice — Persian (fa)** | Large, community-recorded, per-clip transcripts. Gated download with HF auth; export the clips + `validated.tsv` into the manifest format. |
| **Your own recordings** | Best quality control: read sentences aloud, 3-20 s each, one speaker per clip, quiet room. 3-10 h of a single speaker already fine-tunes the released model usefully (`model.init_from`). |
| **CML-TTS / MLS Persian subsets** | Audiobook-style speech with transcripts; check each dataset's license before training a model you plan to distribute. |
| **Gol-e-Zarrin / other Persian ASR corpora** | Several exist on Hugging Face and GitHub; verify transcripts match audio (run `prepare_data` and inspect `skipped.json`). |

Licensing matters twice: the **dataset's** license governs what you may train
on, and the **frozen Mimi codec** inside any exported bundle is CC BY-NC — so a
model trained with this pipeline inherits the non-commercial restriction unless
you also retrain/replace the codec under a permissive license.

## Quality checklist before a long run

1. `prepare_data` prints total hours — for from-scratch training aim for
   hundreds of hours; for fine-tuning the released model, 3-10 h already moves
   the model meaningfully.
2. Listen to ~20 random clips; transcripts must match the audio word-for-word
   (the flow LM learns to pronounce whatever text sits next to a voice).
3. Check `stats.json`: `latent_std` should have no dimension near 0 (a dead
   dimension means something is wrong with the codec path, not the data).
4. Keep holdout_ratio ≥ 0.02 so `evaluate` measures generalization, not memory.

## Smoke data (for testing the pipeline, not for quality)

`scripts/make_smoke_data.py` bootstraps a tiny dataset by generating 16 short
Persian sentences with the **released** model and pairing them with their exact
transcripts — fully self-contained, no external download:

```bash
make smoke-data          # == scripts/make_smoke_data.py + prepare_data
make smoke-train         # trains the tiny flow LM, runs the tests' real-latent path
```

The point is pipeline correctness (loss falls, export round-trips), not speech
quality.
