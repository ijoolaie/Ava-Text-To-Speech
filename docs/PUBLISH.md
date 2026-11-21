# Publishing to GitHub

The repository is git-ready (`git init` was already run; see below). Everything
needed for a public GitHub repository is in place:

- weights are sharded into files **< 100 MB** (GitHub's hard per-file limit) with
  a `model.safetensors.index.json` weight map,
- `.gitignore` keeps training artifacts (runs/, data_cache/, merged weights,
  checkpoints) out of git,
- LICENSE (MIT, code) + NOTICE.md (CC BY-NC 4.0 weights, phonon-tts vendoring)
  are committed,
- CI (GitHub Actions) runs the test suite and the CPU smoke pipeline on every
  push.

## Steps

The repository ships with one commit on `main` and the remote already pointing
at `https://github.com/ijoolaie/Text-To-Speech.git`. Pushing needs **your**
GitHub credentials, which are not (and should not be) stored in this bundle:

```bash
# from the extracted repository folder
git remote -v                            # origin -> github.com/ijoolaie/Text-To-Speech
git push -u origin main

# authentication, pick one:
#   - SSH:  git remote set-url origin git@github.com:ijoolaie/Text-To-Speech.git
#     (requires your SSH key registered on the account)
#   - HTTPS with a fine-grained Personal Access Token limited to this repo:
#     git push  # the credential prompt appears; paste the token
#   - or run `gh auth login` once, then push
```

Everything else is done: single commit containing all 118 tracked files
(weights shards included), `main` branch set, CI workflow in place.

## Size expectations

| Path | Size | In git? |
|---|---|---|
| `weights/model-0000X-of-00003.safetensors` + index | 112 MB | yes (plain git, 3 files < 100 MB each) |
| `weights/tokenizer.json`, `model_config.json`, `voices/*` | ~1 MB | yes |
| `web/vendor/phonon-tts/` (WASM runtime) | ~15 MB | yes |
| `samples/*.wav` | < 1 MB | yes |
| `runs/`, `data_cache/`, `web/model.safetensors`, `web_dist/` | — | no (`.gitignore`) |

Plain git handles this fine (no LFS needed). If you prefer LFS for the shards,
`git lfs migrate import --include="weights/*.safetensors"` before the first push
— but note LFS bandwidth quotas on free accounts.

## GitHub Pages for the web demo

The demo is a static site but **needs COOP/COEP headers** for the multithreaded
WASM build, which GitHub Pages does not send. Options:

1. **`scripts/serve.py` locally or on any small VM** — recommended; it also
   builds the merged `model.safetensors` automatically.
2. Any static host that lets you set headers (Cloudflare Pages `_headers`,
   Netlify, nginx `add_header`):
   ```bash
   python scripts/prepare_web.py            # builds web_dist/ (root layout + model/)
   # upload web_dist/, add:
   #   Cross-Origin-Opener-Policy: same-origin
   #   Cross-Origin-Embedder-Policy: require-corp
   ```
3. GitHub Pages with the single-threaded fallback only — the demo still works
   but is slower (the runtime detects the missing cross-origin isolation and
   loads the non-threaded WASM).

The demo fetches `model/model.safetensors` relative to the page root: on a
static host, either include the merged 112 MB file (over Pages' 100 MB file
limit — not possible) or serve it from a release asset / external URL and adjust
`MODEL_URL` in `web/app.js`. Local `serve.py` has no such problem.

## After publishing

- Add topics: `tts`, `persian`, `text-to-speech`, `pocket-tts`, `kyutai`.
- Link the model card: https://huggingface.co/mehdi-hf/cilla-mini (the weights'
  origin, CC BY-NC 4.0).
- The first CI run takes ~10-15 minutes (CPU: installs torch, runs pytest +
  the smoke data/train/export pipeline).
