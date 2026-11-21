.PHONY: install test smoke-data smoke-train export demo web-dist clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -q

# Bootstrap a tiny self-contained dataset with the released model, then build
# the latent cache. Needed once before `make smoke-train` / real-latent tests.
smoke-data:
	python scripts/make_smoke_data.py --n-clips 16
	python -m training.data.prepare_data --manifest data_cache/smoke/smoke_manifest.csv \
		--arch training/configs/smoke_arch.yaml --out data_cache/smoke

# CPU smoke training: tiny flow LM on the smoke cache; loss must fall.
smoke-train:
	python -m training.train_flow_lm --config training/configs/smoke_cpu.yaml

# Export the latest smoke checkpoint into the Pocket TTS bundle format.
export:
	python -m training.export --checkpoint runs/smoke/checkpoint-latest.pt \
		--arch training/configs/smoke_arch.yaml --out runs/smoke/exported/model.safetensors

# Run the browser demo locally (merges sharded weights on first start).
demo:
	python scripts/serve.py

# Build the static demo bundle for external hosting (needs COOP/COEP headers).
web-dist:
	python scripts/prepare_web.py

clean:
	rm -rf runs data_cache web_dist web/model.safetensors .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
