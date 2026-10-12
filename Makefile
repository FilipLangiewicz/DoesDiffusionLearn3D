# Scripted install, checks and runs. Use "make help" for the list.
PY ?= .venv/bin/python
BLENDER ?= blender

.PHONY: help venv install lint format test audit cues probe-pixels smoke-cpu article clean

help:
	@echo "venv          make the virtual environment .venv"
	@echo "install       install the package with the CPU model extras and dev tools"
	@echo "lint          ruff check (no changes)"
	@echo "format        ruff format and ruff check --fix"
	@echo "test          unit tests (CPU, no model weights)"
	@echo "audit         label and arrow audit of all scenes"
	@echo "cues          pixel cue probes (no diffusion model)"
	@echo "probe-pixels  pixel probe with 95% intervals"
	@echo "smoke-cpu     corrected SANA extraction on 3 example images (CPU, slow)"
	@echo "article       build the article PDF with tectonic"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip

install:
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	$(PY) -m pip install -e ".[model,track,dev]"

lint:
	$(PY) -m ruff check src tests scripts tools/blender

format:
	$(PY) -m ruff format src tests scripts tools/blender
	$(PY) -m ruff check --fix src tests scripts tools/blender

test:
	$(PY) -m pytest tests/unit

audit:
	$(PY) scripts/audit/arrow_audit.py

cues:
	$(PY) scripts/experiments/pixel_cues.py

probe-pixels:
	$(PY) -m dd3d.cli probe pixels --out reports/probes/pixels64_global
	$(PY) -m dd3d.cli probe pixels --target local --out reports/probes/pixels64_local

smoke-cpu:
	$(PY) scripts/cpu/make_prompt_cache.py
	$(PY) -m dd3d.cli extract --config configs/extract_cpu_smoke.yaml

article:
	cd docs/article && tectonic -X compile main.tex --outdir build

clean:
	rm -rf .pytest_cache .ruff_cache docs/article/build
