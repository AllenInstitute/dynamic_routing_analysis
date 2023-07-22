#!/usr/bin/env bash

conda deactivate
python -m pip install pipx
pipx install pdm
pdm venv create
source .venv/bin/activate
pdm install