#!/usr/bin/env bash

conda deactivate
python -m pip install pipx
pipx install pdm
dirname=${PWD##*/} 
pyversion=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
python -m venv .venv --copies --without-pip --prompt "$dirname-$pyversion"
source .venv/bin/activate
pdm install
