#!/bin/bash
# Krea 2 Studio launcher: builds the engine if needed and starts the web UI.
#   ./run.sh            -> http://127.0.0.1:7861
#   ./run.sh setup      -> one-time: convert weights, build the Neural Engine programs (Quality and Fast presets),
#                          fit the live preview, build and install KreaImage.app
# Setup expects the source checkpoints in weights/src (downloaded separately; the Krea 2 license is accepted on
# Hugging Face by the user): the Krea 2 Turbo DiT, the 4-step LoRA, the Qwen-Image VAE, the Krea 2 tokenizer and
# configs, and the text encoder's layers 0-34 (tools/fetch_te.py).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ "$1" = "setup" ]; then
  H1=${KREA_GPU_UNITS:-3072}  # MLP hidden units computed on the GPU; the Neural Engine takes the rest
  make -C engine -s
  $PY tools/convert.py te txt cond vae tokenizer                      # text path, conditioning, VAE
  $PY tools/convert.py dit_split dit_split_fast --gpu-units $H1       # the GPU's share of every DiT layer
  $PY tools/build_ane.py --gpu-units $H1                              # the Neural Engine's share (weights/ane)
  $PY tools/build_ane.py --gpu-units $H1 --fast                       # Fast preset (weights/ane_fast)
  $PY tools/fit_latent_rgb.py                                         # live-preview projection
  app/build.sh --install                                              # /Applications/KreaImage.app
  exit 0
fi

make -C engine -s
exec $PY ui/server.py "$@"
