#!/bin/bash
# Krea 2 Studio launcher: builds the engine if needed and starts the web UI.
#   ./run.sh            -> http://127.0.0.1:7861
#   ./run.sh setup      -> one-time: convert weights, build the Neural Engine programs, fit the preview, build the app
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ "$1" = "setup" ]; then
  H1=${KREA_GPU_UNITS:-3072}; C0=${KREA_QKVG_C0:-3072}
  $PY tools/convert.py te txt cond vae tokenizer                        # text path, conditioning, VAE
  $PY tools/convert.py dit_split --gpu-units $H1 --qkvg-c0 $C0          # the GPU's slice of every DiT layer
  TMPDIR="${TMPDIR:-/tmp}" $PY tools/build_ane.py --gpu-units $H1 --qkvg-c0 $C0   # the Neural Engine's slice
  # optional Fast preset (4-step LoRA merged in f32): the same pair of files with the LoRA
  # $PY tools/convert.py dit_split_fast --gpu-units $H1 --qkvg-c0 $C0 && $PY tools/build_ane.py --fast ...
  $PY tools/fit_latent_rgb.py                                            # live-preview projection
  app/build.sh                                                           # KreaImage.app
  exit 0
fi

make -C engine -s
exec $PY ui/server.py "$@"
