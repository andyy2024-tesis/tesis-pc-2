#!/bin/bash
# Measure and render finished experiment-10 runs: loss and wind (catalog), the "snare" measure, the ticks in the
# harmonics, listening excerpts (mix and noise off) and per-part stems. Usage: evaluate.sh <run> [<run> ...]
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=analysis/results/queue_e10/evaluate_log.txt
for r in "$@"; do
  echo "$(date +%T) evaluate $r" >> $L
  uv run python analysis/measure_catalog.py "$r" >> $L 2>&1
  uv run python analysis/noise_attack.py "$r" >> $L 2>&1
  uv run python analysis/noise_vs_room.py "$r" >> $L 2>&1
  uv run python analysis/harmonic_clicks.py "$r" >> $L 2>&1
  uv run python analysis/render_slide_audio.py "$r" "$r:nonoise" >> $L 2>&1
  uv run python analysis/render_stems.py "$r" >> $L 2>&1
  echo "$(date +%T) done $r" >> $L
done
