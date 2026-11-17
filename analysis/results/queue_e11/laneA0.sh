#!/bin/bash
# Experiment 11 (final model), lane A0 (00:55): the favourite's adversarial recipe on M1 + timbre 32 + per-note attack +
# room background (GRU), with the rise penalty against the harmonic ticks. Control for the attention candidates.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e11
echo "$(date +%T) start adv structured_v10_m1_burst_floor_noise_strongD_rise" >> $Q/status.txt
uv run python scripts/e6_adversarial.py --run structured_v10_m1_burst_floor --steps 3000 --d-ch 64 --d-lr-mult 2.0 \
  --d-kind complex --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise > $R/e11_adv_structured_v10_m1_burst_floor_noise_strongD_rise_log.txt 2>&1
echo "$(date +%T) end adv structured_v10_m1_burst_floor_noise_strongD_rise exit $?" >> $Q/status.txt
