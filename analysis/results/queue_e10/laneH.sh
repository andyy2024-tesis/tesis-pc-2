#!/bin/bash
# Lane H (13:25): the favourite's adversarial recipe on M1 + timbre 32 + per-note attack + room background + strings
# (no snare, 4 dB less wind, strings: 5.70 before adv).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
echo "$(date +%T) start adv structured_v10_m1_burst_floor_strings_noise_strongD" >> $Q/status2.txt
uv run python scripts/e6_adversarial.py --run structured_v10_m1_burst_floor_strings --steps 3000 --d-ch 64 --d-lr-mult 2.0 \
  --d-kind complex --config '{}' --tag _noise_strongD > $R/e10_adv_structured_v10_m1_burst_floor_strings_noise_strongD_log.txt 2>&1
echo "$(date +%T) end adv structured_v10_m1_burst_floor_strings_noise_strongD exit $?" >> $Q/status2.txt
echo "$(date +%T) stage5 done" >> $Q/status2.txt
