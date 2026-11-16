#!/bin/bash
# Lane G (10:40): the best candidate so far: M2 + timbre 32 + strings, adversarial recipe, with the rise penalty that
# removed the ticks (strings + adv brought them back: 23 vs 4). No contact noise (it stayed silent in the contact run).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
echo "$(date +%T) start adv structured_v10_m2_strings_noise_strongD_rise" >> $Q/status2.txt
uv run python scripts/e6_adversarial.py --run structured_v10_m2_strings --steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
  --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise > $R/e10_adv_structured_v10_m2_strings_noise_strongD_rise_log.txt 2>&1
echo "$(date +%T) end adv structured_v10_m2_strings_noise_strongD_rise exit $?" >> $Q/status2.txt
