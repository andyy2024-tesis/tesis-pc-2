#!/bin/bash
# Experiment 11, lane T1 (00:55): M1 + timbre 32 + per-note attack + room background with windowed causal attention
# (the user's ear favourite on M2), then the adversarial recipe with the rise penalty.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e11
M1B='"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32, "rich_burst": true, "room_floor": true'
T=_v11_m1_attn
echo "$(date +%T) start $T" >> $Q/status.txt
uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "{$M1B, \"backbone\": \"attn\"}" --tag $T > $R/e11_structured${T}_log.txt 2>&1
echo "$(date +%T) end $T exit $?" >> $Q/status.txt
echo "$(date +%T) start adv structured${T}_noise_strongD_rise" >> $Q/status.txt
uv run python scripts/e6_adversarial.py --run structured$T --steps 3000 --d-ch 64 --d-lr-mult 2.0 \
  --d-kind complex --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise > $R/e11_adv_structured${T}_noise_strongD_rise_log.txt 2>&1
echo "$(date +%T) end adv structured${T}_noise_strongD_rise exit $?" >> $Q/status.txt
