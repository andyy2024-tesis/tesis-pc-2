#!/bin/bash
# Stage 3: simplified final model (no two decays, no click) and how much audio a new recording needs.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
S=$R/queue_overnight/status.txt
for sec in 10 30 60; do
  echo "$(date +%T) start unseen timbre32 ${sec}s" >> $S
  uv run python analysis/unseen_adapt.py structured_v9_timbre32 --n 10 --steps 1000 --train-seconds $sec > $R/unseen_structured_v9_timbre32_${sec}s_log.txt 2>&1
  echo "$(date +%T) end unseen ${sec}s exit $?" >> $S
done
simple='{"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "noise_gate": true, "note_timbre": true, "timbre_k": 32}'
echo "$(date +%T) start _v9_t32_simple" >> $S
uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$simple" --tag _v9_t32_simple > $R/e9_structured_v9_t32_simple_log.txt 2>&1
echo "$(date +%T) end _v9_t32_simple exit $?" >> $S
uv run python $R/queue_overnight/wind.py structured_v9_t32_simple >> $R/queue_overnight/wind.txt 2>&1
echo "$(date +%T) stage3 done" >> $S
