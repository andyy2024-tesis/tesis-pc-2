#!/bin/bash
# Stage 4: timbre WITHOUT the noise gate (user: the gate brought the artifacts back). Combined and base, each + adversarial.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
S=$R/queue_overnight/status4.txt
ADV="--steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex --clip-every 1000 --tag _noise_strongD"
BASE='{"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32}'
echo "$(date +%T) start base+timbre" >> $S
uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$BASE" --tag _v9_base_t32 > $R/e9_structured_v9_base_t32_log.txt 2>&1
echo "$(date +%T) end base+timbre exit $?" >> $S
for run in structured_v9_base_t32 structured_v9_t32_no_gate; do
  echo "$(date +%T) start adv $run" >> $S
  uv run python scripts/e6_adversarial.py --run $run $ADV > $R/e6_${run}_noise_strongD_log.txt 2>&1
  echo "$(date +%T) end adv $run exit $?" >> $S
done
echo "$(date +%T) all done" >> $S
