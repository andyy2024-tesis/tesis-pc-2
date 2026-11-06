#!/bin/bash
# Stage 2: unseen recordings. Reference run trained with the 10 new recordings, then adaptation of the per-recording
# parameters only, for the final model and for two earlier models (same 50 held-out windows of the 10 new recordings).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
S=$R/queue_overnight/status.txt
full='{"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": true, "note_timbre": true, "timbre_k": 32, "two_decay": true}'
echo "$(date +%T) start data30 ref" >> $S
uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$full" --extra-recordings 10 --eval-extra --tag _v9_timbre32_data30 > $R/e9_structured_v9_timbre32_data30_log.txt 2>&1
echo "$(date +%T) end data30 exit $?" >> $S
for run in structured_v9_timbre32 structured_v5_body_audio ddsp_v5_audio; do
  ref=""; [ $run = structured_v9_timbre32 ] && ref="--ref structured_v9_timbre32_data30"
  echo "$(date +%T) start unseen $run" >> $S
  uv run python analysis/unseen_adapt.py $run --n 10 --steps 1500 $ref > $R/unseen_${run}_log.txt 2>&1
  echo "$(date +%T) end unseen $run exit $?" >> $S
done
