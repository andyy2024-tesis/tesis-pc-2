#!/bin/bash
# Stage 2 (starts when run.sh is done): strings from the tablature (step 3) and the fix for the ticks in the harmonics.
# Lane A: M2 + timbre 32 with one voice per string; then the same with attention across the strings.
# Lane B: adversarial fine-tune of M2 + timbre 32 (same recipe as M2 + timbre 32 + adv) with a per-voice contact noise
#         and a penalty on fast rises of the high-partial correction (the ticks copied the finger/nail contact sounds).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
until grep -q "all done" $Q/status.txt; do sleep 60; done
M2='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true'
run() { # tag config
  echo "$(date +%T) start $1" >> $Q/status2.txt
  uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$2" --tag "$1" > $R/e10_structured$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status2.txt
}
adv() { # base run, config overrides, tag
  echo "$(date +%T) start adv $1$3" >> $Q/status2.txt
  uv run python scripts/e6_adversarial.py --run $1 --steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
    --config "$2" --tag "$3" > $R/e10_adv_$1$3_log.txt 2>&1
  echo "$(date +%T) end adv $1$3 exit $?" >> $Q/status2.txt
}
laneA() {
  run _v10_m2_strings "{$M2, \"strings\": true}"
  run _v10_m2_strings_xvoice "{$M2, \"strings\": true, \"cross_voice\": true}"
}
laneB() {
  adv structured_v9_t32_no_gate '{"contact": true, "delta_rise": 20.0}' _noise_strongD_contact_rise
}
laneA &
sleep 30
laneB &
wait
echo "$(date +%T) stage2 done" >> $Q/status2.txt
