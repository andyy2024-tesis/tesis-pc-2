#!/bin/bash
# Stage 3 (07:40): combine what won. Strings (5.67) > attention (5.73) ~ biLSTM (5.70) > GRU (5.79); per-note attack fixes
# M1's snare. Each lane starts when one of the stage-2 lanes ends.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
M1='"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32'
M2='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true'
after() { until grep -q "$1" $Q/status2.txt; do sleep 30; done; }
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
laneD() {
  after "end adv structured_v9_t32_no_gate_noise_strongD_contact_rise exit"
  adv structured_v10_m2_strings '{}' _noise_strongD
}
laneE() {
  after "end _v10_m1_burst_floor exit"
  run _v10_m2_strings_attn "{$M2, \"strings\": true, \"backbone\": \"attn\"}"
}
laneF() {
  after "end _v10_m2_strings_xvoice exit"
  run _v10_m1_burst_strings "{$M1, \"rich_burst\": true, \"strings\": true}"
  adv structured_v10_m1_burst_strings '{}' _noise_strongD
}
laneD &
laneE &
laneF &
wait
echo "$(date +%T) stage3 done" >> $Q/status2.txt
