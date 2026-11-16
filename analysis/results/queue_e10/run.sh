#!/bin/bash
# Experiment 10 overnight queue (2026-11-16): attack vs noise (step 2) and network backbones (step 4).
# Three lanes share the GPU (~2.5 GB each). All 4000 steps from scratch, evaluated on the usual 85 held-out windows.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
M1='"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32'
M2='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true'
run() { # tag config
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$2" --tag "$1" > $R/e10_structured$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
}
lane1() {
  run _v10_m2_burst "{$M2, \"rich_burst\": true}"
  run _v10_m2_slow "{$M2, \"slow_noise\": 0.2}"
  run _v10_m2_burst_slow "{$M2, \"rich_burst\": true, \"slow_noise\": 0.2}"
  run _v10_m2_burst_slow_floor "{$M2, \"rich_burst\": true, \"slow_noise\": 0.2, \"room_floor\": true}"
}
lane2() {
  run _v10_m1_burst "{$M1, \"rich_burst\": true}"
  run _v10_m2_floor "{$M2, \"room_floor\": true}"
  run _v10_m1_burst_floor "{$M1, \"rich_burst\": true, \"room_floor\": true}"
  run _v10_m2_burst_floor "{$M2, \"rich_burst\": true, \"room_floor\": true}"
}
lane3() {
  run _v10_m2_lstm "{$M2, \"backbone\": \"lstm\"}"
  run _v10_m2_bilstm "{$M2, \"backbone\": \"bilstm\"}"
  run _v10_m2_attn "{$M2, \"backbone\": \"attn\"}"
  run _v10_m2_xvoice "{$M2, \"cross_voice\": true}"
}
lane1 &
sleep 60
lane2 &
sleep 60
lane3 &
wait
echo "$(date +%T) all done" >> $Q/status.txt
