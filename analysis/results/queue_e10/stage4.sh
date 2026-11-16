#!/bin/bash
# Stage 4 (11:55, GPU idle): is strings robust (second seed)? M1 + per-note attack + room background + strings (snare and
# wind fixed, plus strings); strings + bidirectional LSTM (both best offline ingredients).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
M1='"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32'
M2='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true'
run() { # tag config [extra args]
  echo "$(date +%T) start $1" >> $Q/status2.txt
  uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$2" --tag "$1" "${@:3}" > $R/e10_structured$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status2.txt
}
run _v10_m2_strings_seed1 "{$M2, \"strings\": true}" --seed 1 &
sleep 20
run _v10_m1_burst_floor_strings "{$M1, \"rich_burst\": true, \"room_floor\": true, \"strings\": true}" &
sleep 20
run _v10_m2_strings_bilstm "{$M2, \"strings\": true, \"backbone\": \"bilstm\"}" &
wait
echo "$(date +%T) stage4 done" >> $Q/status2.txt
