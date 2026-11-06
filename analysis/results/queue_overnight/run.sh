#!/bin/bash
# Overnight queue: seeds, leave-one-out ablations of timbre32, bigger GRU, then (if present) the unseen-recordings test.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
SP=/tmp/claude-949830348/-home-sis-virtual-uniandes-edu-co-a-boteror-Documentos-tesis/6701c18f-b230-4c31-983d-9e49a9f1bfec/scratchpad
cp $SP/wind.py $R/queue_overnight/wind.py
base='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": true, "note_timbre": true, "timbre_k": 32'
full="{$base, \"two_decay\": true}"
run() { # tag config [extra args]
  echo "$(date +%T) start $1" >> $R/queue_overnight/status.txt
  uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$2" --tag "$1" "${@:3}" > $R/e9_structured$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $R/queue_overnight/status.txt
}
run _v9_timbre32_seed1 "$full" --seed 1
run _v9_timbre32_seed2 "$full" --seed 2
run _v9_t32_no_twodecay "{$base}"
run _v9_t32_no_bodyir "$(echo "$full" | sed 's/"body_ir": true/"body_ir": false/')"
run _v9_t32_no_click "$(echo "$full" | sed 's/"pluck_click": true/"pluck_click": false/')"
run _v9_t32_no_gate "$(echo "$full" | sed 's/"noise_gate": true/"noise_gate": false/')"
run _v9_t32_no_noise "$(echo "$full" | sed 's/}$/, "noise": false}/')"
run _v9_t32_gru512 "$(echo "$full" | sed 's/}$/, "hidden": 512}/')"
uv run python $R/queue_overnight/wind.py structured_v9_timbre32_seed1 structured_v9_timbre32_seed2 structured_v9_t32_no_twodecay structured_v9_t32_no_bodyir structured_v9_t32_no_click structured_v9_t32_no_gate structured_v9_t32_no_noise structured_v9_t32_gru512 > $R/queue_overnight/wind.txt 2>&1
echo "$(date +%T) wind done" >> $R/queue_overnight/status.txt
if [ -f $R/queue_overnight/stage2.sh ]; then bash $R/queue_overnight/stage2.sh; fi
echo "$(date +%T) all done" >> $R/queue_overnight/status.txt
