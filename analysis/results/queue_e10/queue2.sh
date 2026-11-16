#!/bin/bash
# Re-prioritized queue (05:35): replaces the rest of run.sh and stage2.sh. Each lane first waits for its current run.
# Dropped (M2 + per-note attack showed no gain; cross-voice attention moves onto the strings): m2_burst_slow,
# m2_burst_slow_floor, m2_burst_floor, m2_xvoice.
# Lane A: strings (GRU), then strings + attention across strings.
# Lane B: contact fix on M2 + timbre 32 + adv; then the favourite's recipe on M1 + timbre 32 + per-note attack.
# Lane C: windowed attention, then M1 + per-note attack + room background.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
M1='"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32'
M2='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true'
finish() { # pid tag: wait for a run started by run.sh, then record its end for auto_eval
  while kill -0 $1 2>/dev/null; do sleep 30; done
  if grep -q "EVAL step 4000" $R/e10_structured$2_log.txt; then s=0; else s=1; fi
  echo "$(date +%T) end $2 exit $s" >> $Q/status.txt
}
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
  finish 466885 _v10_m2_floor
  run _v10_m2_strings "{$M2, \"strings\": true}"
  run _v10_m2_strings_xvoice "{$M2, \"strings\": true, \"cross_voice\": true}"
}
laneB() {
  finish 474460 _v10_m2_slow
  adv structured_v9_t32_no_gate '{"contact": true, "delta_rise": 20.0}' _noise_strongD_contact_rise
  adv structured_v10_m1_burst '{}' _noise_strongD
}
laneC() {
  finish 475263 _v10_m2_bilstm
  run _v10_m2_attn "{$M2, \"backbone\": \"attn\"}"
  run _v10_m1_burst_floor "{$M1, \"rich_burst\": true, \"room_floor\": true}"
}
laneA &
laneB &
laneC &
wait
echo "$(date +%T) all done" >> $Q/status.txt
echo "$(date +%T) stage2 done" >> $Q/status2.txt
