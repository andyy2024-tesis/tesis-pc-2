#!/bin/bash
# After the 06:33 reboot: resume the two adversarial steps from their checkpoints (contact at 2000/3000, R at 1000/3000),
# then the tail from scratch (training keeps no optimizer state): M2rec, S1, S5, S3, S2, R32.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
advr() { # base run: continue the adversarial step from <out>/checkpoint.pt
  echo "$(date +%T) start adv structured$1_noise_strongD_rise (resumed)" >> $Q/status.txt
  uv run python scripts/e6_adversarial.py --run structured$1 --steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
    --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise --resume >> $R/e11_adv_structured$1_noise_strongD_rise_log.txt 2>&1
  echo "$(date +%T) end adv structured$1_noise_strongD_rise exit $?" >> $Q/status.txt
}
M2B='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true, "rich_burst": true'
M1BS="$M1B, \"strings\": true"
gate 8500 advr _v11_m1_attn_ahead_contact; pT4=$GATED
gate 8500 advr _v11_m1_attn_ahead_rec; pR=$GATED
pids=""
gate 6500 train _v11_m2_rec "{$M2B, \"rec_all\": true}"; pids="$pids $GATED"
gate 6500 train _v11_m1_strings_oldB "{$M1BS, \"strings_b\": \"pitch\"}"; pids="$pids $GATED"
wait $pT4; echo "$(date +%T) all done" >> $Q/status.txt
gate 6500 train _v11_m1_strings_guitarB "{$M1BS, \"strings_b\": \"guitar\"}"; pids="$pids $GATED"
wait $pR; echo "$(date +%T) rec done" >> $Q/status.txt
gate 6500 train _v11_m1_strings_decay "{$M1BS, \"decay_cond\": true}"; pids="$pids $GATED"
gate 6500 train _v11_m1_strings_oldvoices "{$M1BS, \"strings_voices\": false}"; pids="$pids $GATED"
gate 6500 train _v11_m1_attn_ahead_rec32 "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"rec_all\": true, \"rec_dim\": 32}"; pids="$pids $GATED"
wait $pids
echo "$(date +%T) strings done" >> $Q/status.txt
