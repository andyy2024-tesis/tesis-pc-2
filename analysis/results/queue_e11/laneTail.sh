#!/bin/bash
# Tail of experiment 11, in order, no adversarial (replaces laneS.sh): M2 + per-note attack with the recording in every
# part (did M2 fail because its double decay, beating and click were the same for every guitar?); the strings runs
# (S1 old stiffness law, S5 stiffness per guitar, S3 decay that knows string/fret/guitar, S2 old voices); R with a
# 32-dim recording embedding (16 < 20 recordings limits the linear per-recording readouts). Starts once R is in its
# adversarial step, so the final candidates keep priority.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
M2B='"body_ir": true, "audio_feats": true, "body_ir_cutoff": 6000.0, "body_eq_max": 8000.0, "body_eq_bands": 10, "pluck_click": true, "noise_gate": false, "note_timbre": true, "timbre_k": 32, "two_decay": true, "rich_burst": true'
M1BS="$M1B, \"strings\": true"
until grep -qE "start adv structured_v11_m1_attn_ahead_rec_noise_strongD_rise|rec done" $Q/status.txt; do sleep 60; done
echo "$(date +%T) tail queued" >> $Q/status.txt
pids=""
gate 6500 train _v11_m2_rec "{$M2B, \"rec_all\": true}"; pids="$pids $GATED"
gate 6500 train _v11_m1_strings_oldB "{$M1BS, \"strings_b\": \"pitch\"}"; pids="$pids $GATED"
gate 6500 train _v11_m1_strings_guitarB "{$M1BS, \"strings_b\": \"guitar\"}"; pids="$pids $GATED"
gate 6500 train _v11_m1_strings_decay "{$M1BS, \"decay_cond\": true}"; pids="$pids $GATED"
gate 6500 train _v11_m1_strings_oldvoices "{$M1BS, \"strings_voices\": false}"; pids="$pids $GATED"
gate 6500 train _v11_m1_attn_ahead_rec32 "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"rec_all\": true, \"rec_dim\": 32}"; pids="$pids $GATED"
wait $pids
echo "$(date +%T) strings done" >> $Q/status.txt
