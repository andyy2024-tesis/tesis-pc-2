#!/bin/bash
# Strings runs (why strings sound synthetic), from M1 + timbre 32 + per-note attack + room background + strings (GRU,
# 5.70): S1 the old stiffness law over pitch, S5 stiffness per guitar and string, S3 a decay that knows string, fret and
# guitar, S2 the old voice assignment. No adversarial. Start once T4 is in its adversarial step.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
M1BS="$M1B, \"strings\": true"
until grep -q "start adv structured_v11_m1_attn_ahead_contact" $Q/status.txt; do sleep 60; done
echo "$(date +%T) strings queued" >> $Q/status.txt
gate 6500 train _v11_m1_strings_oldB "{$M1BS, \"strings_b\": \"pitch\"}"; p1=$GATED
gate 6500 train _v11_m1_strings_guitarB "{$M1BS, \"strings_b\": \"guitar\"}"; p2=$GATED
gate 6500 train _v11_m1_strings_decay "{$M1BS, \"decay_cond\": true}"; p3=$GATED
gate 6500 train _v11_m1_strings_oldvoices "{$M1BS, \"strings_voices\": false}"; p4=$GATED
wait $p1 $p2 $p3 $p4
echo "$(date +%T) strings done" >> $Q/status.txt
