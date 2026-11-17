#!/bin/bash
# R: T2 (attention + look-ahead) where every part that could depend on the guitar, microphone or room reads the
# recording (rec_all), then the adversarial recipe with the rise penalty. Starts once T4 is training (before the
# strings runs, which wait for T4's adversarial step).
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
until grep -q "start _v11_m1_attn_ahead_contact" $Q/status.txt; do sleep 60; done
lane _v11_m1_attn_ahead_rec "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"rec_all\": true}"
echo "$(date +%T) rec done" >> $Q/status.txt
