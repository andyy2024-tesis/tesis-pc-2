#!/bin/bash
# T4: T2 + contact part trained from scratch (starts audible) with the rise penalty during training too, so fast high
# events can't go to the harmonics and must go to the contact part.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
while kill -0 2577234 2>/dev/null; do sleep 30; done  # after A0's adversarial run
lane _v11_m1_attn_ahead_contact "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"contact\": true, \"contact_init\": -8.0, \"delta_rise\": 20.0}"
echo "$(date +%T) all done" >> $Q/status.txt
