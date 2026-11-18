#!/bin/bash
# Final DDSP model (before the vocoder stage): M1 + timbre 32 + per-note attack + room floor, attention that looks
# ahead, contact part, strings from the tablature (one stiffness table for all guitars), the recording in every part
# with a 32-dim embedding; then the adversarial recipe with the rise penalty. Two versions of the guitar's stiffness:
# "none" = only the shared per-string table; "learned" = a per-string shift read from the embedding.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
FB="$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"contact\": true, \"contact_init\": -8.0, \"delta_rise\": 20.0, \"strings\": true, \"rec_all\": true, \"rec_dim\": 32"
lane _v12_final "{$FB, \"rec_b\": \"none\"}" & p1=$!
sleep 200
lane _v12_final_learnedB "{$FB, \"rec_b\": \"learned\"}" & p2=$!
wait $p1 $p2
echo "$(date +%T) all done" >> $Q/status.txt
