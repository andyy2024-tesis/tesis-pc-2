#!/bin/bash
# Experiment 12, the user's idea: the final DDSP (learned stiffness) trained on 100 recordings (the 20 + 80 more whose
# tab matched, analysis/results/e12_more_ids.txt) so the vocoder gets ~5 h of DDSP/real pairs instead of 0.73 h.
# 12000 steps (E9: 80 recordings plateau from step 8000), then the same adversarial recipe, then render the pairs.
# Started by night.sh after V1 (needs ~20 GB). First (added 2026-11-18 06:10) the two other ways of connecting the
# DDSP to the vocoder on the same 20-recording pairs as V1, so the three compare directly: V2 (the DDSP waveform
# injected into every upsampling stage, NSF-style) and S (the DDSP parts as extra mel channels).
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
REN=structured_v12_final_learnedB_noise_strongD_rise
voc() { # tag args...: from phase 2, 6000 steps on the final DDSP's pairs (as V1)
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e12_vocoder.py --tag $1 --init p2_real --renders $REN --steps 6000 --batch 8 --seg 32768 \
    --eval-every 2000 "${@:2}" > $R/e12_$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
}
voc v2_inject20 --input mix --inject mix
voc s_stems20 --input mix --stems
FB="$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"contact\": true, \"contact_init\": -8.0, \"delta_rise\": 20.0, \"strings\": true, \"rec_all\": true, \"rec_dim\": 32, \"rec_b\": \"learned\""
t100() {
  echo "$(date +%T) start _v12_final_learnedB_data100" >> $Q/status.txt
  uv run python scripts/e4_train.py --steps 12000 --eval-every 2000 --config "{$FB}" --tag _v12_final_learnedB_data100 \
    --extra-ids-file $R/e12_more_ids.txt > $R/e11_structured_v12_final_learnedB_data100_log.txt 2>&1
  echo "$(date +%T) end _v12_final_learnedB_data100 exit $?" >> $Q/status.txt
}
gate 6500 t100; wait $GATED
gate 8500 adv _v12_final_learnedB_data100; wait $GATED
r=e6/structured_v12_final_learnedB_data100_noise_strongD_rise
echo "$(date +%T) start render $r" >> $Q/status.txt
uv run python scripts/e12_render_ddsp.py $r > $R/e12_render_$(basename $r)_log.txt 2>&1
echo "$(date +%T) end render $r exit $?" >> $Q/status.txt
echo "$(date +%T) data100 done" >> $Q/status.txt
# The user's question: does the vocoder gain from ~7x more DDSP/real pairs? V1 as v1_mix20, on the 100-recording pairs
# (same 85 held-out windows: the 20 test recordings stay cut at 85%).
REN=structured_v12_final_learnedB_data100_noise_strongD_rise
[ -d $R/e12/renders/$REN ] && voc v1_mix100 --input mix
echo "$(date +%T) queue_final finished" >> $Q/status.txt
