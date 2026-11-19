#!/bin/bash
# 2026-11-19, the user: which piece of the final DDSP lowers its KAD (T4adv 1.99 -> final 3.22)? Two DDSPs the chain
# lacked, trained and fine-tuned exactly as the final (4000 steps, then the adversarial recipe), in parallel lanes:
#   _v13_ahead_strings     attention ahead + strings (no contact, no recording-everywhere)
#   _v13_final_nocontact   the final DDSP without contact (strings + recording everywhere 32 + learned stiffness)
# then KAD of both against the same 85 windows, paired with T4adv and the final DDSP.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
A="$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"delta_rise\": 20.0, \"strings\": true"
lane _v13_ahead_strings "{$A}" & p1=$!
sleep 200
lane _v13_final_nocontact "{$A, \"rec_all\": true, \"rec_dim\": 32, \"rec_b\": \"learned\"}" & p2=$!
wait $p1 $p2
echo "$(date +%T) isolate trained" >> $Q/status.txt
M="model-e6-structured_v13_ahead_strings_noise_strongD_rise model-e6-structured_v13_final_nocontact_noise_strongD_rise"
uv run python analysis/e12_kad_clips.py $M > $Q/isolate_kad.out 2>&1
CUDA_VISIBLE_DEVICES="" KAD_BASES=model-e6-structured_v11_m1_attn_ahead_contact_noise_strongD_rise,model-e6-structured_v12_final_learnedB_noise_strongD_rise,model-e6-structured_v11_m1_attn_ahead_noise_strongD_rise,model-e6-structured_v10_m1_burst_strings_noise_strongD \
  external/kadtk/.venv/bin/python analysis/e12_kad.py ddsp voc-v1_mix20 model-e6-structured_v11_m1_attn_ahead_noise_strongD_rise \
  model-e6-structured_v11_m1_attn_ahead_contact_noise_strongD_rise model-e6-structured_v12_final_learnedB_noise_strongD_rise \
  model-e6-structured_v10_m1_burst_strings_noise_strongD $M 2>&1 | grep -E "vs |^  " >> $Q/isolate_kad.out
echo "$(date +%T) isolate finished" >> $Q/status.txt
