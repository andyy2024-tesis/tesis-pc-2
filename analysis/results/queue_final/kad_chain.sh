#!/bin/bash
# 2026-11-19: which change from T4adv (PANNs-KAD 1.99) to the final DDSP (3.22) made it less real? The adversarial
# DDSP models between the earlier favourite and the final one, by KAD; plus last night's V1-100 and V2 on the
# favourite's pairs.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
M="model-e6-structured_v10_m1_burst_noise_strongD model-e6-structured_v10_m1_burst_strings_noise_strongD
model-e6-structured_v10_m1_burst_floor_noise_strongD_rise model-e6-structured_v10_m1_burst_floor_strings_noise_strongD
model-e6-structured_v11_m1_attn_noise_strongD_rise model-e6-structured_v11_m1_attn_ahead_noise_strongD_rise
model-e6-structured_v11_m1_attn_ahead_xvoice_noise_strongD_rise model-e6-structured_v11_m1_attn_ahead_rec_noise_strongD_rise
model-e6-structured_v12_final_learnedB_data100_noise_strongD_rise"
uv run python analysis/e12_kad_clips.py $M voc-v1_mix100 voc-v2_fav20 2>&1 | grep -vi warn | grep clips
OLD="model-e6-structured_v9_base_t32_noise_strongD model-e6-structured_v11_m1_attn_ahead_contact_noise_strongD_rise
model-e6-structured_v12_final_noise_strongD_rise model-e6-structured_v12_final_learnedB_noise_strongD_rise"
CUDA_VISIBLE_DEVICES="" KAD_BASES=voc-v2_inject20,model-e6-structured_v11_m1_attn_ahead_contact_noise_strongD_rise,model-e6-structured_v12_final_learnedB_noise_strongD_rise,model-e6-structured_v10_m1_burst_noise_strongD,model-e6-structured_v11_m1_attn_ahead_noise_strongD_rise \
  external/kadtk/.venv/bin/python analysis/e12_kad.py ddsp voc-v1_mix20 voc-v2_inject20 $OLD $M voc-v1_mix100 voc-v2_fav20 2>&1 | grep -E "vs |^  "
echo "chain done"
