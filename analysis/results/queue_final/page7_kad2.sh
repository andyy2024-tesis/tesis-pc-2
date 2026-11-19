#!/bin/bash
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
uv run python analysis/e12_kad_clips.py voc-pretrained44 voc-pretrained44-real 2>&1 | grep -vi warn | tail -2
KAD_BASES=voc-v2_inject20,voc-p2_real-real,voc-p2_real,voc-pretrained44-real,voc-pretrained44 external/kadtk/.venv/bin/python analysis/e12_kad.py \
  ddsp voc-v1_mix20 voc-v2_inject20 voc-s_stems20b voc-p2_real-real voc-p0_pretrained-real voc-p2_real voc-p4_v2_strings20 \
  voc-p5_real44 voc-p5_real44-real voc-pretrained44 voc-pretrained44-real 2>&1 | grep -E "vs |^  voc|^  ddsp|^  real"
echo "kad2 done"
