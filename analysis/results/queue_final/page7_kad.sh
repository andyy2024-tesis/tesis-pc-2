#!/bin/bash
# KAD of the page-7 runs on the CPU (the GPU trains p5_v2_mix20_44k).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=10
echo "$(date +%T) clips"
uv run python analysis/e12_kad_clips.py voc-p4_v2_strings20 voc-p5_real44 voc-p5_real44-real 2>&1 | grep -vi warn
echo "$(date +%T) kad"
external/kadtk/.venv/bin/python analysis/e12_kad.py ddsp voc-v1_mix20 voc-v2_inject20 voc-s_stems20b voc-p2_real-real \
  voc-p0_pretrained-real voc-p2_real voc-p4_v2_strings20 voc-p5_real44 voc-p5_real44-real 2>&1 | grep -vi warn
echo "$(date +%T) kad done"
