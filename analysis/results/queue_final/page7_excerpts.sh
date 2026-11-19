#!/bin/bash
# Page-7 excerpts on the CPU (the GPU trains p5_v2_mix20_44k): S rerun, phase 4, phase-5 real-audio vocoder.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
L=e6-structured_v12_final_learnedB_noise_strongD_rise
for args in "s_stems20b $L" "p4_v2_strings20 $L" "p5_real44" "p5_real44 $L"; do
  echo "$(date +%T) excerpts $args"
  uv run python analysis/e12_refine_excerpts.py $args 2>&1 | grep -vi warn | tail -2
done
echo "$(date +%T) excerpts done"
