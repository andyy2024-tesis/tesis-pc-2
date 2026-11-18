#!/bin/bash
# CPU jobs after phase 2 (GPU busy with V1): excerpts through p2_real, diffusion runs re-evaluated with p2_real
cd ~/Documentos/tesis
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
S=analysis/results/queue_final/status.txt
echo "$(date +%T) start cpu_p2" >> $S
uv run python analysis/e12_refine_excerpts.py p2_real
uv run python analysis/e12_refine_excerpts.py p2_real e6-structured_v12_final_learnedB_noise_strongD_rise
uv run python analysis/e12_diffuse_excerpts.py v3_x0res p2_real e6-structured_v12_final_learnedB_noise_strongD_rise 20
uv run python analysis/e12_diffuse_excerpts.py v3b_flow p2_real e6-structured_v12_final_learnedB_noise_strongD_rise
uv run python scripts/e12_diffusion.py --tag v3_x0res --kind shallow --param x0res --renders structured_v12_final_learnedB_noise_strongD_rise --vocoder p2_real --depths 0,20 --unpaired 0 --eval-only
uv run python scripts/e12_diffusion.py --tag v3b_flow --kind flow --renders structured_v12_final_learnedB_noise_strongD_rise --vocoder p2_real --unpaired 0 --eval-only
echo "$(date +%T) end cpu_p2 marker_cpu_p2_done" >> $S
