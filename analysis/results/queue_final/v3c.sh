#!/bin/bash
# V3 x0res again (the first run crashed saving its model: a variable named like the output folder).
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
REN=structured_v12_final_learnedB_noise_strongD_rise
echo "$(date +%T) start v3_x0res (rerun)" >> $Q/status.txt
uv run python scripts/e12_diffusion.py --tag v3_x0res --kind shallow --param x0res --renders $REN --vocoder p0_pretrained \
  --steps 40000 --batch 32 --eval-every 10000 > $R/e12_v3_x0res_log.txt 2>&1
echo "$(date +%T) end v3_x0res exit $?" >> $Q/status.txt
