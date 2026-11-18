#!/bin/bash
# V3b again with noise at the start (the first flow kept only 27% of the real mel's frame-to-frame detail: it learned
# the average correction). After the x0res rerun.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
REN=structured_v12_final_learnedB_noise_strongD_rise
until grep -q "end v3_x0res exit" $Q/status.txt; do sleep 30; done
echo "$(date +%T) start v3b_flow_noise" >> $Q/status.txt
uv run python scripts/e12_diffusion.py --tag v3b_flow_noise --kind flow --src-noise 0.3 --bridge 0.3 --renders $REN \
  --vocoder p0_pretrained --steps 40000 --batch 32 --eval-every 10000 > $R/e12_v3b_flow_noise_log.txt 2>&1
echo "$(date +%T) end v3b_flow_noise exit $?" >> $Q/status.txt
