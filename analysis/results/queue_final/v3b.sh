#!/bin/bash
# After the first V3 run: V3 with the residual clean-mel prediction, then V3b (flow from the DDSP mel to the real mel).
# Small nets beside phase 2 / V1; evaluated with the pretrained vocoder, re-evaluated with p2_real later (--eval-only).
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
REN=structured_v12_final_learnedB_noise_strongD_rise
until grep -q "end v3_shallow exit" $Q/status.txt; do sleep 30; done
echo "$(date +%T) start v3_x0res" >> $Q/status.txt
uv run python scripts/e12_diffusion.py --tag v3_x0res --kind shallow --param x0res --renders $REN --vocoder p0_pretrained \
  --steps 40000 --batch 32 --eval-every 10000 > $R/e12_v3_x0res_log.txt 2>&1
echo "$(date +%T) end v3_x0res exit $?" >> $Q/status.txt
echo "$(date +%T) start v3b_flow" >> $Q/status.txt
uv run python scripts/e12_diffusion.py --tag v3b_flow --kind flow --renders $REN --vocoder p0_pretrained \
  --steps 40000 --batch 32 --eval-every 10000 > $R/e12_v3b_flow_log.txt 2>&1
echo "$(date +%T) end v3b_flow exit $?" >> $Q/status.txt
