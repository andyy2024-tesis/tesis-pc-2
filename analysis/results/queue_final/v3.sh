#!/bin/bash
# V3: shallow diffusion on the mel, conditioned on the final DDSP's mel; runs beside phase 2 (small net). Evaluated
# with the pretrained vocoder until phase 2 is done (re-evaluated with p2_real afterwards).
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
echo "$(date +%T) start v3_shallow" >> $Q/status.txt
uv run python scripts/e12_diffusion.py --tag v3_shallow --kind shallow --renders structured_v12_final_learnedB_noise_strongD_rise \
  --vocoder p0_pretrained --steps 30000 --batch 32 --eval-every 10000 > $R/e12_v3_shallow_log.txt 2>&1
echo "$(date +%T) end v3_shallow exit $?" >> $Q/status.txt
