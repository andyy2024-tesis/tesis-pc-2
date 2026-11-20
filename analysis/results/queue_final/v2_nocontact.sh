#!/bin/bash
# 2026-11-19 evening, the user: V2 on the pairs of the final DDSP retrained without contact (KAD 1.98 vs 3.22 for the
# final DDSP), "just to see the result". Same recipe as v2_inject20. Then its KAD against V2 on the final DDSP.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
run() { # tag args...
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e12_vocoder.py --tag $1 --batch 8 --seg 32768 --eval-every 2000 "${@:2}" > $R/e12_$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
}
r=e6/structured_v13_final_nocontact_noise_strongD_rise
echo "$(date +%T) start render $r" >> $Q/status.txt
uv run python scripts/e12_render_ddsp.py $r > $R/e12_render_$(basename $r)_log.txt 2>&1
echo "$(date +%T) end render $r exit $?" >> $Q/status.txt
REN=structured_v13_final_nocontact_noise_strongD_rise
[ -d $R/e12/renders/$REN ] && run v2_nocontact20 --init p2_real --renders $REN --steps 6000 --input mix --inject mix
uv run python analysis/e12_kad_clips.py voc-v2_nocontact20 > $Q/v2_nocontact_kad.out 2>&1
CUDA_VISIBLE_DEVICES="" KAD_BASES=voc-v2_inject20,model-e6-$REN external/kadtk/.venv/bin/python analysis/e12_kad.py \
  ddsp voc-v1_mix20 voc-v2_inject20 model-e6-$REN voc-v2_nocontact20 2>&1 | grep -v -i warn >> $Q/v2_nocontact_kad.out
echo "$(date +%T) v2_nocontact finished" >> $Q/status.txt
