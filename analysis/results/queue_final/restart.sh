#!/bin/bash
# The machine rebooted at 21:24 on 2026-11-18 and killed phase45.sh during p5_v2_mix20_44k (step ~2000 of 6000; the
# vocoder has no exact resume, so it starts again from zero to stay comparable with the 24 kHz V2). Then the
# data100 steps that were appended to phase45.sh.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
run() { # tag args...
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e12_vocoder.py --tag $1 --batch 8 --seg 32768 --eval-every 2000 "${@:2}" > $R/e12_$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
}
echo "$(date +%T) restart after reboot" >> $Q/status.txt
REN=structured_v12_final_learnedB_noise_strongD_rise
run p5_v2_mix20_44k --sr 44100 --init p5_real44 --renders $REN --steps 6000 --input mix --inject mix
echo "$(date +%T) phase45 finished" >> $Q/status.txt
adv _v12_final_learnedB_data100
r=e6/structured_v12_final_learnedB_data100_noise_strongD_rise
if [ -f $R/$r/model.pt ]; then
  echo "$(date +%T) start render $r" >> $Q/status.txt
  uv run python scripts/e12_render_ddsp.py $r > $R/e12_render_$(basename $r)_log.txt 2>&1
  echo "$(date +%T) end render $r exit $?" >> $Q/status.txt
fi
REN=structured_v12_final_learnedB_data100_noise_strongD_rise
[ -d $R/e12/renders/$REN ] && run v1_mix100 --init p2_real --renders $REN --steps 6000 --input mix
echo "$(date +%T) data100 requeue finished" >> $Q/status.txt
# 2026-11-19, the user: V2 on the pairs of the DDSP that KAD finds most real (M1 + timbre 32 + adv, PANNs 1.54 vs 3.22
# for the final DDSP), even though by ear the user prefers the final DDSP. Same recipe as v2_inject20.
r=e6/structured_v9_base_t32_noise_strongD
echo "$(date +%T) start render $r" >> $Q/status.txt
uv run python scripts/e12_render_ddsp.py $r > $R/e12_render_$(basename $r)_log.txt 2>&1
echo "$(date +%T) end render $r exit $?" >> $Q/status.txt
REN=structured_v9_base_t32_noise_strongD
[ -d $R/e12/renders/$REN ] && run v2_fav20 --init p2_real --renders $REN --steps 6000 --input mix --inject mix
echo "$(date +%T) v2_fav20 finished" >> $Q/status.txt
