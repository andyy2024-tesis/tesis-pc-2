#!/bin/bash
# Experiment 12, phases 4 and 5, after the night queue (started by phase45_wait.sh when ddsp100_run.sh writes
# "queue_final finished"). The winner is picked from the held-out loss at the last evaluation; the user judges by ear
# and differences under 0.05 are ties, so a route only wins over V1 (the simplest) if it is at least 0.05 better.
# Same for the pairs: the 100-recording ones only if V1 on them is at least 0.05 better than on the 20.
# Phase 4: the winner fed the harmonic part only ("strings": the simple DDSP that sends harmonics alone).
# Phase 5: BigVGAN-v2 44 kHz fine-tuned on real audio (as phase 2), then the winner at 44.1 kHz from it.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
last() { grep "EVAL step" $R/e12/$1/log.txt 2>/dev/null | tail -1 | awk '{print $NF}'; }
pick=$(python3 - <<EOF
v1, v2, s = "$(last v1_mix20)", "$(last v2_inject20)", "$(last s_stems20)"
v100 = "$(last v1_mix100)"
f = lambda t: float(t) if t else 99.0
best = min([("v1", f(v1)), ("v2", f(v2)), ("s", f(s))], key=lambda kv: kv[1])
route = best[0] if best[1] <= f(v1) - 0.05 else "v1"
pairs = "100" if f(v100) <= f(v1) - 0.05 else "20"
print(route, pairs)
EOF
)
read route pairs <<< "$pick"
REN=structured_v12_final_learnedB_noise_strongD_rise
[ "$pairs" = 100 ] && REN=structured_v12_final_learnedB_data100_noise_strongD_rise
echo "$(date +%T) phase45: route $route, pairs $pairs ($REN)" >> $Q/status.txt
case $route in
  v2) full="--input mix --inject mix"; simple="--input strings --inject strings" ;;
  s) full="--input mix --stems"; simple="--input strings" ;;  # S's point is all the parts: its simple form is V1's
  *) full="--input mix"; simple="--input strings" ;;
esac
run() { # tag args...
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e12_vocoder.py --tag $1 --batch 8 --seg 32768 --eval-every 2000 "${@:2}" > $R/e12_$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
}
run p4_${route}_strings$pairs --init p2_real --renders $REN --steps 6000 $simple
run p5_real44 --sr 44100 --init pretrained --input real --steps 8000
[ -f $R/e12/p5_real44/generator.pt ] || run p5_real44 --sr 44100 --init pretrained --input real --steps 8000 --batch 4
[ -f $R/e12/p5_real44/generator.pt ] && run p5_${route}_mix${pairs}_44k --sr 44100 --init p5_real44 --renders $REN --steps 6000 $full
echo "$(date +%T) phase45 finished" >> $Q/status.txt
