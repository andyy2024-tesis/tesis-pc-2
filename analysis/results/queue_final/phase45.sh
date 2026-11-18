#!/bin/bash
# Experiment 12, phases 4 and 5, after the night queue (started by phase45_wait.sh when ddsp100_run.sh writes
# "queue_final finished"). The winner is picked by KAD (below); the user judges by ear and may override.
# Phase 4: the winner fed the harmonic part only ("strings": the simple DDSP that sends harmonics alone).
# Phase 5: BigVGAN-v2 44 kHz fine-tuned on real audio (as phase 2), then the winner at 44.1 kHz from it.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
# S again first (the first run, s_stems20, diverged: raw log-mels of silent parts; fixed in Refiner.forward)
echo "$(date +%T) start s_stems20b" >> $Q/status.txt
uv run python scripts/e12_vocoder.py --tag s_stems20b --init p2_real --renders structured_v12_final_learnedB_noise_strongD_rise \
  --input mix --stems --steps 6000 --batch 8 --seg 32768 --eval-every 2000 > $R/e12_s_stems20b_log.txt 2>&1
echo "$(date +%T) end s_stems20b exit $?" >> $Q/status.txt
# Winner by KAD (analysis/e12_kad.py; decided with the user 2026-11-18: MSS misranks vocoders, KAD replaces it).
# A route replaces V1 only if its PANNs-KAD to the real windows is lower than V1's with the whole 95% interval of the
# paired difference below 0, and EnCodec-KAD does not say it is worse (interval of the difference not above 0).
# Same rule for the pairs: the 100-recording ones only if V1 on them beats V1 on the 20.
new=""
for r in s_stems20b v1_mix100; do [ -f $R/e12/$r/generator.pt ] && new="$new voc-$r"; done
if [ -n "$new" ]; then
  CUDA_VISIBLE_DEVICES="" uv run python analysis/e12_kad_clips.py $new > $Q/kad45_clips.out 2>&1
  CUDA_VISIBLE_DEVICES="" external/kadtk/.venv/bin/python analysis/e12_kad.py ddsp voc-v1_mix20 voc-v2_inject20 \
    $new > $Q/kad45.out 2>&1
fi
pick=$(python3 - <<'EOF2'
import json
try:
    k = json.load(open("analysis/results/e12_kad.json"))
    p, e = k["panns-wavegram-logmel vs real"]["versions"], k["encodec-emb vs real"]["versions"]
except Exception:
    p, e = {}, {}
def beats_v1(v):
    return v in p and p[v]["minus_voc-v1_mix20_ci"][1] < 0 and e[v]["minus_voc-v1_mix20_ci"][0] <= 0
cands = [v for v in ("voc-v2_inject20", "voc-s_stems20b") if beats_v1(v)]
route = {"voc-v2_inject20": "v2", "voc-s_stems20b": "s"}[min(cands, key=lambda v: p[v]["kad"])] if cands else "v1"
print(route, "100" if beats_v1("voc-v1_mix100") else "20")
EOF2
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
