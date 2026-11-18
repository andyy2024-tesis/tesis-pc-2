#!/bin/bash
# After the two final DDSP runs: render their vocoder training pairs (GPU, minutes), start phase 2 of experiment 12
# (BigVGAN-v2 fine-tuned on real GAPS audio, real mel -> real audio), and measure + render the final runs on the CPU.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
until grep -q "all done" $Q/status.txt; do sleep 60; done
F1=e6/structured_v12_final_noise_strongD_rise
F2=e6/structured_v12_final_learnedB_noise_strongD_rise
for r in $F1 $F2; do
  echo "$(date +%T) start render $r" >> $Q/status.txt
  uv run python scripts/e12_render_ddsp.py $r > $R/e12_render_$(basename $r)_log.txt 2>&1
  echo "$(date +%T) end render $r exit $?" >> $Q/status.txt
done
p2() {  # phase 2: largest batch that fits
  for b in 8 6 4; do
    echo "$(date +%T) start phase2 batch $b" >> $Q/status.txt
    uv run python scripts/e12_vocoder.py --tag p2_real --input real --steps 40000 --batch $b --seg 32768 \
      --eval-every 5000 > $R/e12_p2_real_log.txt 2>&1
    s=$?; echo "$(date +%T) end phase2 batch $b exit $s" >> $Q/status.txt
    grep -q "out of memory" $R/e12_p2_real_log.txt || break
    rm -rf $R/e12/p2_real
  done
}
p2 &
L=$Q/evaluate_log.txt
export COMPARE_DEV=cpu
for r in structured_v12_final structured_v12_final_learnedB $F1 $F2; do
  echo "$(date +%T) evaluate $r" >> $L
  for s in measure_catalog noise_attack noise_vs_room harmonic_clicks; do uv run python analysis/$s.py "$r" >> $L 2>&1; done
  uv run python analysis/render_slide_audio.py "$r" "$r:nonoise" >> $L 2>&1
  uv run python analysis/render_stems.py "$r" >> $L 2>&1
  echo "$(date +%T) done $r" >> $L
done
echo "$(date +%T) eval done" >> $Q/status.txt
wait
echo "$(date +%T) phase2 done" >> $Q/status.txt
