#!/bin/bash
# Experiment 12, one job after another (each needs ~20 GB, so nothing else big can share the GPU):
# 1. phase 2: BigVGAN-v2 24 kHz fine-tuned on real GAPS audio (gradient checkpointing; 1.3 s/step, so 8000 steps);
# 2. V1: DDSP mel -> real audio on the final DDSP's 20-recording pairs, from phase 2;
# 3. the 100-recording DDSP (then its adversarial fine-tune and its pair renders).
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
REN=structured_v12_final_learnedB_noise_strongD_rise
echo "$(date +%T) start p2_real (8000 steps)" >> $Q/status.txt
uv run python scripts/e12_vocoder.py --tag p2_real --input real --steps 8000 --batch 8 --seg 32768 \
  --eval-every 2000 > $R/e12_p2_real_log.txt 2>&1
echo "$(date +%T) end p2_real exit $?" >> $Q/status.txt
[ -f $R/e12/p2_real/generator.pt ] || exit 1
echo "$(date +%T) start v1_mix20" >> $Q/status.txt
uv run python scripts/e12_vocoder.py --tag v1_mix20 --init p2_real --input mix --renders $REN --steps 6000 --batch 8 \
  --seg 32768 --eval-every 2000 > $R/e12_v1_mix20_log.txt 2>&1
echo "$(date +%T) end v1_mix20 exit $?" >> $Q/status.txt
bash $Q/ddsp100_run.sh
