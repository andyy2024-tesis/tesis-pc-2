#!/bin/bash
# 2026-11-19 night, the user: a recording embedding for the vocoder (scripts/e12_vocoder.py --rec-emb), all V2 (mix +
# injected DDSP waveform, 6000 steps from p2_real, as v2_inject20):
#   v2_recddsp100  the 100-recording DDSP's pairs, the DDSP's own embedding (frozen) -> vs V2 on 20 (KAD 1.35)
#   v2_recddsp20   the final DDSP's 20-recording pairs, the DDSP's own embedding (frozen)
#   v2_rectable20  the same pairs, a fresh table learned by the vocoder
# then the 100-recording DDSP with 5x the 20-recording budget per recording (20000 steps + 15000 adversarial; the first
# one had 12000 + 3000) and its pairs. V2 on those pairs, with and without the better embedding, once it is picked.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
until grep -q "v2_nocontact finished" $Q/status.txt; do sleep 60; done
run() { # tag renders args...
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e12_vocoder.py --tag $1 --init p2_real --renders $2 --steps 6000 --batch 8 --seg 32768 \
    --eval-every 2000 --input mix --inject mix "${@:3}" > $R/e12_$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
  uv run python analysis/e12_kad_clips.py voc-$1 >> $Q/embed_kad.out 2>&1
}
F=structured_v12_final_learnedB_noise_strongD_rise
H=structured_v12_final_learnedB_data100_noise_strongD_rise
run v2_recddsp100 $H --rec-emb ddsp
run v2_recddsp20 $F --rec-emb ddsp
run v2_rectable20 $F --rec-emb table
CUDA_VISIBLE_DEVICES="" KAD_BASES=voc-v2_inject20,voc-v1_mix100 external/kadtk/.venv/bin/python analysis/e12_kad.py \
  ddsp voc-v1_mix20 voc-v2_inject20 voc-v1_mix100 voc-v2_recddsp100 voc-v2_recddsp20 voc-v2_rectable20 2>&1 \
  | grep -v -i warn >> $Q/embed_kad.out
echo "$(date +%T) embed finished" >> $Q/status.txt
FB="$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"contact\": true, \"contact_init\": -8.0, \"delta_rise\": 20.0, \"strings\": true, \"rec_all\": true, \"rec_dim\": 32, \"rec_b\": \"learned\""
T=_v12_final_learnedB_data100x5
echo "$(date +%T) start $T" >> $Q/status.txt
uv run python scripts/e4_train.py --steps 20000 --eval-every 4000 --config "{$FB}" --tag $T \
  --extra-ids-file $R/e12_more_ids.txt > $R/e11_structured${T}_log.txt 2>&1
echo "$(date +%T) end $T exit $?" >> $Q/status.txt
echo "$(date +%T) start adv structured${T}_noise_strongD_rise" >> $Q/status.txt
uv run python scripts/e6_adversarial.py --run structured$T --steps 15000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
  --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise > $R/e11_adv_structured${T}_noise_strongD_rise_log.txt 2>&1
echo "$(date +%T) end adv structured${T}_noise_strongD_rise exit $?" >> $Q/status.txt
r=e6/structured${T}_noise_strongD_rise
echo "$(date +%T) start render $r" >> $Q/status.txt
uv run python scripts/e12_render_ddsp.py $r > $R/e12_render_$(basename $r)_log.txt 2>&1
echo "$(date +%T) end render $r exit $?" >> $Q/status.txt
uv run python analysis/e12_kad_clips.py model-e6-structured${T}_noise_strongD_rise >> $Q/embed_kad.out 2>&1
echo "$(date +%T) ddsp x5 finished" >> $Q/status.txt
