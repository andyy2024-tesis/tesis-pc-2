#!/bin/bash
# 2026-11-19 night, replaces embed.sh (stopped before it started anything). The user: the 100-recording runs use the
# embedding type that wins on the 20-recording pairs, and every 100-recording vocoder gets the same training per
# recording as the 20-recording one (6000 steps on 20 -> 30000 on 100).
#   1. V2 + embedding on the final DDSP's 20-recording pairs: the DDSP's own (frozen) and a fresh table; KAD picks
#      the winner (PANNs, lower) into embed_choice.txt. The user can override it by ear: the file is read only at 3.
#   2. the 100-recording DDSP at 5x (20000 + 15000 adversarial steps), its pairs and KAD
#   3. V2 on those pairs with the winning embedding and without, 30000 steps each
#   4. V2 + the winning embedding on the first 100-recording DDSP's pairs, 30000 steps
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/lib.sh
until grep -q "v2_nocontact finished" $Q/status.txt; do sleep 60; done
run() { # tag renders steps args...
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e12_vocoder.py --tag $1 --init p2_real --renders $2 --steps $3 --batch 8 --seg 32768 \
    --eval-every $(( $3 / 3 )) --input mix --inject mix "${@:4}" > $R/e12_$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
  uv run python analysis/e12_kad_clips.py voc-$1 >> $Q/embed_kad.out 2>&1
}
kad() { # bases versions...
  CUDA_VISIBLE_DEVICES="" KAD_BASES=$1 external/kadtk/.venv/bin/python analysis/e12_kad.py ddsp "${@:2}" 2>&1 \
    | grep -v -i warn >> $Q/embed_kad.out
}
F=structured_v12_final_learnedB_noise_strongD_rise
H=structured_v12_final_learnedB_data100_noise_strongD_rise
T=_v12_final_learnedB_data100x5
X=structured${T}_noise_strongD_rise
# 1
run v2_recddsp20 $F 6000 --rec-emb ddsp
run v2_rectable20 $F 6000 --rec-emb table
( kad voc-v2_inject20 voc-v2_inject20 voc-v2_recddsp20 voc-v2_rectable20
  uv run python -c "
import json; v = json.load(open('$R/e12_kad.json'))['panns-wavegram-logmel vs real']['versions']
print('ddsp' if v['voc-v2_recddsp20']['kad'] <= v['voc-v2_rectable20']['kad'] else 'table')" > $Q/embed_choice.txt
  echo "$(date +%T) embed20 finished, pick $(cat $Q/embed_choice.txt)" >> $Q/status.txt ) &
PICK=$!
# 2
FB="$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"contact\": true, \"contact_init\": -8.0, \"delta_rise\": 20.0, \"strings\": true, \"rec_all\": true, \"rec_dim\": 32, \"rec_b\": \"learned\""
echo "$(date +%T) start $T" >> $Q/status.txt
uv run python scripts/e4_train.py --steps 20000 --eval-every 4000 --config "{$FB}" --tag $T \
  --extra-ids-file $R/e12_more_ids.txt > $R/e11_structured${T}_log.txt 2>&1
echo "$(date +%T) end $T exit $?" >> $Q/status.txt
echo "$(date +%T) start adv $X" >> $Q/status.txt
uv run python scripts/e6_adversarial.py --run structured$T --steps 15000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
  --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise > $R/e11_adv_${X}_log.txt 2>&1
echo "$(date +%T) end adv $X exit $?" >> $Q/status.txt
echo "$(date +%T) start render e6/$X" >> $Q/status.txt
uv run python scripts/e12_render_ddsp.py e6/$X > $R/e12_render_${X}_log.txt 2>&1
echo "$(date +%T) end render e6/$X exit $?" >> $Q/status.txt
uv run python analysis/e12_kad_clips.py model-e6-$X >> $Q/embed_kad.out 2>&1
( kad model-e6-$F,model-e6-$H model-e6-$F model-e6-$H model-e6-$X
  echo "$(date +%T) ddsp x5 finished" >> $Q/status.txt ) &
# 3
wait $PICK
C=$(cat $Q/embed_choice.txt)
echo "$(date +%T) using embedding $C" >> $Q/status.txt
run v2_rec${C}x5 $X 30000 --rec-emb $C
run v2_x5 $X 30000
wait
kad voc-v2_inject20,voc-v2_x5 voc-v2_inject20 voc-v2_x5 voc-v2_rec${C}x5
echo "$(date +%T) v2 x5 finished" >> $Q/status.txt
# 4
run v2_rec${C}100 $H 30000 --rec-emb $C
kad voc-v2_inject20,voc-v1_mix100 voc-v2_inject20 voc-v1_mix100 voc-v2_rec${C}100
echo "$(date +%T) v2 rec100 finished" >> $Q/status.txt
