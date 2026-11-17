#!/bin/bash
# Re-run, once, any experiment-11 job that ended with a non-zero exit (e.g. out of GPU memory), gated on 10 GB free;
# a re-trained run then gets its adversarial step too.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
touch $Q/retried.txt
cfg() { case $1 in
  _v11_m1_attn) echo "{$M1B, \"backbone\": \"attn\"}" ;;
  _v11_m1_attn_ahead) echo "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16}" ;;
  _v11_m1_attn_ahead_xvoice) echo "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"cross_voice\": true}" ;;
  _v11_m1_attn_ahead_contact) echo "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"contact\": true, \"contact_init\": -8.0, \"delta_rise\": 20.0}" ;;
  _v11_m1_strings_oldB) echo "{$M1B, \"strings\": true, \"strings_b\": \"pitch\"}" ;;
  _v11_m1_strings_guitarB) echo "{$M1B, \"strings\": true, \"strings_b\": \"guitar\"}" ;;
  _v11_m1_strings_decay) echo "{$M1B, \"strings\": true, \"decay_cond\": true}" ;;
  _v11_m1_strings_oldvoices) echo "{$M1B, \"strings\": true, \"strings_voices\": false}" ;;
esac; }
while true; do
  grep -E " end .* exit [1-9]" $Q/status.txt | grep -v stopped | sed 's/^[0-9:]* end //; s/ exit [0-9]*$//' | while read -r job; do
    grep -qxF "$job" $Q/retried.txt && continue
    echo "$job" >> $Q/retried.txt
    echo "$(date +%T) retry $job" >> $Q/status.txt
    case $job in
      "adv structured_v10_m1_burst_floor_noise_strongD_rise")
        gate 10000 bash -c 'source analysis/results/queue_e11/lib.sh; echo "$(date +%T) start adv structured_v10_m1_burst_floor_noise_strongD_rise" >> $Q/status.txt; uv run python scripts/e6_adversarial.py --run structured_v10_m1_burst_floor --steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex --config "{\"delta_rise\": 20.0}" --tag _noise_strongD_rise > $R/e11_adv_structured_v10_m1_burst_floor_noise_strongD_rise_log.txt 2>&1; echo "$(date +%T) end adv structured_v10_m1_burst_floor_noise_strongD_rise exit $?" >> $Q/status.txt'
        wait $GATED ;;
      adv\ *) b=${job#adv structured}; b=${b%_noise_strongD_rise}; gate 10000 adv $b; wait $GATED ;;
      *) gate 10000 train $job "$(cfg $job)"; wait $GATED
         case $job in _v11_m1_strings_*) ;; *) grep -q "end $job exit 0" $Q/status.txt && { gate 10000 adv $job; wait $GATED; } ;; esac ;;
    esac
  done
  s=$(grep -c " start " $Q/status.txt); e=$(grep -c " end " $Q/status.txt)
  grep -q "all done" $Q/status.txt && grep -q "strings done" $Q/status.txt && [ "$s" -le "$e" ] && break
  sleep 60
done
