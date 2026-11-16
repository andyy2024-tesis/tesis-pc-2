#!/bin/bash
# Lane B again (06:00): the contact fine-tune started with its contact part far too loud (loss 12.4 before training) and
# was stopped. First the favourite's recipe on M1 + timbre 32 + per-note attack, then the contact fine-tune with the
# contact part starting silent and a 30x learning rate for the new parts. Ends the night's queue when lanes A and C
# (left running from queue2.sh) are done too.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e10
adv() { # base run, config overrides, tag
  echo "$(date +%T) start adv $1$3" >> $Q/status2.txt
  uv run python scripts/e6_adversarial.py --run $1 --steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
    --config "$2" --tag "$3" > $R/e10_adv_$1$3_log.txt 2>&1
  echo "$(date +%T) end adv $1$3 exit $?" >> $Q/status2.txt
}
adv structured_v10_m1_burst '{}' _noise_strongD
adv structured_v9_t32_no_gate '{"contact": true, "delta_rise": 20.0}' _noise_strongD_contact_rise
while kill -0 576825 2>/dev/null || kill -0 576827 2>/dev/null; do sleep 30; done
echo "$(date +%T) all done" >> $Q/status.txt
echo "$(date +%T) stage2 done" >> $Q/status2.txt
