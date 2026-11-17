#!/bin/bash
# T1's adversarial step, gated on memory (replaces the ungated tail of laneT1.sh): wait for T1's training to end.
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
while kill -0 2577900 2>/dev/null; do sleep 30; done
if [ -f $R/e4/structured_v11_m1_attn/model.pt ] && grep -q "4000" $R/e11_structured_v11_m1_attn_log.txt; then st=0; else st=1; fi
echo "$(date +%T) end _v11_m1_attn exit $st" >> $Q/status.txt
[ $st = 0 ] && { gate 8500 adv _v11_m1_attn; wait $GATED; }
