#!/bin/bash
# Exit (waking Claude) when a job fails, an evaluation finishes, or auto_eval ends.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11
c() { echo "$(grep -cE 'exit [1-9]' status.txt) $(grep -c ' done ' evaluate_log.txt 2>/dev/null) $(grep -c finished evaluate_log.txt 2>/dev/null)"; }
s0=$(c)
while [ "$(c)" = "$s0" ]; do sleep 60; done
echo "event: $(c) (was $s0)"; tail -3 status.txt; tail -2 evaluate_log.txt 2>/dev/null
