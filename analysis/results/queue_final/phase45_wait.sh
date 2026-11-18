#!/bin/bash
# Starts phase45.sh once ddsp100_run.sh writes a new "queue_final finished" line (counted, so old lines never fire it).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
S=analysis/results/queue_final/status.txt
n0=$(grep -c "queue_final finished" $S)
until [ "$(grep -c 'queue_final finished' $S)" -gt "$n0" ]; do sleep 60; done
bash analysis/results/queue_final/phase45.sh
