#!/bin/bash
# Evaluate every experiment-11 run as soon as it ends (one at a time, on the CPU); stop after "all done" once every
# started job has ended and every finished one is evaluated.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
Q=analysis/results/queue_e11
touch $Q/evaluated.txt
while true; do
  for line in $(grep "exit 0" $Q/status.txt | sed 's/^[0-9:]* end //; s/ exit 0$//; s/ /:/g'); do
    case $line in
      adv:*) run="e6/${line#adv:}" ;;
      *) run="structured$line" ;;
    esac
    grep -qx "$run" $Q/evaluated.txt && continue
    bash $Q/evaluate.sh "$run"
    echo "$run" >> $Q/evaluated.txt
  done
  s=$(grep -c " start " $Q/status.txt); e=$(grep -c " end " $Q/status.txt); ok=$(grep -c "exit 0" $Q/status.txt)
  grep -q "all done" $Q/status.txt && grep -q "strings done" $Q/status.txt && [ "$s" -le "$e" ] && [ "$ok" -le "$(wc -l < $Q/evaluated.txt)" ] && break
  sleep 60
done
echo "$(date +%T) auto_eval finished" >> $Q/evaluate_log.txt
