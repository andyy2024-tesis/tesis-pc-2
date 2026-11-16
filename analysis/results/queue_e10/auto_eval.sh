#!/bin/bash
# Evaluate every experiment-10 run as soon as it ends (one at a time): evaluate.sh on each "end ... exit 0" line.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
Q=analysis/results/queue_e10
touch $Q/evaluated.txt
while true; do
  for line in $(cat $Q/status.txt $Q/status2.txt 2>/dev/null | grep "exit 0" | sed 's/^[0-9:]* end //; s/ exit 0$//; s/ /:/g'); do
    case $line in
      adv:*) run="e6/${line#adv:}" ;;
      *) run="structured$line" ;;
    esac
    grep -qx "$run" $Q/evaluated.txt && continue
    bash $Q/evaluate.sh "$run"
    echo "$run" >> $Q/evaluated.txt
  done
  grep -q "stage5 done" $Q/status2.txt 2>/dev/null && [ "$(grep -c 'exit 0' $Q/status.txt $Q/status2.txt | awk -F: '{s+=$2} END {print s}')" -le "$(wc -l < $Q/evaluated.txt)" ] && break
  sleep 60
done
echo "$(date +%T) auto_eval finished" >> $Q/evaluate_log.txt
