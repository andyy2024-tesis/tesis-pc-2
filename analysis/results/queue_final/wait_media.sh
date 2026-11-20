#!/bin/bash
until grep -q "media_isolate done" /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/status.txt; do sleep 30; done
tail -8 /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_final/media_isolate.out
