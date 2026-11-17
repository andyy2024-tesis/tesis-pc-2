#!/bin/bash
# T2: attention + 160 ms look-ahead (strums; the biLSTM's edge was seeing ahead)
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
lane _v11_m1_attn_ahead "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16}"
