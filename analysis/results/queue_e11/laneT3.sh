#!/bin/bash
# T3: T2 + attention across voices at every frame
source /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis/analysis/results/queue_e11/lib.sh
lane _v11_m1_attn_ahead_xvoice "{$M1B, \"backbone\": \"attn\", \"attn_ahead\": 16, \"cross_voice\": true}"
