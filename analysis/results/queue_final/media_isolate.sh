#!/bin/bash
# 2026-11-19 evening: page media for the two isolating DDSPs (adversarial versions): excerpts, media, stems.
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
for B in structured_v13_ahead_strings structured_v13_final_nocontact; do
  R=e6/${B}_noise_strongD_rise
  uv run python analysis/render_slide_audio.py $R 2>&1 | grep -vi warn | tail -1
  uv run python analysis/make_listening_media.py --only e6-${B}_noise_strongD_rise 2>&1 | grep -vi warn | tail -1
  uv run python analysis/render_stems.py $R 2>&1 | grep -vi warn | tail -1
done
echo "$(date +%T) media_isolate done" >> analysis/results/queue_final/status.txt
