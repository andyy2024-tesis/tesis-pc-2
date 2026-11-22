#!/bin/bash
# 2026-11-22: page-8 media: the 100-recording DDSP at 5x (excerpts, media, stems), then the embedding / 5x / no-contact
# vocoders (excerpts through each vocoder, media, parts).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
X=e6-structured_v12_final_learnedB_data100x5_noise_strongD_rise
L=e6-structured_v12_final_learnedB_noise_strongD_rise
H=e6-structured_v12_final_learnedB_data100_noise_strongD_rise
N=e6-structured_v13_final_nocontact_noise_strongD_rise
uv run python analysis/render_slide_audio.py e6/${X#e6-} 2>&1 | grep -vi warn | tail -1
uv run python analysis/make_listening_media.py --only $X 2>&1 | grep -vi warn | tail -1
uv run python analysis/render_stems.py e6/${X#e6-} 2>&1 | grep -vi warn | tail -1
K=""
for a in "v2_recddsp20 $L" "v2_rectable20 $L" "v2_nocontact20 $N" "v2_x5 $X" "v2_recddspx5 $X" "v2_recddsp100 $H"; do
  set -- $a; echo "$(date +%T) excerpts $1 $2"
  uv run python analysis/e12_refine_excerpts.py $1 $2 2>&1 | grep -vi warn | tail -1
  K="$K voc-$1-$2"
done
uv run python analysis/e12_listen_media.py $K 2>&1 | grep -vi warn | grep -o "done [^ ]*"
echo "$(date +%T) media_p8 done" >> analysis/results/queue_final/status.txt
