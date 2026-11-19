#!/bin/bash
# 2026-11-19: page media for last night's runs (GPU idle): the 100-recording DDSP (excerpts, media, stems), then V1-100
# and V2 on the favourite's pairs (excerpts, media, parts).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
D=structured_v12_final_learnedB_data100
uv run python analysis/render_slide_audio.py $D $D:nonoise e6/${D}_noise_strongD_rise e6/${D}_noise_strongD_rise:nonoise 2>&1 | grep -vi warn | tail -2
uv run python analysis/make_listening_media.py --only $D $D-nonoise e6-${D}_noise_strongD_rise e6-${D}_noise_strongD_rise-nonoise 2>&1 | grep -vi warn | tail -2
uv run python analysis/render_stems.py $D e6/${D}_noise_strongD_rise 2>&1 | grep -vi warn | tail -2
uv run python analysis/e12_refine_excerpts.py v1_mix100 e6-${D}_noise_strongD_rise 2>&1 | grep -vi warn | tail -1
uv run python analysis/e12_refine_excerpts.py v2_fav20 e6-structured_v9_base_t32_noise_strongD 2>&1 | grep -vi warn | tail -1
uv run python analysis/e12_listen_media.py voc-v1_mix100-e6-${D}_noise_strongD_rise voc-v2_fav20-e6-structured_v9_base_t32_noise_strongD 2>&1 | grep -vi warn | grep -o "done [^ ]*"
echo "media done"
