# Shared helpers for experiment-11 lanes. wait_mem MB: start only when the GPU has MB free (one lane at a time decides,
# then waits 150 s so the job it started has allocated before the next lane looks).
cd /home/sis.virtual.uniandes.edu.co/a.boteror/Documentos/tesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=analysis/results
Q=$R/queue_e11
M1B='"body_ir": true, "audio_feats": true, "note_timbre": true, "timbre_k": 32, "rich_burst": true, "room_floor": true'
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gate() { # MB command...: run command in the background once MB are free; returns its pid in $GATED
  exec 9>$Q/gpu.lock; flock 9
  while [ "$(free_mb)" -lt "$1" ]; do sleep 30; done
  "${@:2}" & GATED=$!
  sleep 150; flock -u 9
}
train() { # tag config
  echo "$(date +%T) start $1" >> $Q/status.txt
  uv run python scripts/e4_train.py --steps 4000 --eval-every 1000 --config "$2" --tag $1 > $R/e11_structured$1_log.txt 2>&1
  echo "$(date +%T) end $1 exit $?" >> $Q/status.txt
}
adv() { # base run (without "structured")
  echo "$(date +%T) start adv structured$1_noise_strongD_rise" >> $Q/status.txt
  uv run python scripts/e6_adversarial.py --run structured$1 --steps 3000 --d-ch 64 --d-lr-mult 2.0 --d-kind complex \
    --config '{"delta_rise": 20.0}' --tag _noise_strongD_rise > $R/e11_adv_structured$1_noise_strongD_rise_log.txt 2>&1
  echo "$(date +%T) end adv structured$1_noise_strongD_rise exit $?" >> $Q/status.txt
}
lane() { # tag config: train, then adversarial (each gated on memory)
  gate 6500 train $1 "$2"; wait $GATED
  gate 8500 adv $1; wait $GATED
}
