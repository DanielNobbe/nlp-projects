#!/usr/bin/env bash
save_path="results$1/mdr10"
python vae.py --model_save_path $save_path --MDR 10.0
