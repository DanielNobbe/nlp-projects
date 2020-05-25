#!/usr/bin/env bash
save_path="results$1/vanilla"
python vae.py --model_save_path $save_path
