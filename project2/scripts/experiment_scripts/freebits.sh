#!/usr/bin/env bash
save_path="results/freebits_05"
python vae.py --model_save_path $save_path --freebits 0.5
