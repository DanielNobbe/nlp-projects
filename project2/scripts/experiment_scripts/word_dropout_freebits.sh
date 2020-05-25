#!/usr/bin/env bash
save_path="results$1/word_dropout_066_freebits_05"
python vae.py --model_save_path $save_path --word_dropout 0.66 --freebits 0.5
