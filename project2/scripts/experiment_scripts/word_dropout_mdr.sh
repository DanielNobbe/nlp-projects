#!/usr/bin/env bash
save_path="results$1/word_dropout_066_mdr_10"
python vae.py --model_save_path $save_path --word_dropout 0.66 --MDR 10.0
