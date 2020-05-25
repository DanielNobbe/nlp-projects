#!/usr/bin/env bash
save_path="results/word_dropout_066"
python vae.py --model_save_path $save_path --word_dropout 0.66
