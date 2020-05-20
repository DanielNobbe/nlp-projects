#!/usr/bin/env bash
python vae.py --model_save_path word_dropout_freebits_mdr --word_dropout 0.66 --freebits 0.5 --MDR 10.0 --save_every 500
