# Mitigating Posterior Collapse in Variational Autoencoders for Text

## Requirements
Python 3.6

Install the requried packages:
```
pip install -r requirements.txt
```


## Training
To get our results, run the scripts in the `experiment_scripts` folder:

*Simple RNN model without VAE:*
```
./experiment_scripts/rnn.sh
```

*Vanilla Sentence VAE:*
```
./experiment_scripts/vanilla_vae.sh
```

*Sentence VAE with word dropout:*
```
./experiment_scripts/word_dropout.sh
```

*Sentence VAE with Free Bits:*
```
./experiment_scripts/freebits.sh
```

*Sentence VAE with Minimum Desired Rate:*
```
./experiment_scripts/minimum_desired_rate.sh
```

*Sentence VAE with both word dropout and free bits:*
```
./experiment_scripts/word_dropout_freebits.sh
```

*Sentence VAE with both word dropout and minimum desired rate:*
```
./experiment_scripts/word_dropout_mdr.sh
```

The results of these experiments are written to their corresponding folder (see the content of the scripts)


## Results

See the included ```results.ipynb``` 

## Additional parameters:
Additional parameters can be specified for the vae.py script. To see all of these you can run:

```
python vae.py --help
```

The output of this command is:

```

usage: vae.py [-h] [--data_path DATA_PATH] [-d {cuda,cpu}] [-ne NUM_EPOCHS]
              [-sbt BATCH_SIZE_TRAIN] [-sbv BATCH_SIZE_VALID]
              [-lr LEARNING_RATE] [-nl NUM_LAYERS] [-se EMBEDDING_SIZE]
              [-sh HIDDEN_SIZE] [-sl LATENT_SIZE] [-wd WORD_DROPOUT]
              [-v PRINT_EVERY] [-tb] [-m MODEL_SAVE_PATH] [-si SAVE_EVERY]
              [-es EARLY_STOPPING_PATIENCE] [-fb FREEBITS] [-mdr MDR]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Folder for the Penn Treebank dataset. This folder
                        should contain the 3 files of the provided data:
                        train, val and test. Default: ../Data/Dataset
  -d {cuda,cpu}, --device {cuda,cpu}
                        The device to use (either cpu or cuda). Default: gpu
  -ne NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Maximum number of epochs to train for. Default: 10
  -sbt BATCH_SIZE_TRAIN, --batch_size_train BATCH_SIZE_TRAIN
                        Batch size to use for training. Default: 32
  -sbv BATCH_SIZE_VALID, --batch_size_valid BATCH_SIZE_VALID
                        Batch size to use for validation. Default: 64
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for the optimizer. Default: 0.001
  -nl NUM_LAYERS, --num_layers NUM_LAYERS
                        Number of layers for the recurrent models. Default: 1
  -se EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
                        Embedding size. Default: 300
  -sh HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Hidden size. Default: 256
  -sl LATENT_SIZE, --latent_size LATENT_SIZE
                        Latent size. Default: 16
  -wd WORD_DROPOUT, --word_dropout WORD_DROPOUT
                        Word dropout keep probability. Default: 1.0 (keep
                        every word)
  -v PRINT_EVERY, --print_every PRINT_EVERY
                        Status printing interval. Default: 50
  -tb, --tensorboard_logging
  -m MODEL_SAVE_PATH, --model_save_path MODEL_SAVE_PATH
                        Folder to save the pretrained models to. If doesn't
                        exist, it is created. Default: models
  -si SAVE_EVERY, --save_every SAVE_EVERY
                        Checkpointing interval in iterations. Default: 500
  -es EARLY_STOPPING_PATIENCE, --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Early stopping patience. Default: 2
  -fb FREEBITS, --freebits FREEBITS
                        Free Bits parameter (if not provided, free bits method
                        is not used for training)
  -mdr MDR, --MDR MDR   MDR minimum rate (if not provided, MDR won't be used
                        for training)

```
