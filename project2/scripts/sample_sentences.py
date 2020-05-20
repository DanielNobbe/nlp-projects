from vae import Encoder, Sampler, WordDropout, Decoder, Lagrangian, SentenceVAE, evaluate

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import padded_collate, get_datasets


def sample_sentence(model, tokenizer, number):
    
    mean = torch.zeros(model.sampler.latent_size)
    std = torch.ones(model.sampler.latent_size)
    z = model.sampler(mean, std, 0) # Sample from standard Gaussian
    # z = std
    # packed = model._embed_and_pack(z, lengths)
    
    h = model.decoder.l2h(z)
    h = h.reshape(1, model.decoder.num_layers, model.decoder.hidden_size)
    h = h.transpose(0, 1)

    sentence = []
    token = ''
    word = model.embedding(torch.Tensor([tokenizer.bos_token_id]).long()).unsqueeze(dim=0)
    # print(word.shape)

    while token != tokenizer.eos_token:
        word, h = model.decoder.rnn(word, h)
        # print("A: ", word.shape)
        token_index = model.decoded2vocab(word).argmax(dim=2).squeeze()
        # print("B: ", token_index)
        token = tokenizer.decode(token_index, skip_special_tokens=False)
        # token = token
        word = model.embedding(token_index).view(1, 1, -1)
        # print("Word: ", word.shape)
        # print("D: ", token)
        sentence.append(token)
        # print(token)
        # print("C: ", sentence)

    # decoded = model.decoder(z, packed)
    sentence = (' ').join(sentence)
    print(sentence)
    return sentence


def main():

    data_path = '../Data/Dataset'
    train_data, val_data, test_data = get_datasets(data_path)
    tokenizer = train_data.tokenizer

    test_loader = DataLoader(
        val_data, batch_size=32, shuffle=False, collate_fn=padded_collate
    )

    model = SentenceVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_size=300,
        hidden_size=256,
        latent_size=16,
        num_layers=1,
        word_dropout_probability=0.0,
        unk_token_idx=tokenizer.unk_token_id,
        freebits = 0, # Freebits value is the lambda value as described in Kingma et al. 
    )

    # model_load_name = Path('1vanilla.pt')
    # model_load_name = Path('3word_dropout.pt')
    # model_load_name = Path('5freebits_dropout.pt')
    model_load_name = Path('6freebits_worddropout_mdr.pt')
    models_path = Path('models')

    model_load_path = models_path / model_load_name 

    model.load_from(model_load_path)
    # print(model.state_dict)

    sentence = sample_sentence(model, tokenizer, number=2)
    # print(sentences)

    model.to(torch.device('cuda'))
    test_loss = evaluate(model, test_loader, torch.device('cuda'), padding_index=0, print_every=50)
    print("Test loss: ", test_loss)


if __name__ == '__main__':
    main()
