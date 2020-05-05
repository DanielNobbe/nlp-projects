from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.utils.data as data
import pickle

# The following code is based on code from Tom Runia (2019)
class SentenceDataset(data.Dataset):
    def __init__(self, train_list, test_list, val_list):
        self._train_data = train_list
        self._test_data = test_list
        self._val_data = val_list

        max_train_length = max([len(seq) for seq in train_list])
        max_test_length = max([len(seq) for seq in test_list])
        max_val_length = max([len(seq) for seq in val_list])
        tokens = ['[UNK]', '[PAD]']

        for data_list in [train_list, test_list, val_list]:
            for seq in data_list:
                tokens.extend( list(set(seq)) )
            
        self._tokens = list(set(tokens))
        self._train_size, self._test_size, self._val_size, self._vocab_size = len(self._train_data), len(self._test_data), len(self._val_data), len(self._tokens)
        self._tok_to_ix = { tok:i for i,tok in enumerate(self._tokens) }
        self._ix_to_tok = { i:tok for i,tok in enumerate(self._tokens) }

        self.split_dict = {'train': self._train_data, 'test': self._test_data, 'val': self._val_data}
        self.sequence_lengths = {'train': max_train_length, 'test': max_test_length, 'val': max_val_length}
        self.active_split = 'train'

    def __getitem__(self, index):
        
        output_tensor = torch.ones(self.sequence_lengths[self.active_split], dtype=torch.long) # ones are [PAD]

        split = self.split_dict[self.active_split]
        
        offset = np.random.randint(0, len(split)) # TODO: Make this unable to repeat itself? Or not a random iteration?
        sentence =  [self._tok_to_ix[tok] for tok in split[offset]]
        output_tensor[0:len(sentence)] = torch.LongTensor(sentence)
        return output_tensor

    def __len__(self):
        return (self._train_size + self._test_size + self._val_size)

    def convert_to_string(self, tok_ix):
        return ''.join(self._ix_to_tok[ix] for ix in tok_ix)

    def activate_split(self, split):
        assert split in self.split_dict.keys(), "Split should be 'train', 'test', or 'val', not %s".format(split)
        self.active_split = split

    @property
    def vocab_size(self):
        return self._vocab_size


def convert_to_list(data_text):

    POS_tags = [ # All POS tags, including brackets, to filter them out. - Apparently we use a weird version of the treebank?
    '(CC ', '(CD ', '(DT ', '(EX ', '(FW ', '(IN ', '(JJ ', '(JJR ', '(JJS ', 
    '(LS ', '(MD ', '(NN ', '(NNS ', '(NNP ', '(NNPS ', '(PDT ', '(POS ', '(PRP ',
    '(PRP$ ', '(RB ', '(RBR ', '(RBS ', '(RP ', '(SYM ', '(TO ', '(UH ', '(VB ', 
    '(VBD ', '(VBG ', '(VBN ', '(VBP ', '(VBZ ', '(WDT ', '(WP ', '(WP$ ', '(WRB ', 
    '(TOP ', '(S ', '(PP ', '(NP ', '(, ', '(SBAR ', '(WHNP ', '(VP ', '(PRT ', 
     '(ADVP ', '(PRN ', '(WHADVP ', '(-LRB- ', '(-RRB- ', '(. ', '(IN ', '($ ',
     '(ADJP ', '(NX ', '(QP ', '(SINV ', "('' ", '(SQ ', "(`` ", '(: ', '(CONJP ',
     '(UCP ', '(FRAG ', '(NAC ', '(WHPP ', '(SBARQ ', '(WHADJP ', '(# ', '(INTJ ', '(X ',
     '(RRC ', '(LST '
    ]


    for POS_tag_string in POS_tags:
    # print(POS_tag_string)
        data_text = data_text.replace(POS_tag_string, '')
    data_text = data_text.replace(')', '')
    data_text = data_text.replace('.\n', '[EOS]\n[BOS] ') # This creates one extra line at the end, which we'll remove later
    
    assert data_text.count('(') == 0, "ERROR: POS Tag not recognized."

    data_list = data_text.split('\n')

    for index, sequence in enumerate(data_list):
        data_list[index] = sequence.split(' ')
        # print(data_list[index])
    del data_list[-1] # delete last entry


    return data_list


def main():
    file_path = os.path.dirname(os.path.abspath(__file__))
    relative_data_path = '/../Data'
    training_set_relative_path = '/Dataset/train'
    test_set_relative_path = '/Dataset/test'
    val_set_relative_path = '/Dataset/val'

    training_set_path = file_path + relative_data_path +  training_set_relative_path
    test_set_path = file_path + relative_data_path + test_set_relative_path
    val_set_path = file_path + relative_data_path +  val_set_relative_path
    

    # Convert training data to iterable
    with open(training_set_path, 'r') as train_file:
        train_text = train_file.read()

    train_list = convert_to_list(train_text)


    # Convert test data to iterable
    with open(test_set_path, 'r') as test_file:
        test_text = test_file.read()

    test_list = convert_to_list(test_text)

    # Convert val data to iterable
    with open(val_set_path, 'r') as val_file:
        val_text = val_file.read()

    val_list = convert_to_list(val_text)

    # We need to create 1 dataset object for all three splits
    dataset = SentenceDataset(train_list, test_list, val_list)


    # Do we save these in a pickle now? To be unpickled? Or do we build a function that can return the dataloaders?
    batch_size = 2
    data_loader = data.DataLoader(dataset, batch_size, num_workers=1)

    dataset.activate_split('train')

    for sequence in data_loader:
        # print(sequence)
        break

    relative_pickle_path = '/Dataset/Dataloader.pkl'
    pickle_path = file_path + relative_data_path + relative_pickle_path
    with open(pickle_path, 'wb') as file:
        pickle.dump(dataset, file)

if __name__ == '__main__':
    main()

# Now, we should feed these iterables into torchtext

# # The following code is based on code from Tom Runia (2019)

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os
# import numpy as np
# import torch.utils.data as data
# import torch


# class TextDataset(data.Dataset):

#     def __init__(self, filename, seq_length):
#         assert os.path.splitext(filename)[1] == ".txt"
#         self._seq_length = seq_length
#         self._data = open(filename, 'r').read() 
# # TODO: This treats the complete dataset as just a list of letters. We should feed it a list of sentences, which each is a list of words. 
# Conversion to numbers per token should also be done here. Or we can use torchtext?
#         self._chars = list(set(self._data))
#         self._data_size, self._vocab_size = len(self._data), len(self._chars)
#         print("Initialize dataset with {} characters, {} unique.".format(
#             self._data_size, self._vocab_size))
#         self._char_to_ix = { ch:i for i,ch in enumerate(self._chars) }
#         self._ix_to_char = { i:ch for i,ch in enumerate(self._chars) }
#         self._offset = 0

#     def __getitem__(self, item):
#         offset = np.random.randint(0, len(self._data)-self._seq_length-2)
#         inputs =  [self._char_to_ix[ch] for ch in self._data[offset:offset+self._seq_length]]
#         targets = [self._char_to_ix[ch] for ch in self._data[offset+1:offset+self._seq_length+1]]
#         return torch.LongTensor(inputs), torch.LongTensor(targets)

#     def convert_to_string(self, char_ix):
#         return ''.join(self._ix_to_char[ix] for ix in char_ix)

#     def __len__(self):
#         return self._data_size

#     @property
#     def vocab_size(self):
#         return self._vocab_size