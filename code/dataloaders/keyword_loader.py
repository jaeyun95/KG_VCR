"""
Dataloaders for VCR
"""
import json
import os

import numpy as np
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR


# Here's an example jsonl
# {
# "word_id": 0,
# "word_orig": "apple",
# "word": ["apple"]
# }

class Word(Dataset):
    def __init__(self, split, embs_to_load='bert_da'):
        """
        :param split: word_file name ex) all_word.jsonl
        :param embs_to_load: Which precomputed embeddings to load. ex) bert_da_all_word

        """

        #load word file        
        with open(os.path.join(VCR_ANNOTS_DIR, '/media/ailab/songyoungtak/vcr_new/new/add_keyword/word_embedding/{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        #setting word embedding file
        self.h5fn = os.path.join(VCR_ANNOTS_DIR, f'{self.embs_to_load}_all_word.h5')
        print("Loading embeddings from {}".format(self.h5fn), flush=True)

        self.word_list = []
        #load word embedding file
        with h5py.File(self.h5fn, 'r') as h5:
            for num in range(len(self.items)):
                grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(num)].items()}
                self.items[num]['bert'] = grp_items[f'word']
    def __len__(self):
        return len(self.items)

# You could use this for debugging maybe
# if __name__ == '__main__':
#     train, val, test = VCR.splits()
#     for i in range(len(train)):
#         res = train[i]
#         print("done with {}".format(i))
