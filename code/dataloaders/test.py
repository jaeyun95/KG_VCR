"""
Dataloaders for VCR
"""
import json
import os

import numpy as np
import torch

import h5py


GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']


# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }
max_len = 0
h5fn = os.path.join('/media/ailab/songyoungtak/vcr_new/new/add_keyword/', f'bert_da_keyword.h5')
with h5py.File(h5fn, 'r') as h5:
    for index in range(len(h5)): 
        grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}
        if len(grp_items['word']) > max_len:
            max_len = len(grp_items['word'])
            print(grp_items['word'])
print(max_len)
            
