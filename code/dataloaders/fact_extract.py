import os
import json
from pymongo import MongoClient

VCR_ANNOTS_DIR = os.path.join(os.path.dirname('/media/ailab/songyoungtak/vcr_new/new/add_keyword/'))
split = 'val_scene_version'
client = MongoClient('localhost',27017)
db = client.datasetfact
collection = db.collect


with open(os.path.join(VCR_ANNOTS_DIR, '/media/ailab/songyoungtak/vcr_new/new/add_keyword/{}.jsonl'.format(split)), 'r') as f:
    items = [json.loads(s) for s in f]
    for i,item in enumerate(items):
        
        
