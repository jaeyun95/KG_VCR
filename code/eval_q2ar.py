"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
from config import VCR_ANNOTS_DIR,VCR_IMAGES_DIR
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorsys
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import matplotlib.lines as lines
from PIL import Image

parser = argparse.ArgumentParser(description='Evaluate question -> answer and rationale')
parser.add_argument(
    '-answer_preds',
    dest='answer_preds',
    default='/home/ailab/r2c/models/saves/mlp/answer2/valpreds.npy',
    help='Location of question->answer predictions',
    type=str,
)
parser.add_argument(
    '-rationale_preds',
    dest='rationale_preds',
    default='/home/ailab/r2c/models/saves/mlp/rationale/valpreds.npy',
    help='Location of question+answer->rationale predictions',
    type=str,
)
parser.add_argument(
    '-split',
    dest='split',
    default='val_scene_version',
    help='Split you\'re using. Probably you want val.',
    type=str,
)

# def random_colors(N, bright=True):
#     """
#     Generate random colors.
#     To get visually distinct colors, generate them in HSV space then
#     convert to RGB.
#     """
#     brightness = 1.0 if bright else 0.7
#     hsv = [(i / N, 1, brightness) for i in range(N)]
#     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#     random.shuffle(colors)
#     return colors
#
#
def collect_num(question, answers, rationales):
    all_num = []
    for num_list in question:
        if type(num_list) == type([]):
            for num in num_list:
                if num not in all_num:
                    all_num.append(num)

    for answer in answers:
        for num_list in answer:
            if type(num_list) == type([]):
                for num in num_list:
                    if num not in all_num:
                        all_num.append(num)

    for rationale in rationales:
        for num_list in rationale:
            if type(num_list) == type([]):
                for num in num_list:
                    if num not in all_num:
                        all_num.append(num)

    return all_num

def draw_box(image, boxList, objects, mentioned_num):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    [n, (y1, x1, y2, x2)]
    """
    masked_image = image.copy()
    fig, ax = plt.subplots(1, figsize=(12, 12))
    #rect = patches.Rectangle((x1,y2), x2-x1, y1-y2, linewidth=1, edgecolor=color, facecolor='none')
    #ax.add_patch(rect)
    #-----------------
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, box in enumerate(boxList):
        if i in mentioned_num:

            colors = np.random.rand(3)
            x1, y1, x2, y2, score = box
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colors, facecolor='none',
                                  linestyle="dashed")
            label = objects[i]
            obj = objects[:i]
            count = 1
            if label in obj:
                for o in obj:
                    if o == label:
                        count = count + 1
            label = label + str(count)
            #p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none',linestyle="dashed")
            ax.text(x1+10, y1 - 15, "{}".format(label),color='black', size=11, bbox=dict(linewidth=2,facecolor='white',edgecolor=colors))
            ax.add_patch(p)
        #image[y1:y1 + 2, x1:x2] = color
        #image[y2:y2 + 2, x1:x2] = color
        #image[y1:y2, x1:x1 + 2] = color
        #image[y1:y2, x2:x2 + 2] = color

    #image[y1:y1 + 2, x1:x2] = color
    #image[y2:y2 + 2, x1:x2] = color
    #image[y1:y2, x1:x1 + 2] = color
    #image[y1:y2, x2:x2 + 2] = color
    ax.imshow(masked_image)
    return ax

#def draw_all_box(ax, boxList,objects):
#    for index,box in enumerate(boxList):
#        color = np.random.rand(3)
#
#        ax = draw_box(ax,box,random_colors(100)[0])
#    return ax

#def draw_all_box(image, boxList):
#    for box in boxList:

args = parser.parse_args()

answer_preds = np.load(args.answer_preds)
rationale_preds = np.load(args.rationale_preds)

rationale_labels = []
answer_labels = []
rationale = []
answer = []
question = []
img_path = []
img_box = []
img_seg = []
img_object = []
with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(args.split)), 'r') as f:
    for l in f:
        pre_answer = []
        pre_rationale = []
        item = json.loads(l)
        answer_labels.append(item['answer_label'])
        rationale_labels.append(item['rationale_label'])
        answer.append(item['answer_choices'])
        rationale.append(item['rationale_choices'])
        img_path.append(item['img_fn'])
        img_object.append(item['objects'])
        question.append(item['question'])
        with open(os.path.join(VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            for o in f:
                item2 = json.loads(o)
                img_box.append(item2['boxes'])
                img_seg.append(item2['segms'])

answer_labels = np.array(answer_labels)
rationale_labels = np.array(rationale_labels)

# Sanity checks
assert answer_preds.shape[0] == answer_labels.size
assert rationale_preds.shape[0] == answer_labels.size
assert answer_preds.shape[1] == 4
assert rationale_preds.shape[1] == 4

answer_hits = answer_preds.argmax(1) == answer_labels
rationale_hits = rationale_preds.argmax(1) == rationale_labels
joint_hits = answer_hits & rationale_hits
'''
for i,data in enumerate(joint_hits):
    if (answer_hits[i] == True) and (rationale_hits[i] == False) and (img_path[i] == 'movieclips_Tupac_Resurrection/-yx9JK_HeQc@14.jpg'):
        print('image object : ',img_object[i])
        print('image : ',img_path[i])
        print('question : ',question[i])
        print('<answer>\n(1)',answer[i][0],' ,per: ',answer_preds[i][0],'\n(2) ',answer[i][1],' ,per: ',answer_preds[i][1],'\n(3) ',' ,per: ',answer_preds[i][2],answer[i][2],'\n(4) ',answer[i][3],' ,per: ',answer_preds[i][3],)
        print('correct answer : ',answer[i][answer_labels[i]])
        print('selected answer : ',answer[i][answer_preds.argmax(1)[i]])
        print('<rationale>\n(1) ',rationale[i][0],' ,per: ',rationale_preds[i][0],'\n(2) ',rationale[i][1],' ,per: ',rationale_preds[i][1],'\n(3) ',rationale[i][2],' ,per: ',rationale_preds[i][2],'\n(4) ',rationale[i][3],' ,per: ',rationale_preds[i][3])
        print('correct rationale : ', rationale[i][rationale_labels[i]])
        print('selected rationale : ',rationale[i][rationale_preds.argmax(1)[i]])
        img_path_2 = '/media/ailab/songyoungtak/vcr1/vcr1images/'+img_path[i]
        img = mpimg.imread(img_path_2)
        img.setflags(write=1)
        img = draw_box(img,img_box[i],img_object[i],collect_num(question[i], answer[i], rationale[i]))
        #plt.imshow(img)
        plt.show()
        print('======================================================================================================')
'''

print("Answer acc:    {:.3f}".format(np.mean(answer_hits)), flush=True)
print("Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
print("Joint acc:     {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)
