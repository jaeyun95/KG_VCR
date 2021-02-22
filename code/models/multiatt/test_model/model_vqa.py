"""
Let's get the relationships yo
"""
##now vqa model
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torchvision.models as models
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
import h5py
import os
import time
from PIL import Image
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR, VCR_IMAGE_RESIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
@Model.register("MultiHopAttentionQA")
class AttentionQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQA, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.reasoning_encoder = TimeDistributed(reasoning_encoder)
        #add gcn model here


        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )
        self.node_pool = torch.nn.AdaptiveAvgPool3d((4,100,1024))
        self.final_node_pool = torch.nn.AdaptiveAvgPool2d((4,1024))
        #self.final_pooled = torch.nn.AdaptiveAvgPool2d((4, 1536))
        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        dim = sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        (span_encoder.get_output_dim(), self.pool_answer),
                                        (span_encoder.get_output_dim(), self.pool_question)] if to_pool])
        #final_input_size = 2560
        # torch.nn.Linear(dim, hidden_dim_maxpool),
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(2560, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        #here for vqa
        vgg_model = models.vgg16(pretrained=True)
        in_features = vgg_model.classifier[-1].in_features
        vgg_model.classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-1])
        self.vgg_model = vgg_model
        self.vgg_fc = nn.Linear(in_features, 1024)
        #vqa end
        ##vqa two
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.vqa_fc1 = nn.Linear(500,500)
        self.vqa_fc2 = nn.Linear(500,400)
        initializer(self)

    #_collect_obj_reps(span_tags, object_reps)  (question_tags, obj_reps['obj_reps'])
    # matching span_tags and object_reps / connect
    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        # all number is over zero /minimum is 0
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here

        # all zero size is same span_tag_fixed
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)

        # start:0, end:row_id.shape[0],
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]


        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster

        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    # self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)
        #and concat embedding span and retrieved feature
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
        #print('@@span : ',span_rep.shape)
        #@@span :  torch.Size([96, 4, 16, 1280])
        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def image_loads(self, name):
        '''
        image = load_image(os.path.join(VCR_IMAGES_DIR, item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape
        '''
        #image = load_image(os.path.join(VCR_IMAGES_DIR, name))
        image = Image.open(os.path.join(VCR_IMAGE_RESIZE, name)).convert('RGB')
        #image_fn = resize_image(image, desired_width=768, desired_height=384, random_pad=False)
        with torch.no_grad():
            img_feature = self.vgg_model(image)  # [batch_size, vgg16(19)_fc=4096]
        img_feature = self.vgg_fc(img_feature)

        norm_12 = img_feature.norm(p=2,dim=1,keepdim=True).detach()
        img_feature = img_feature.div(norm_12)

        return img_feature
    #add make visual concepts reps
    def forward(self,
                images: torch.Tensor, # image data
                objects: torch.LongTensor, # object data
                segms: torch.Tensor, #
                boxes: torch.Tensor, # bbox
                box_mask: torch.LongTensor, # mask for bbox
                question: Dict[str, torch.Tensor], # 'bert'
                question_tags: torch.LongTensor, # tag is
                question_mask: torch.LongTensor, #
                answers: Dict[str, torch.Tensor], # 'bert'
                answer_tags: torch.LongTensor, #
                answer_mask: torch.LongTensor, #
                #node: Dict[str, torch.Tensor],  # 'bert'
                #node_tags: torch.LongTensor,  # tag is
                #adjacent: Dict[str, torch.Tensor],
                #visual_concept: Dict[str, torch.Tensor],  # 'bert'
                #visual_concept_tags: torch.LongTensor,  # tag is
                metadata: List[Dict[str, Any]] = None, # annot_id, ind, movie, img_fn, question_number, // Ignore
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]: # Optional, which item is valid  # add here
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param keyword: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        #"objects": ["person", "person", "person", "car"],
        #objects = objects[:, :max_len]
        #compress segms
        #box_mask = box_mask[:, :max_len]
        #bounding box
        #boxes = boxes[:, :max_len]
        #extract mask
        #segms = segms[:, :max_len]

        #loading question & answer
        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))



        #for image detection : image --> bbox, mask, class --> compress
        #obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        #################################### <This is Grounding> ####################################
        # Now get the question representations
        #q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        #a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])
        padd_pre_qq_rep = torch.zeros(question.shape[0], question.shape[1], 100, question.shape[3]).cuda()
        padd_pre_qq_rep[:,:,:question.shape[2],:] = question


        image_input = torch.zeros(32,1024).cuda()
        for i,items in enumerate(metadata):
            image_input[i,:] = self.image_loads(items['img_fn'])
            #for j,item in enumerate(items):
                #graph_rep[i,j,:,:] = self.gcn(item,adjacent['adj'][i][j].cuda())
        print('image ',image_input)

        combined_feature = torch.mul(image_input,padd_pre_qq_rep)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)




        #################################### <This is Answer Generation> ####################################
        #pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]
        logits = self.final_mlp(combined_feature).squeeze(2)

        class_probabilities = F.softmax(logits, dim=-1)
        #print('reasoning output44 : ', class_probabilities.shape)
        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       #'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
