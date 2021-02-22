"""
Let's get the relationships yo
"""

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator
from .gcn.pygcn.models import GCN
import h5py
import os
import time

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
        self.gcn = GCN(nfeat = 1024 ,nhid = 1024 ,noutput =  1024,dropout = 0.5).cuda()

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

    def node_compress(self, node, visual_concept, question):
        """
        :param node: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param visual_concept: [batch_size, ..leading_dims.., L]
        :param question: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        # zero padding --> answer size : 42, rationale size : 82
        """
        # concat embedding visual_concept and question 5-dim
        pre_node_rep = torch.cat((visual_concept['bert'], question['bert']), 2)
        #4-dim version
        padd_pre_node_rep = torch.zeros(pre_node_rep.shape[0], pre_node_rep.shape[1], 83, pre_node_rep.shape[3]).cuda()
        padd_pre_node_rep[:, :, :pre_node_rep.shape[2], :] = pre_node_rep

        padd_pre_node_rep = padd_pre_node_rep.view(padd_pre_node_rep.shape[0],padd_pre_node_rep.shape[1],padd_pre_node_rep.shape[2]*padd_pre_node_rep.shape[3])
        padd_pre_node_rep = torch.unsqueeze(padd_pre_node_rep,2)
        padd_pre_node_rep = padd_pre_node_rep.expand(padd_pre_node_rep.shape[0],padd_pre_node_rep.shape[1],node['bert'].shape[2],padd_pre_node_rep.shape[3])

        node_rep = torch.cat((node['bert'],padd_pre_node_rep),3)
        padd_node_rep = torch.zeros(node_rep.shape[0],node_rep.shape[1],100,node_rep.shape[3])
        padd_node_rep[:,:,:node_rep.shape[2],:] = node_rep


        return self.node_pool(node_rep)
        #return padd_node_rep



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
                node: Dict[str, torch.Tensor],  # 'bert'
                #node_tags: torch.LongTensor,  # tag is
                adjacent: Dict[str, torch.Tensor],
                visual_concept: Dict[str, torch.Tensor],  # 'bert'
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
        objects = objects[:, :max_len]
        #compress segms
        box_mask = box_mask[:, :max_len]
        #bounding box
        boxes = boxes[:, :max_len]
        #extract mask
        segms = segms[:, :max_len]

        #loading question & answer
        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))



        #for image detection : image --> bbox, mask, class --> compress
        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        #################################### <This is Grounding> ####################################
        # Now get the question representations
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        #################################### <This is Knowledge Encoding> ####################################
        #(1) Concat and Pass MaxPooling
        node_rep = self.node_compress(node, visual_concept, question)
        #print('node_rep!! ',node_rep.shape)
        #################################### <This is Knowledge Reasoning> ####################################
        #(1) Input
        #start
        #graph_rep = torch.zeros(node_rep.shape[0],node_rep.shape[1],node_rep.shape[2],node_rep.shape[3]).cuda()
        #for i,items in enumerate(node_rep):
        #    for j,item in enumerate(items):
        #        graph_rep[i,j,:,:] = self.gcn(item,adjacent['adj'][i][j].cuda())
        #end

        #print('node : ',graph_rep)
        #gcn_rep = self.gcn(node_rep,adjacent['adj'])



        #################################### <This is Contextualization> ####################################
        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]

        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                            a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:,None,None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))




        #################################### <This is Reasoning> ####################################
        reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                           (attended_o, self.reasoning_use_obj),
                                                           (attended_q, self.reasoning_use_question)]
                                      if to_pool], -1)

        if self.rnn_input_dropout is not None:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)

        reasoning_output = self.reasoning_encoder(reasoning_inp, answer_mask)

        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (a_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)

        #################################### <This is Answer Generation> ####################################
        pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]
        #print('@@@ node_rep ',node_rep.shape)
        #start
        #final_node_rep = graph_rep.view(graph_rep.shape[0],graph_rep.shape[1],graph_rep.shape[2]*graph_rep.shape[3])
        #end
        #add for no_gcn test
        final_node_rep = node_rep.view(node_rep.shape[0], node_rep.shape[1], node_rep.shape[2] * node_rep.shape[3])
        final_node_rep_pool = self.final_node_pool(final_node_rep)
        #print('@@@@ final_node ',final_node_rep.shape)
        final_material_rep = torch.cat((pooled_rep,final_node_rep_pool),-1)
        #final_rep = self.final_pooled(final_material_rep)
        #print('@@@ final_material ',final_material_rep.shape)
        #print('@@@ final_mlp ', pooled_rep.shape)
        logits = self.final_mlp(final_material_rep).squeeze(2)


        class_probabilities = F.softmax(logits, dim=-1)
        #print('reasoning output44 : ', class_probabilities.shape)
        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
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
