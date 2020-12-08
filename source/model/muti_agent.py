#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/model/muti-agent.py
"""
import torch
import sys
from source.utils.misc import Pack
from source.model.base_model import BaseModel
from source.utils.misc import sequence_mask
from source.utils.misc import sequence_kd_mask
from source.utils.criterions import KDLoss
from tools.eval import eval_bleu_online
from tools.eval import eval_entity_f1_kvr_online
from tools.eval import eval_entity_f1_camrest_online


class Muti_Agent(BaseModel):
    """
    Muti_Agent
    """
    def __init__(self, data_name, ent_idx, nen_idx, model_S, model_TB, model_TE=None,
                 generator_S=None, generator_TB=None, generator_TE=None,
                 lambda_s=0.5, lambda_tb=0.5, lambda_te=0.5):
        super(Muti_Agent, self).__init__()

        self.name = "muti_agent"
        self.data_name = data_name

        self.model_S = model_S
        self.model_TB = model_TB
        self.model_TE = model_TE

        self.generator_S = generator_S
        self.generator_TB = generator_TB
        self.generator_TE = generator_TE

        self.ent_idx = ent_idx
        self.nen_idx = nen_idx

        # compute lambda for pre-training model (in all test data)
        # adding the ensemble model is still three lambda
        self.lambda_s = lambda_s
        self.lambda_tb = lambda_tb
        self.lambda_te = lambda_te

        # compute KD between two single model
        self.kd_loss = KDLoss()

    def one_to_one_learning(self, res_S, res_TB, kd_mask_b, bleu_S_gt_TB):
        """ one-to-one learning between model_S and model_TB """

        assert self.model_TE is None
        assert self.lambda_s + self.lambda_tb == 1.0
        NLL_S, logit_S, prob_S = res_S
        NLL_TB, logit_TB, prob_TB = res_TB

        # ignore equal in metric (low probability)
        KD_S = NLL_S if bleu_S_gt_TB else self.kd_loss(input=logit_S, target=prob_TB, mask=kd_mask_b)
        L_S = self.lambda_s * NLL_S + (1 - self.lambda_s) * KD_S

        KD_TB = self.kd_loss(input=logit_TB, target=prob_S, mask=kd_mask_b) if bleu_S_gt_TB else NLL_TB
        L_TB = self.lambda_tb * NLL_TB + (1 - self.lambda_tb) * KD_TB

        L_final = L_S + L_TB
        return L_final

    def one_to_two_learning(self, res_S, res_TB, res_TE, kd_mask_b, kd_mask_e, bleu_S_gt_TB, f1score_S_gt_TE):
        """ one-to-two learning, one for model_S, two for model_TB and model_TE """

        assert self.model_TE is not None
        NLL_S, logit_S, prob_S = res_S
        NLL_TB, logit_TB, prob_TB = res_TB
        NLL_TE, logit_TE, prob_TE = res_TE

        lambda_s_tb, lambda_s_te = 1 - self.lambda_tb, 1 - self.lambda_te

        KD_S_TB = NLL_S if bleu_S_gt_TB else self.kd_loss(input=logit_S, target=prob_TB, mask=kd_mask_b)
        KD_S_TE = NLL_S if f1score_S_gt_TE else self.kd_loss(input=logit_S, target=prob_TE, mask=kd_mask_e)
        L_S_TB = lambda_s_tb * NLL_S + (1 - lambda_s_tb) * KD_S_TB
        L_S_TE = lambda_s_te * NLL_S + (1 - lambda_s_te) * KD_S_TE
        L_S = (L_S_TB + L_S_TE) / 2

        KD_TB = self.kd_loss(input=logit_TB, target=prob_S, mask=kd_mask_b) if bleu_S_gt_TB else NLL_TB
        L_TB = self.lambda_tb * NLL_TB + (1 - self.lambda_tb) * KD_TB

        KD_TE = self.kd_loss(input=logit_TE, target=prob_S, mask=kd_mask_e) if f1score_S_gt_TE else NLL_TE
        L_TE = self.lambda_te * NLL_TE + (1 - self.lambda_te) * KD_TE

        L_final = L_S + L_TB + L_TE
        return L_final

    def one_to_three_learning(self, res_S, res_TB, res_TE, kd_mask_s, kd_mask_b, kd_mask_e,
                              bleu_ENS_gt_S, bleu_ENS_gt_TB, f1score_ENS_gt_TE):
        """ one-to-three learning, one for ensemble model, three for model_S, model_TB and model_TE """

        assert self.model_TE is not None
        NLL_S, logit_S, prob_S = res_S
        NLL_TB, logit_TB, prob_TB = res_TB
        NLL_TE, logit_TE, prob_TE = res_TE

        n_word = prob_S.size(-1)
        prob_mask_b = kd_mask_b.unsqueeze(-1).repeat(1, 1, n_word)
        prob_mask_e = kd_mask_e.unsqueeze(-1).repeat(1, 1, n_word)
        prob_ENS = prob_S + 0.5 * prob_TB * prob_mask_b.float() + 0.5 * prob_TE * prob_mask_e.float()
        prob_ENS = prob_ENS / prob_ENS.sum(dim=-1, keepdim=True)  # TODO: NOTE SUM == 0

        KD_S = self.kd_loss(input=logit_S, target=prob_ENS, mask=kd_mask_s) if bleu_ENS_gt_S else NLL_S
        L_S = self.lambda_s * NLL_S + (1 - self.lambda_s) * KD_S

        KD_TB = self.kd_loss(input=logit_TB, target=prob_ENS, mask=kd_mask_b) if bleu_ENS_gt_TB else NLL_TB
        L_TB = self.lambda_tb * NLL_TB + (1 - self.lambda_tb) * KD_TB

        KD_TE = self.kd_loss(input=logit_TE, target=prob_ENS, mask=kd_mask_e) if f1score_ENS_gt_TE else NLL_TE
        L_TE = self.lambda_te * NLL_TE + (1 - self.lambda_te) * KD_TE

        L_final = L_S + L_TB + L_TE
        return L_final

    def compare_metric(self, generator_1, generator_2, turn_inputs, kb_inputs, type='bleu', data_name='camrest'):
        """
        The metric of type in model_1 gt that in model_2 return True about a batch, otherwise False
        Default deal camrest dataset (ignore equal in metric because of low probability)
        """
        hyps_1, refs_1, tasks_1, gold_entity_1, kb_word_1 = generator_1.generate_batch(turn_inputs=turn_inputs,
                                                                                       kb_inputs=kb_inputs)
        hyps_2, refs_2, tasks_2, gold_entity_2, kb_word_2 = generator_2.generate_batch(turn_inputs=turn_inputs,
                                                                                       kb_inputs=kb_inputs)

        model_1_name, model_2_name = generator_1.model.name, generator_2.model.name

        if type == 'bleu':
            bleu_1 = eval_bleu_online(hyps=hyps_1, refs=refs_1)
            bleu_2 = eval_bleu_online(hyps=hyps_2, refs=refs_2)
            res = True if bleu_1 > bleu_2 else False
            report_str = type + ": " + model_1_name + '-' + str(bleu_1) + (' > ' if res else ' < ') + \
                         model_2_name + '-' + str(bleu_2)
            return res, report_str
        else:
            # default compute F1_score as metric
            if data_name == 'camrest':
                F1_score_1 = eval_entity_f1_camrest_online(hyps=hyps_1, tasks=tasks_1, gold_entity=gold_entity_1,
                                                           kb_word=kb_word_1)
                F1_score_2 = eval_entity_f1_camrest_online(hyps=hyps_2, tasks=tasks_2, gold_entity=gold_entity_2,
                                                           kb_word=kb_word_2)
            else:
                assert data_name == 'kvr'
                # default compute kvret as dataset todo complete like above camrest
                F1_score_1 = eval_entity_f1_kvr_online(hyps=hyps_1, tasks=tasks_1, gold_entity=gold_entity_1,
                                                           kb_word=kb_word_1)
                F1_score_2 = eval_entity_f1_kvr_online(hyps=hyps_2, tasks=tasks_2, gold_entity=gold_entity_2,
                                                           kb_word=kb_word_2)
            res = True if F1_score_1 > F1_score_2 else False
            report_str = type + ": " + model_1_name + '-' + str(F1_score_1) + (' > ' if res else ' < ') + \
                         model_2_name + '-' + str(F1_score_2)
            return res, report_str

    def iterate(self, turn_inputs, kb_inputs,
                optimizer=None, grad_clip=None, is_training=True, method="1-1"):
        """
        iterate
        note: this function iterate in the whole model (muti-agent) instead of single sub_model
        """

        # clear all memory before the begin of a new batch computation
        for name, model in self.named_children():
            if name.startswith("model_"):
                model.reset_memory()
                model.load_kb_memory(kb_inputs)

        # store the whole model (muti_agent)'s metric
        metrics_list_M, metrics_list_S, metrics_list_TB, metrics_list_TE = [], [], [], []
        # store the whole model (muti_agent)'s loss
        total_loss = 0
        # use to compute final loss (sum of each agent's loss) per turn for the cumulated total_loss in a batch
        loss = Pack()
        # use to store kb_mask for three single model
        kd_masks = Pack()

        # compare evaluation metric (bleu/f1score) among models
        if method == '1-3':
            # TODO complete
            bleu_ENS_gt_S, bleu_ENS_gt_TB, f1score_ENS_gt_TE = True, True, True
        else:
            # compute bleu_S_gt_TB per batch (compute metric for the following training batch)
            # (key: batch/following/training)
            res_bleu = self.compare_metric(generator_1=self.generator_S, generator_2=self.generator_TB,
                                           turn_inputs=turn_inputs, kb_inputs=kb_inputs, type='bleu',
                                           data_name=self.data_name)
            if isinstance(res_bleu, tuple):
                bleu_S_gt_TB, bleu_S_gt_TB_str = res_bleu
            else:
                assert isinstance(res_bleu, bool)
                bleu_S_gt_TB, bleu_S_gt_TB_str = res_bleu, ''
            if self.model_TE is not None:
                res_f1score = self.compare_metric(generator_1=self.generator_S, generator_2=self.generator_TE,
                                                  turn_inputs=turn_inputs, kb_inputs=kb_inputs, type='f1score',
                                                  data_name=self.data_name)
                if isinstance(res_f1score, tuple):
                    f1score_S_gt_TE, f1score_S_gt_TE_str = res_f1score
                else:
                    assert isinstance(res_f1score, bool)
                    f1score_S_gt_TE, f1score_S_gt_TE_str = res_f1score, ''


        # clear all memory again because of cumulation of the memory in the computation of the above generator
        for name, model in self.named_children():
            if name.startswith("model_"):
                model.reset_memory()
                model.load_kb_memory(kb_inputs)

        # begin iterate (a dialogue batch)
        for i, inputs in enumerate(turn_inputs):

            for name, model in self.named_children():
                if name.startswith("model_"):
                    if model.use_gpu:
                        inputs = inputs.cuda()
                    src, src_lengths = inputs.src
                    if name == "model_S":
                        tgt, tgt_lengths = inputs.tgt
                    elif name == "model_TB":
                        tgt, tgt_lengths = inputs.tgt_b
                    else:
                        assert name == "model_TE"
                        tgt, tgt_lengths = inputs.tgt_e
                    task_label = inputs.task
                    gold_entity = inputs.gold_entity
                    ptr_index, ptr_lengths = inputs.ptr_index
                    kb_index, kb_index_lengths = inputs.kb_index
                    enc_inputs = src[:, 1:-1], src_lengths - 2  # filter <bos> <eos>
                    dec_inputs = tgt[:, :-1], tgt_lengths - 1  # filter <eos>
                    target = tgt[:, 1:]  # filter <bos>
                    target_mask = sequence_mask(tgt_lengths - 1)
                    kd_mask = sequence_kd_mask(tgt_lengths - 1, target, name, self.ent_idx, self.nen_idx)

                    outputs = model.forward(enc_inputs, dec_inputs)
                    metrics = model.collect_metrics(outputs, target, ptr_index, kb_index)

                    if name == "model_S":
                        metrics_list_S.append(metrics)
                    elif name == "model_TB":
                        metrics_list_TB.append(metrics)
                    else:
                        metrics_list_TE.append(metrics)

                    kd_masks[name] = kd_mask
                    loss[name] = metrics

                    model.update_memory(dialog_state_memory=outputs.dialog_state_memory,
                                        kb_state_memory=outputs.kb_state_memory)

            # store necessary data for three single model
            res_S = loss.model_S.loss, loss.model_S.logits, loss.model_S.prob
            res_TB = loss.model_TB.loss, loss.model_TB.logits, loss.model_TB.prob
            if self.model_TE is not None:
                res_TE = loss.model_TE.loss, loss.model_TE.logits, loss.model_TE.prob
                kd_mask_e = kd_masks.model_TE
            kd_mask_s = kd_masks.model_S
            kd_mask_b = kd_masks.model_TB

            # muti-agent learning approaches
            if method == '1-1':
                L_final = self.one_to_one_learning(res_S=res_S, res_TB=res_TB, kd_mask_b=kd_mask_b,
                                                   bleu_S_gt_TB=bleu_S_gt_TB)
            elif method == '1-2':
                L_final = self.one_to_two_learning(res_S=res_S, res_TB=res_TB, res_TE=res_TE,
                                                   kd_mask_b=kd_mask_b, kd_mask_e=kd_mask_e,
                                                   bleu_S_gt_TB=bleu_S_gt_TB, f1score_S_gt_TE=f1score_S_gt_TE)
            elif method == '1-3':
                L_final = self.one_to_three_learning(res_S=res_S, res_TB=res_TB, res_TE=res_TE,
                                                     kd_mask_s=kd_mask_s, kd_mask_b=kd_mask_b, kd_mask_e=kd_mask_e,
                                                     bleu_ENS_gt_S=bleu_ENS_gt_S, bleu_ENS_gt_TB=bleu_ENS_gt_TB,
                                                     f1score_ENS_gt_TE=f1score_ENS_gt_TE)
            else:
                print("invalid training approach!")
                sys.exit(0)

            # collect muti-agent‘s total loss
            metrics_M = Pack(num_samples=metrics.num_samples)
            metrics_M.add(loss=L_final, logits=0.0, prob=0.0)
            metrics_list_M.append(metrics_M)

            # update in a batch
            total_loss += L_final
            loss.clear()
            kd_masks.clear()

        # check loss
        if torch.isnan(total_loss):
            raise ValueError("NAN loss encountered!")

        # compute and update gradient
        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=grad_clip)
            optimizer.step()

        # return
        if method == '1-3':
            return metrics_list_M, metrics_list_S, metrics_list_TB, metrics_list_TE
                   # bleu_ENS_gt_S, bleu_ENS_gt_TB, f1score_ENS_gt_TE
        else:
            return metrics_list_M, metrics_list_S, metrics_list_TB, metrics_list_TE, \
                   bleu_S_gt_TB_str, f1score_S_gt_TE_str
