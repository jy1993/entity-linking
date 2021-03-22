#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import json
from tqdm import tqdm
from functools import reduce, partial

import numpy as np
import argparse

import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L
import json
from collections import defaultdict
import random

#from model.bert import BertConfig, BertModelLayer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
from ernie.tokenizing_ernie import ErnieTokenizer, ErnieTinyTokenizer
from ernie.optimization import AdamW, LinearDecay
from utils import *
from sklearn.metrics import f1_score, classification_report

class FGM(object):
    """docstring for FGM"""
    def __init__(self, model, epislon=1):
        super(FGM, self).__init__()
        self.model = model
        self.store = {}
        self.epislon = epislon

    def attack(self, name='word_emb.weight'):
        for n, p in self.model.named_parameters():
            # print(n, p)
            if not p.stop_gradient and n == name:
                self.store[n] = p
                # print(p.shape)
                # print(p.gradient() ** 2)
                grad = FD.to_variable(p.gradient())
                norm = L.reduce_sum(grad ** 2) ** 0.5
                # print(norm.shape)
                if norm > 0:
                    p = p + self.epislon * grad / norm
        assert len(self.store) > 0
        #print(self.store.keys())

    def restore(self, name='word_emb.weight'):
        for n, p in self.model.named_parameters():
            if not p.stop_gradient and n == name:
                p = self.store[n]
        self.store = {}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_emb.weight', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name == name:
                if is_first_attack:
                    self.emb_backup[name] = param
                grad = FD.to_variable(param.gradient())
                norm = L.reduce_sum(grad ** 2) ** 0.5
                if norm != 0:
                    r_at = alpha * grad / norm
                    param = param + r_at
                    param = self.project(name, param, epsilon)

    def restore(self, emb_name='word_emb.weight'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name == name:
                assert name in self.emb_backup
                param = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        norm = L.reduce_sum(r ** 2) ** 0.5      
        if norm > epsilon:
            r = epsilon * r / norm
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                grad = FD.to_variable(param.gradient())
                self.grad_backup[name] = grad
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                param = self.grad_backup[name]

def soft_dice_loss(logits, labels):
    probs = L.softmax(logits, axis=-1)
    one_hot = F.one_hot(labels, depth=probs.shape[-1])
    intersection = L.reduce_sum(probs * one_hot, dim=-1)
    # union = L.reduce_sum(probs, axis=-1) + L.reduce_sum(labels, axis=-1)
    loss = 1 - intersection
    return L.reduce_mean(loss)

def focal_loss(logits, label, gamma=1):
    probs = L.softmax(logits, axis=-1)
    one_hot = F.one_hot(label, depth=probs.shape[-1])
    loss = - L.reduce_sum(one_hot * ((1.0 - probs) ** gamma) * L.log(probs)) / logits.shape[0]
    return loss

def chunk_softmax(logits, labels, topk=10):
    after_exp = L.exp(logits)
    out, _ = L.argsort(after_exp, axis=-1)
    denorm = L.reduce_sum(out[:, -topk:], dim=-1, keep_dim=True)
    probs = after_exp / denorm
    one_hot = F.one_hot(labels, depth=probs.shape[-1])
    loss = - L.reduce_sum(one_hot * L.log(probs)) / logits.shape[0]
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser('classify model with ERNIE')
    parser.add_argument('--from_pretrained', type=str, required=True, help='pretrained model directory or tag')
    parser.add_argument('--max_seqlen', type=int, default=72, help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=64, help='batchsize')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--use_lr_decay', action='store_true', help='if set, learning rate will decay to zero at `max_steps`')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='if use_lr_decay is set, '
            'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--inference_model_dir', type=str, default=None, help='inference model output directory')
    parser.add_argument('--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument('--max_steps', type=int, default=None, help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--wd', type=float, default=0, help='weight decay, aka L2 regularizer')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='checkpoint to warm start from')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--preround', action='store_true')
    parser.add_argument('--use_pseudo_label', action='store_true')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--non_nil_total', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--use_dice_loss', action='store_true')
    parser.add_argument('--use_chunk_softmax', action='store_true')
    parser.add_argument('--use_all', action='store_true')
    parser.add_argument('--use_fgm', action='store_true')
    parser.add_argument('--use_pgd', action='store_true')
    parser.add_argument('--use_dev_data', action='store_true')
    parser.add_argument('--kfold', type=int, default=None)
    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained, special_token_list=['###'])
    #tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    kb = read_json('./data/data60899/kb.json')
    train = read_json('./data/data60899/train.json')
    if args.kfold is not None:
        print('reading fold %s data...' % args.kfold)
        train = read_json('./work/data/train_fold%s.json' % args.kfold)

    dev = read_json('./data/data60899/dev.json')
    if args.use_dev_data:
        train = train + dev    
    sub2id = defaultdict(list)
    id2info = {}
    for row in kb:
        subject = row['subject']
        alias = row['alias']
        subject_id = row['subject_id']
        sub2id[subject].append(subject_id)
        for a in alias:
            sub2id[a].append(subject_id)
        id2info[subject_id] = row

    print('building data...')
    # using non-nil to train first
    if args.preround:
    	train_data = build_preround_nil_data(train, id2info)
    	dev_data = build_preround_nil_data(dev, id2info)
    elif args.use_all:
        train_data = build_preround_nil_data(train, id2info) + build_nil_data(train)
        dev_data = build_preround_nil_data(dev, id2info) + build_nil_data(dev)   
    elif args.use_aug:
        train_data = build_aug_nil_data(train, kb)
        dev_data = build_nil_data(dev)        
    else:
        # add type info of non-nil
        train_data = build_nil_data(train)
        dev_data = build_nil_data(dev)

        # using non-nil data to train
        if args.non_nil_total > 0:
            extra_data = add_non_nil_data(train, id2info, args.non_nil_total)
            print('using non nil data...')
            print('adding %s rows' % len(extra_data))
            train_data += extra_data

        # using test data to train
        if args.use_pseudo_label:
            test = read_json('./work/result/result.json')
            train_data = train_data + build_nil_data(test)
    print(len(train_data), len(dev_data))

    if args.debug:
        train_data = train_data[:1000]
        dev_data = dev_data[:1000]
    print(np.percentile([len(row[0])for row in train_data], [0, 50, 95, 99, 100]))

    print('converting data to ernie format')
    train_features = [tokenizer.encode(row[0], [], args.max_seqlen-2) + (row[1],) for row in train_data]
    dev_features = [tokenizer.encode(row[0], [], args.max_seqlen-2) + (row[1],) for row in dev_data]

    print(np.percentile([len(row[0]) for row in train_features], [0, 50, 95, 99, 100]))
    print(np.percentile([len(row[0]) for row in dev_features], [0, 50, 95, 99, 100]))
    # to batch
    print('start training...')
    bst_f1, global_step = 0, 0
    args.max_steps = (len(train_features) // args.bsz + 1) * args.epochs
    try:
        place = F.CUDAPlace(0)
    except:
        place = F.CPUPlace()

    with FD.guard(place):
        if 'ernie' in args.from_pretrained:
            model = ErnieModelForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=23, name='')
            if args.init_checkpoint is not None:
                print('loading checkpoint from %s' % args.init_checkpoint)
                sd, _ = FD.load_dygraph(args.init_checkpoint)
                model.set_dict(sd)

        elif 'wwm' in args.from_pretrained:
            config = json.load(open(os.path.join(args.from_pretrained, 'ernie_config.json'), 'rt', encoding='utf-8'))
            config['num_labels'] = 23
            model = ErnieModelForSequenceClassification(config)
            # print(model)
            print('loading checkpoint from %s' % 'chinese_roberta_wwm_pp')
            sd, _ = FD.load_dygraph('%s/roberta_wwm.pdparams' % args.from_pretrained)
            for k, v in model.state_dict().items():
                if k not in sd:
                    print('param:%s not set in pretrained model, skip' % k)
                    sd[k] = v # FIXME: no need to do this in the future
            model.set_dict(sd)  

        elif 'uer' in args.from_pretrained:
            config = json.load(open(os.path.join(args.from_pretrained, 'ernie_config.json'), 'rt', encoding='utf-8'))
            config['num_labels'] = 2
            model = ErnieModelForSequenceClassification(config)
            # print(model)
            print('loading checkpoint from %s' % args.from_pretrained)
            sd, _ = FD.load_dygraph('%s/uer_base.pdparams' % args.from_pretrained)
            for k, v in model.state_dict().items():
                if k not in sd:
                    print('param:%s not set in pretrained model, skip' % k)
                    sd[k] = v # FIXME: no need to do this in the future
            model.set_dict(sd)  
             
        elif 'macbert' in args.from_pretrained:
            config = json.load(open(os.path.join(args.from_pretrained, 'ernie_config.json'), 'rt', encoding='utf-8'))
            config['num_labels'] = 2
            model = ErnieModelForSequenceClassification(config)
            # print(model)
            print('loading checkpoint from %s' % args.from_pretrained)
            sd, _ = FD.load_dygraph('%s/chinese_macbert_base.pdparams' % args.from_pretrained)
            for k, v in model.state_dict().items():
                if k not in sd:
                    print('param:%s not set in pretrained model, skip' % k)
                    sd[k] = v # FIXME: no need to do this in the future
            model.set_dict(sd)            

        g_clip = F.clip.GradientClipByGlobalNorm(1.0) #experimental
        if args.use_lr_decay:
            opt = AdamW(learning_rate=LinearDecay(args.lr, int(args.warmup_proportion * args.max_steps), args.max_steps), parameter_list=model.parameters(), weight_decay=args.wd, grad_clip=g_clip)
        else:
            opt = AdamW(args.lr, parameter_list=model.parameters(), weight_decay=args.wd, grad_clip=g_clip)

        if args.use_fgm:
            fgm = FGM(model) 
        if args.use_pgd:
            pgd = PGD(model)
            K = 3       
        dev_batch_data = batchify(dev_features, args.bsz, args.max_seqlen)
        for epoch in range(args.epochs):
            # dynamic generating non-nil data
            if epoch > 0:
                if args.non_nil_total > 0:
                    print('building data...')
                    train_data = build_nil_data(train) + add_non_nil_data(train, id2info, args.non_nil_total)
                    print(len(train_data))
                    print('converting data to ernie format')
                    train_features = [tokenizer.encode(row[0], [], args.max_seqlen-2) + (row[1],) for row in train_data]                    

                elif args.use_aug:
                    train_data = build_aug_nil_data(train, kb)
                    print('converting data to ernie format')
                    train_features = [tokenizer.encode(row[0], [], args.max_seqlen-2) + (row[1],) for row in train_data]                                        

                if args.debug:
                    train_data = train_data[:1000]
                
            random.shuffle(train_features)
            train_batch_data = batchify(train_features, args.bsz, args.max_seqlen)
            if args.debug:
                print(len(train_batch_data))
                print(train_batch_data[0])
                token_ids, seg_ids, labels = train_batch_data[0]
                for r1, r2, r3 in zip(token_ids, seg_ids, labels):
                    print(r1)
                    print(r2)
                    print(r3)
                    print(convert_ids_to_tokens(tokenizer.vocab, r1))
            for step, d in enumerate(tqdm(train_batch_data, desc='training')):
                ids, sids, labels = d
                # print(ids.shape, sids.shape, labels.shape)
                ids, sids, labels = FD.to_variable(ids), FD.to_variable(sids), FD.to_variable(labels)
                loss, logits = model(ids, sids, labels=labels)
                if args.use_focal_loss:
                    loss = focal_loss(logits, labels)
                elif args.use_dice_loss:
                    loss = soft_dice_loss(logits, labels)
                elif args.use_chunk_softmax:
                    loss = chunk_softmax(logits, labels)
                loss.backward()
                global_step += 1
                if step % 100 == 0 and step > 0:
                    print('train loss %.5f lr %.3e' % (loss.numpy(), opt.current_step_lr()))

                if args.use_fgm:
                    fgm.attack()
                    adv_loss, _ = model(ids, sids, labels=labels)
                    adv_loss.backward()
                    fgm.restore()
                    loss = loss + adv_loss

                if args.use_pgd:
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0))
                        if t != K-1:
                            model.clear_gradients()
                        else:
                            pgd.restore_grad()
                        adv_loss, _ = model(ids, sids, labels=labels)
                        adv_loss.backward()
                    pgd.restore()
                    loss = loss + adv_loss
                opt.minimize(loss)
                model.clear_gradients()
                if global_step % args.save_steps == 0:
                    F.save_dygraph(model.state_dict(), args.save_dir + '_%s' % global_step)
                if step % args.eval_steps == 0 and step > 0:
                    y_true, y_pred = [], []
                    with FD.base._switch_tracer_mode_guard_(is_train=False):
                        model.eval()
                        for step, d in enumerate(tqdm(dev_batch_data, desc='evaluating %d' % epoch)):
                            ids, sids, labels = d
                            ids, sids, labels = FD.to_variable(ids), FD.to_variable(sids), FD.to_variable(labels)
                            loss, logits = model(ids, sids, labels=labels)
                            #print('\n'.join(map(str, logits.numpy().tolist())))
                            y_pred += L.argmax(logits, -1).numpy().tolist()
                            y_true += labels.numpy().tolist()
                        model.train()

                    if args.debug:
                        print(y_pred[:10], y_true[:10])
                    f1 = f1_score(y_true, y_pred, average='micro')
                    print('f1 %.5f' % f1)
                    print(classification_report(y_true, y_pred))

                    if f1 > bst_f1:
                        F.save_dygraph(model.state_dict(), args.save_dir)
                        bst_f1 = f1
                        print('saving model with best f1: %.3f' % bst_f1)