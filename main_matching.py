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
import pickle

def build_test_data(test, sub2id, id2info, args):
    data = []
    segs = []
    for row in test:
        text_a = row['text']
        for mention in row['mention_data']:
            mention['kb_id'] = 'test'
            if args.use_type_filter:
                neg_samples = get_neg_samples_with_type_filter(text_a, mention, sub2id, id2info, negative_sampling=-1)
            else:
                neg_samples = get_neg_samples(text_a, mention, sub2id, id2info, negative_sampling=-1, use_nil_as_cand=args.use_nil_as_cand)
            segs.append((len(data), len(data) + len(neg_samples)))
            data += neg_samples
    return data, segs

if __name__ == '__main__':
    parser = argparse.ArgumentParser('classify model with ERNIE')
    parser.add_argument('--from_pretrained', type=str, required=True, help='pretrained model directory or tag')
    parser.add_argument('--max_seqlen', type=int, default=184, help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=64, help='batchsize')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
    parser.add_argument('--use_lr_decay', action='store_true', help='if set, learning rate will decay to zero at `max_steps`')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='if use_lr_decay is set, '
            'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--inference_model_dir', type=str, default=None, help='inference model output directory')
    parser.add_argument('--save_dir', type=str, default=None, help='model output directory')
    parser.add_argument('--max_steps', type=int, default=None, help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='checkpoint to warm start from')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_test_data', action='store_true')
    parser.add_argument('--use_train_data', action='store_true')
    parser.add_argument('--use_type_filter', action='store_true')
    parser.add_argument('--use_nil_as_cand', action='store_true')
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained, special_token_list=['###'])
    #tokenizer = ErnieTinyTokenizer.from_pretrained(args.from_pretrained)

    kb = read_json('./data/data60899/kb.json')
    train = read_json('./data/data60899/train.json')
    if args.use_type_filter:
        dev = read_json('work/result/dev_nil_all.json')
        test = read_json('work/result/test_nil_all.json')
    else:
        dev = read_json('./data/data60899/dev.json')
        test = read_json('./data/data68533/eval.json')
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
    if args.use_test_data:
        test_data, test_segs = build_test_data(test, sub2id, id2info, args)
    elif args.use_train_data:
        test_data, test_segs = build_test_data(train, sub2id, id2info, args)
    else:
        test_data, test_segs = build_test_data(dev, sub2id, id2info, args)
    print(len(test_data))

    print('converting data to ernie format')
    test_features = [tokenizer.encode(row[0], row[1], args.max_seqlen-3) + (0, ) for row in test_data]

    # print(np.percentile([len(row[0]) for row in train_features], [0, 50, 95, 99, 100]))
    # print(np.percentile([len(row[0]) for row in dev_features], [0, 50, 95, 99, 100]))
    # to batch
    try:
        place = F.CUDAPlace(0)
    except:
        place = F.CPUPlace()
    with FD.guard(place):
        if 'wwm' in args.from_pretrained:
            config = json.load(open(os.path.join(args.from_pretrained, 'ernie_config.json'), 'rt', encoding='utf-8'))
            config['num_labels'] = 2
            model = ErnieModelForSequenceClassification(config)
            # print(model)
            print('loading checkpoint from %s' % 'chinese_roberta_wwm_pp')
            sd, _ = FD.load_dygraph('%s/roberta_wwm.pdparams' % args.from_pretrained)
            for k, v in model.state_dict().items():
                if k not in sd:
                    print('param:%s not set in pretrained model, skip' % k)
                    sd[k] = v # FIXME: no need to do this in the future
            model.set_dict(sd)
        else:       
            model = ErnieModelForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=2, name='')
            if args.init_checkpoint is not None:
                print('loading checkpoint from %s' % args.init_checkpoint)
                sd, _ = FD.load_dygraph(args.init_checkpoint)
                model.set_dict(sd)

        test_batch_data = batchify(test_features, args.bsz, args.max_seqlen)
        if args.debug:
            print(len(test_batch_data))
            print(test_batch_data[0])
            token_ids, seg_ids, labels = test_batch_data[0]
            for r1, r2 in zip(token_ids[:5], seg_ids[:5]):
                print(r1)
                print(r2)
                print(convert_ids_to_tokens(tokenizer.vocab, r1))        
        y_pred = []
        with FD.base._switch_tracer_mode_guard_(is_train=False):
            model.eval()
            for step, d in enumerate(tqdm(test_batch_data, desc='predicting')):
                ids, sids, _ = d
                ids, sids = FD.to_variable(ids), FD.to_variable(sids)
                _, logits = model(ids, sids)
                #print('\n'.join(map(str, logits.numpy().tolist())))
                y_pred += L.softmax(logits, -1).numpy().tolist()
                if args.debug and len(y_pred) > 5:
                    break

    print(len(y_pred), y_pred[:5])
    print(test_segs[:5])

    with open(args.save_path, 'wb') as f:
        pickle.dump({'segs': test_segs, 'y_pred': y_pred}, f)