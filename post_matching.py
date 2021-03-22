from utils import *
from collections import defaultdict
import pickle
import numpy as np
from eval import eval
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--use_test_data', action='store_true')
parser.add_argument('--input_path', type=str, default='test_matching.pkl')
parser.add_argument('--output_path', type=str, default='test_matching_predictions.json')
parser.add_argument('--thres', type=float, default=0.5)
args = parser.parse_args()

def get_best_index(alist, thres):
    if len(alist) == 0:
        return -1
    probs_1 = [v[1] for v in alist]
    if max(probs_1) < thres:
        return -1
    else:
        return np.argmax(probs_1) 

kb = read_json('./data/data60899/kb.json')
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

if args.use_test_data:
    with open('./work/result/%s' % args.input_path, 'rb') as f:
        result = pickle.load(f)
    rows = read_json('./data/data68533/eval.json')     
else:
    with open('./work/result/%s' % args.input_path, 'rb') as f:
        result = pickle.load(f)
    rows = read_json('./data/data60899/dev.json')

segs = result['segs']
y_pred = result['y_pred']

cnt = 0
final = []
for row in rows:
    new_row = {'text_id': row['text_id'], 'text': row['text'], 'mention_data': []}
    for mention in row['mention_data']:
        start, end = segs[cnt]
        bst_index = get_best_index(y_pred[start:end], args.thres)
        kb_ids = sub2id[mention['mention']]
        if len(kb_ids) > 0:
            # found
            if bst_index != -1:
                mention['kb_id'] = kb_ids[bst_index]
            # < 0.5
            else:
                mention['kb_id'] = 'NIL'
        # not found
        else:
            mention['kb_id'] = 'NIL'
        new_row['mention_data'].append(mention)
        cnt += 1
    final.append(new_row)

print(cnt, len(y_pred), len(segs))
print(len(final), final[:2])

if args.use_test_data:
    with open('work/result/%s' % args.output_path, 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
else:
    with open('work/result/%s' % args.output_path, 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    #print(eval('./data/data60899/dev.json', 'work/result/dev_matching_predictions.json'))
