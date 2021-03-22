from utils import *
from collections import defaultdict
import pickle
import numpy as np
from eval import eval
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--use_test_data', action='store_true')
args = parser.parse_args()

def get_best_index(alist, blist, clist):
    res = [a + b + c for a, b, c in zip(alist, blist, clist)]
    return np.argmax(res) 

if args.use_test_data:
    rows = read_json('work/result/test_matching_predictions.json')
    with open('./work/result/test_nil_prob.pkl', 'rb') as f:
        result1 = pickle.load(f)
    with open('./work/result/test_nil_two_prob.pkl', 'rb') as f:
        result2 = pickle.load(f)
    with open('./work/result/test_nil_three_prob.pkl', 'rb') as f:
        result3 = pickle.load(f)      
else:
    rows = read_json('work/result/dev_matching_predictions.json')
    with open('./work/result/dev_nil_prob.pkl', 'rb') as f:
        result1 = pickle.load(f)
    with open('./work/result/dev_nil_two_prob.pkl', 'rb') as f:
        result2 = pickle.load(f)    
    with open('./work/result/dev_nil_three_prob.pkl', 'rb') as f:
        result3 = pickle.load(f)  

y_pred1 = result1['y_pred']
y_pred2 = result2['y_pred']
y_pred3 = result3['y_pred']

cnt = 0
final = []
for row in rows:
    new_row = {'text_id': row['text_id'], 'text': row['text'], 'mention_data': []}
    for mention in row['mention_data']:
        if mention['kb_id'] == 'NIL':
            idx = get_best_index(y_pred1[cnt], y_pred2[cnt], y_pred3[cnt])
            mention['kb_id'] = id2type[idx]
            cnt += 1
        new_row['mention_data'].append(mention)
    final.append(new_row)

print(cnt, len(y_pred1))

if args.use_test_data:
    with open('work/result/result.json', 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
else:
    with open('work/result/dev_nil_predictions.json', 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(eval('./data/data60899/dev.json', 'work/result/dev_nil_predictions.json'))
