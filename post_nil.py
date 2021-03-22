from utils import *
from collections import defaultdict
import pickle
import numpy as np
from eval import eval
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--use_test_data', action='store_true')
args = parser.parse_args()

if args.use_test_data:
    rows = read_json('work/result/test_matching_predictions.json')
    with open('./work/result/test_nil.pkl', 'rb') as f:
        result = pickle.load(f)
else:
    rows = read_json('work/result/dev_matching_predictions.json')
    with open('./work/result/dev_nil.pkl', 'rb') as f:
        result = pickle.load(f)

y_pred = result['y_pred']

cnt = 0
final = []
for row in rows:
    new_row = {'text_id': row['text_id'], 'text': row['text'], 'mention_data': []}
    for mention in row['mention_data']:
        if mention['kb_id'] == 'NIL':
        	mention['kb_id'] = id2type[y_pred[cnt]]
        	cnt += 1
        new_row['mention_data'].append(mention)
    final.append(new_row)

print(cnt, len(y_pred))

if args.use_test_data:
    with open('work/result/result.json', 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
else:
    with open('work/result/dev_nil_predictions.json', 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(eval('./data/data60899/dev.json', 'work/result/dev_nil_predictions.json'))
