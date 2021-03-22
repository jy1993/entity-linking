from utils import *
from collections import defaultdict
import pickle
import numpy as np
from eval import eval
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--use_test_data', action='store_true')
parser.add_argument('--thres', type=float, default=0.4)
parser.add_argument('--input_path', type=str, default=None)
args = parser.parse_args()

def get_best_index(alist, thres):
    if len(alist) == 0:
        return -1
    probs_1 = [v[1] for v in alist]
    if max(probs_1) < thres:
        return -1
    else:
        return np.argmax(probs_1) 

def get_list_mean(alist, blist, clist, dlist, elist, flist, glist):
    assert len(alist) == len(blist) == len(clist) == len(dlist) == len(elist) == len(flist) == len(glist)
    # print(np.array(alist), np.array(blist))
    result = []
    for a, b, c, d, e, f, g in zip(alist, blist, clist, dlist, elist, flist, glist):
        # result.append((0.4 * a[0] + 0.6 * b[0], 0.4 * a[1] + 0.6 * b[1]))
        #result.append(((1.2 * a[0] + 1.2 * b[0] + c[0] + 0.8 * d[0] + 0.8 * e[0]) / 5, (1.2 * a[1] + 1.2 * b[1] + c[1] + 0.8 * d[1] + 0.8 * e[1]) / 5))
        #result.append(((1.2 * a[0] + 1.2 * b[0] + c[0] + 0.8 * d[0] + 0.8 * e[0] + f[0]) / 6, (1.2 * a[1] + 1.2 * b[1] + c[1] + 0.8 * d[1] + 0.8 * e[1] + f[1]) / 6))
        #result.append(((1.1 * a[0] + 1.1 * b[0] + 1 * c[0] + 0.9 * d[0] + 0.9 * e[0] + 1 * f[0]) / 6, (1 * a[1] + 1 * b[1] + 1 * c[1] + 1 * d[1] + 1 * e[1] + 1 * f[1]) / 6))
        result.append(((1.1 * a[0] + 1.1 * b[0] + 1 * c[0] + 0.9 * d[0] + 0.9 * e[0] + 1 * f[0]) / 6, (1 * a[1] + 1 * b[1] + 1 * c[1] + 1 * d[1] + 1 * e[1] + 1 * f[1] + g[1]) / 7))
    # print(alist, blist, result)
    return result

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
    # with open('./work/result/test_matching_neg1.pkl', 'rb') as f: 
    with open('./work/result/test_matching_a.pkl', 'rb') as f:
        result1 = pickle.load(f)
    with open('./work/result/test_matching_b.pkl', 'rb') as f:
        result2 = pickle.load(f)
    with open('./work/result/test_matching_f.pkl', 'rb') as f:
        result3 = pickle.load(f)    
    with open('./work/result/test_matching_d.pkl', 'rb') as f:
        result4 = pickle.load(f)   
    with open('./work/result/test_matching_e.pkl', 'rb') as f:
        result5 = pickle.load(f)   
    with open('./work/result/test_matching_g.pkl', 'rb') as f:
        result6 = pickle.load(f) 
    with open('./work/result/test_matching_i.pkl', 'rb') as f:
        result7 = pickle.load(f)
    rows = read_json('./data/data68533/eval.json')     
else:
    # with open('./work/result/dev_matching_neg1.pkl', 'rb') as f:
    with open('./work/result/dev_matching_neg2_two.pkl', 'rb') as f:
        result1 = pickle.load(f)
    with open('./work/result/dev_matching_neg2.pkl', 'rb') as f:
        result2 = pickle.load(f)
    with open('./work/result/dev_matching_neg2_three.pkl', 'rb') as f:
        result3 = pickle.load(f)    
    rows = read_json('./data/data60899/dev.json')  

segs = result1['segs']
y_pred1 = result1['y_pred']
y_pred2 = result2['y_pred']
y_pred3 = result3['y_pred']
y_pred4 = result4['y_pred']
y_pred5 = result5['y_pred']
y_pred6 = result6['y_pred']
y_pred7 = result7['y_pred']

cnt = 0
final = []
for row in rows:
    new_row = {'text_id': row['text_id'], 'text': row['text'], 'mention_data': []}
    for mention in row['mention_data']:
        start, end = segs[cnt]
        bst_index = get_best_index(get_list_mean(y_pred1[start:end], y_pred2[start:end], y_pred3[start:end], y_pred4[start:end], y_pred5[start:end], y_pred6[start:end], y_pred7[start:end]), args.thres)
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
        if mention['offset'] == '-1':
            mention['offset'] = '0'
        new_row['mention_data'].append(mention)
        cnt += 1
    final.append(new_row)

print(cnt, len(y_pred1), len(segs))
print(len(final), final[:2])

if args.use_test_data:
    with open('work/result/test_matching_predictions.json', 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
else:
    with open('work/result/dev_matching_predictions.json', 'w', encoding='utf8') as f:
        for line in final:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(eval('./data/data60899/dev.json', 'work/result/dev_matching_predictions.json'))
