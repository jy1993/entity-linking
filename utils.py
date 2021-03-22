import json
from collections import defaultdict
import random
import numpy as np

nil_types = ['NIL_Work', 'NIL_Organization', 'NIL_VirtualThings', 'NIL_Person', 'NIL_Other', 'NIL_Culture', 'NIL_Event', 'NIL_Website', 'NIL_Location', 'NIL_Food', 'NIL_Vehicle', 'NIL_Diagnosis&Treatment', 'NIL_Game', 'NIL_Education', 'NIL_Natural&Geography', 'NIL_Biological', 'NIL_Law&Regulation', 'NIL_Brand', 'NIL_Time&Calendar', 'NIL_Medicine', 'NIL_Disease&Symptom', 'NIL_Awards', 'NIL_Software']
nil_ratios = [0.16004793448953097, 0.026530408441796213, 0.06527745414600046, 0.4426284078426151, 0.14057454811757264, 0.019340235012150063, 0.01268266702173696, 0.06071702007256749, 0.03641689690755967, 0.006590992310508971, 0.002629739356213175, 0.0003994540794247861, 0.0068905828700775605, 0.001963982557171865, 0.0003661662394727206, 0.005093039512666023, 0.0004327419193768516, 0.005425917912186678, 0.001830831197363603, 0.0009320595186578343, 0.0011650743983222928, 0.0005991811191371791, 0.0014646649578908824]
type2id = dict(zip(nil_types, range(len(nil_types))))
id2type = {v: k for k, v in type2id.items()}
type2des = {"Event":"事件活动", "Person":"人物", "Work":	"作品", "Location":	"区域场所", "Time&Calendar":	"时间历法", "Brand":	"品牌", "Natural&Geography":	"自然地理", "Game":	"游戏", "Biological":	"生物", "Medicine":	"药物", "Food":	"食物	", "Software":	"软件", "Vehicle":	"车辆", "Website":	"网站平台", "Disease&Symptom":	"疾病症状", "Organization":	"组织机构", "Awards":	"奖项", "Education":	"教育", "Culture":	"文化", "Constellation":	"星座", "Law&Regulation":	"法律法规","Other": "其他", "VirtualThings":'虚拟事物'}
type2ratio = dict(zip(nil_types, nil_ratios))

def read_json(filename):
	data = []
	with open(filename, encoding='utf8') as f:
		for line in f.readlines():
			data.append(json.loads(line.rstrip()))
	return data

def to_tsv(data, filename):
	with open(filename, 'w', encoding='utf8') as f:
		for line in data:
			f.write('%s\t%s\t%s\n' % (line[0], line[1], line[2]))

def get_text_a(text, mention, marker):
	# add marker before and after mention body
	if marker is not None:
		entity = mention['mention']
		offset = int(mention['offset'])
		text_a = text[:offset] + marker + entity + marker + text[offset+len(entity):]
	else:
		text_a = text
	return text_a	

def check_order(alist):
	# make sure the list is in ascending order
	offsets = [int(v['offset']) for v in alist]
	for i in range(len(alist)-1):
		if offsets[i] > offsets[i+1]:
			return False
	return True

def get_text_a_info(text, mention, all_mentions, id2info, marker):
	# for train_nil(provide type info for non-nil mentions)
	if not check_order(all_mentions):
		print(all_mentions)
		raise ValueError('not in ascending order')
	for m in all_mentions[::-1]:
		kb_id = m['kb_id']
		entity = m['mention']
		offset = int(m['offset'])		
		if 'NIL' not in kb_id:
			_type = id2info[kb_id]['type'].split('|')[0]
			text = text[:offset] + entity + '（类型:%s）' % type2des[_type] + text[offset+len(entity):]
		else:
			if m == mention:
				text = text[:offset] + marker + entity + marker + text[offset+len(entity):]
	return text	

def get_text_b(info, marker):
	# only use one spo list
	rel2obj = {row['predicate']: row['object'] for row in info['data']}
	assert len(rel2obj) > 0
	if '摘要' in rel2obj:
		key = '摘要'
	elif '义项描述' in rel2obj:
		key = '义项描述'
	else:
		key = list(rel2obj.keys())[0]
	text_b = key + ':' + rel2obj[key]
	subject = info['subject']
	if marker is not None:
		text_b = marker + subject + marker + text_b
	else:
		text_b = subject + text_b
	return text_b

def get_text_b_all(info, marker):
	# use all info
	rel2obj = {row['predicate']: row['object'] for row in info['data']}
	assert len(rel2obj) > 0
	_type = info['type'].split('|')[0]
	text_b = '类型:' + type2des[_type] + ';'
	other_keys = [i for i in rel2obj if i not in ['摘要', '义项描述']]
	for key in ['摘要', '义项描述'] + other_keys:
		if key in rel2obj:
			text_b += key + ':' + rel2obj[key] + ';'
	return text_b

def get_text_b_all_v2(info, marker):
	# use all info
	rel2obj = {row['predicate']: row['object'] for row in info['data']}
	assert len(rel2obj) > 0
	text_b = ''
	other_keys = [i for i in rel2obj if i not in ['义项描述', '摘要', '标签']]
	for key in ['义项描述', '摘要', '标签'] + other_keys:
		if key in rel2obj:
			text_b += key + ':' + rel2obj[key] + ';'
	return text_b

def get_text_b_type_desc(mention, all_mentions, id2info, marker=None):
	# use type and desc
	text_b = ''
	for m in all_mentions[::-1]:
		kb_id = m['kb_id']
		if 'NIL' not in kb_id:
			info = id2info[kb_id]
			# rel2obj = {row['predicate']: row['object'] for row in info['data']}
			_type = info['type'].split('|')[0]		
			entity = m['mention']
			text_b += entity + '类型:' + type2des[_type] + ';'
			# for key in ['义项描述']:
			# 	if key in rel2obj:
			# 		text_b += key + ':' + rel2obj[key] + ';'
	return text_b

def get_pos_sample(text, mention, info, marker_a='###', marker_b=None):
	if 'type' in mention:
		text_a = get_text_a(text, mention, marker_a) + ';类型:' + type2des[mention['type']]
	else:
		text_a = get_text_a(text, mention, marker_a)
	# text_b = get_text_b(info, marker_b)
	text_b = get_text_b_all(info, marker_b)
	return (text_a, text_b, 1)

def get_neg_samples(text, mention, sub2id, id2info, negative_sampling, marker_a='###', marker_b=None, use_nil_as_cand=False):
	data = []
	if 'type' in mention:
		text_a = get_text_a(text, mention, marker_a) + ';类型:' + type2des[mention['type']]	
	else:
		text_a = get_text_a(text, mention, marker_a)
	# if 'NIL' not in mention['kb_id']:
	# 	text_a = text_a + ';类型:' + type2des[id2info[mention['kb_id']]['type'].split('|')[0]]
	# else:
	# 	text_a = text_a + ';类型:' + type2des.get(mention['kb_id'].replace('NIL_', ''), '其他')
	cands = [_id for _id in sub2id[mention['mention']] if _id != mention['kb_id']]
	if len(cands) > negative_sampling and negative_sampling != -1:
		_ids = random.sample(cands, negative_sampling)
	else:
		_ids = cands
	# print(mention)
	# print(sub2id[mention['mention']])
	# print(cands)
	# print(_ids)
	for _id in _ids:
		# text_b = get_text_b(id2info[_id], marker_b)
		text_b = get_text_b_all(id2info[_id], marker_b)
		data.append((text_a, text_b, 0))

	# if use_nil_as_cand:
	# 	data.append((text_a, 'NIL', 0))
	return data

def build_matching_data(train, sub2id, id2info, is_training=False, use_nil_as_cand=False):
	# data generation process for train_matching.py
	data = []
	for row in train:
		text_a = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			neg_samples = get_neg_samples(text_a, mention, sub2id, id2info, negative_sampling=2 if is_training else -1)
			if 'NIL' not in kb_id:
				pos_sample = get_pos_sample(text_a, mention, id2info[kb_id])
				data.append(pos_sample)
				# if use_nil_as_cand:
				# 	data.append((get_text_a(text_a, mention, '###'), 'NIL', 0))
			else:
				# if use_nil_as_cand:
				# 	data.append((get_text_a(text_a, mention, '###'), 'NIL', 1))
				pass
			data += neg_samples
	return data

def convert_ids_to_tokens(vocab, ids):
	id2token = {v: k for k, v in vocab.items()}
	return [id2token[_id] for _id in ids]
# test_data = build_all_data(test, sub2id, id2info)
# to_tsv(train_data, 'static/train/train.tsv')
# to_tsv(dev_data, 'static/dev/dev.tsv')

def get_one_batch(batch, max_seq_len):
	batch_size = len(batch)
	token_ids = np.zeros((batch_size, max_seq_len), dtype='int64')
	segment_ids = np.zeros((batch_size, max_seq_len), dtype='int64')
	labels = np.zeros(batch_size, dtype='int64')
	# print(batch)
	for idx, one in enumerate(batch):
		token_ids[idx, :len(one[0])] = one[0]
		segment_ids[idx, :len(one[1])] = one[1]
		labels[idx] = one[2]
	return (token_ids, segment_ids, labels)

def batchify(features, batch_size, max_seq_len):
	# for batching
	batch_num = len(features) // batch_size
	if batch_num * batch_size != len(features):
		batch_num += 1
	data = [get_one_batch(features[idx*batch_size:(idx+1)*batch_size], max_seq_len) for idx in range(batch_num)]
	# for row in data[:5]:
	# 	print(row)
	return data

def build_nil_data(rows, marker_a='###'):
	# standard data generation process for train_nil.py
	data = []
	for row in rows:
		text = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			if 'NIL' in kb_id and 'NIL' != kb_id:
				text_a = get_text_a(text, mention, marker_a)
				_type = kb_id.split('|')[0]
				data.append((text_a, type2id[_type]))
	return data

def build_preround_nil_data(rows, id2info, marker_a='###'):
	# preround data generation process for train_nil.py(use non-nil mentions to train)
	data = []
	for row in rows:
		text = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			if 'NIL' not in kb_id:
				text_a = get_text_a(text, mention, marker_a)
				_type = 'NIL_' + id2info[kb_id]['type'].split('|')[0]
				if _type in type2id:
					data.append((text_a, type2id[_type]))
	return data	

# -------------------------------type-------------------------------
def build_nil_data_with_types(rows, id2info, marker_a='###'):
	# data generation process for train_nil.py(add type info for non-nil mentions)
	data = []
	for row in rows:
		text = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			if 'NIL' in kb_id:
				# text_a = get_text_a_info(text_a, mention, row['mention_data'], id2info, marker_a)
				text_a = get_text_a(text, mention, marker_a)
				text_b = get_text_b_type_desc(mention, row['mention_data'], id2info)
				_type = kb_id.split('|')[0]
				data.append((text_a, text_b, type2id[_type]))
	return data

# -------------------------------type-------------------------------
def add_non_nil_data(rows, id2info, non_nil_total, marker_a='###'):
	# add non nil data for nil type training
	corpus = defaultdict(list)
	for row in rows:
		text = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			if 'NIL' not in kb_id:
				text_a = get_text_a(text, mention, marker_a)
				_type = 'NIL_' + id2info[kb_id]['type'].split('|')[0]
				if _type in type2id:
					corpus[_type].append((text_a, type2id[_type]))

	data = []
	for _type, samples in corpus.items():
		n = min([int(type2ratio[_type] * non_nil_total), len(samples)])
		data += random.sample(samples, n)
	return data	

# -------------------------------hem-------------------------------
def hard_example_mining(train, idt_to_negids, id2info, marker_a='###', marker_b=None):
	# data generation process for train_hem.py
	data = []
	for row in train:
		text_a = row['text']
		text_id = row['text_id']
		for idx, mention in enumerate(row['mention_data']):
			kb_id = mention['kb_id']
			entity = mention['mention']
			text_a = get_text_a(text_a, mention, marker_a)
			_id = random.choice(idt_to_negids[(text_id, idx)])
			text_b = get_text_b_all(id2info[_id], marker_b)
			neg_samples = [(text_a, text_b, 0)]
			if 'NIL' not in kb_id:
				pos_sample = get_pos_sample(text_a, mention, id2info[kb_id])
				data.append(pos_sample)
			data += neg_samples
	return data
# -------------------------------hem-------------------------------

# def collect_nil_corpus(rows):
# 	# collect nil data
# 	type2nil = defaultdict(list)
# 	for row in rows:
# 		for mention in row['mention_data']:
# 			kb_id = mention['kb_id']
# 			if 'NIL' in kb_id:
# 				_type = kb_id.split('|')[0]
# 				type2nil[_type].append(mention)
# 	return type2nil	

# -------------------------------aug(replace)-------------------------------
def collect_nil_corpus(kb):
	# collect nil data
	type2sub = defaultdict(list)
	for row in kb:
		_type = 'NIL_' + row['type'].split('|')[0]
		type2sub[_type].append(row['subject'])
	return type2sub	

def build_aug_nil_data(rows, kb, replace_prob=0.1, marker_a='###'):
	# standard data generation process for train_nil.py
	# nil_corpus = collect_nil_corpus(rows)
	nil_corpus = collect_nil_corpus(kb)
	data = []
	for row in rows:
		text = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			if 'NIL' in kb_id:
				_type = kb_id.split('|')[0]
				# replace
				if random.random() < replace_prob:
					cand = random.choice(nil_corpus[_type])
					entity = mention['mention']
					offset = int(mention['offset'])
					text_a = text[:offset] + marker_a + cand + marker_a + text[offset+len(entity):]					
				# original
				else:
					text_a = get_text_a(text, mention, marker_a)
				data.append((text_a, type2id[_type]))
	return data
# -------------------------------aug(replace)-------------------------------

# -------------------------------se-------------------------------
def get_se_pos(token_ids, marker_id=12041):
	pos = [idx for idx, token_id in enumerate(token_ids) if token_id == marker_id]
	assert len(pos) == 2
	return pos

def convert_data_to_se_features(rows, tokenizer, max_seqlen):
	features = []
	for row in rows:
		bert_input = tokenizer.encode(row[0], [], max_seqlen)
		start_pos, end_pos = get_se_pos(bert_input[0])
		label = row[1]
		features.append(bert_input + (start_pos, end_pos, label))
	return features

def get_one_batch_se(batch, max_seq_len):
	batch_size = len(batch)
	token_ids = np.zeros((batch_size, max_seq_len), dtype='int64')
	segment_ids = np.zeros((batch_size, max_seq_len), dtype='int64')
	start_pos = np.zeros((batch_size, 2), dtype='int64')
	end_pos = np.zeros((batch_size, 2), dtype='int64')
	labels = np.zeros(batch_size, dtype='int64')
	# print(batch)
	for idx, one in enumerate(batch):
		token_ids[idx, :len(one[0])] = one[0]
		segment_ids[idx, :len(one[1])] = one[1]
		start_pos[idx] = [idx, one[2]]
		start_pos[idx] = [idx, one[3]]
		labels[idx] = one[4]
	return (token_ids, segment_ids, start_pos, end_pos, labels)

def batchify_se(features, batch_size, max_seq_len):
	# for batching
	batch_num = len(features) // batch_size
	if batch_num * batch_size != len(features):
		batch_num += 1
	data = [get_one_batch_se(features[idx*batch_size:(idx+1)*batch_size], max_seq_len) for idx in range(batch_num)]
	# for row in data[:5]:
	# 	print(row)
	return data
# -------------------------------se-------------------------------

# -------------------------------filter-------------------------------
def get_neg_samples_with_type_filter(text, mention, sub2id, id2info, negative_sampling, marker_a='###', marker_b=None):
	data = []
	text_a = get_text_a(text, mention, marker_a) + ';类型:%s' % type2des[mention['type']]
	# cands = [_id for _id in sub2id[mention['mention']] if _id != mention['kb_id'] and mention['type'] in id2info[_id]['type']]
	cands = [_id for _id in sub2id[mention['mention']] if _id != mention['kb_id']]
	if len(cands) > negative_sampling and negative_sampling != -1:
		_ids = random.sample(cands, negative_sampling)
	else:
		_ids = cands
	# print(mention)
	# print(sub2id[mention['mention']])
	# print(cands)
	# print(_ids)
	for _id in _ids:
		# text_b = get_text_b(id2info[_id], marker_b)
		text_b = get_text_b_all(id2info[_id], marker_b)
		data.append((text_a, text_b, 0))
	return data

# ------------------------------aux---------------------------------------
def get_pos_sample_with_aux(text, mention, info, marker_a='###', marker_b=None):
	if 'NIL' not in mention['kb_id']:
		_type = 'NIL_' + info['type'].split('|')[0]
	else:
		_type = mention['kb_id']
	text_a = get_text_a(text, mention, marker_a)
	text_b = get_text_b_all(info, marker_b)
	return (text_a, text_b, 1, type2id.get(_type, 4))

def get_neg_samples_with_aux(text, mention, sub2id, id2info, negative_sampling, marker_a='###', marker_b=None):
	data = []
	text_a = get_text_a(text, mention, marker_a)
	cands = [_id for _id in sub2id[mention['mention']] if _id != mention['kb_id']]
	if len(cands) > negative_sampling and negative_sampling != -1:
		_ids = random.sample(cands, negative_sampling)
	else:
		_ids = cands
	# print(mention)
	# print(sub2id[mention['mention']])
	# print(cands)
	# print(_ids)
	for _id in _ids:
		# text_b = get_text_b(id2info[_id], marker_b)
		text_b = get_text_b_all(id2info[_id], marker_b)
		_type = 'NIL_' + id2info[_id]['type'].split('|')[0]
		if _type in type2id:
			data.append((text_a, text_b, 0, type2id[_type]))
	return data

def build_matching_with_aux_data(train, sub2id, id2info, is_training=False):
	# data generation process for train_matching.py
	data = []
	for row in train:
		text_a = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			neg_samples = get_neg_samples_with_aux(text_a, mention, sub2id, id2info, negative_sampling=2 if is_training else -1)
			if 'NIL' not in kb_id:
				pos_sample = get_pos_sample_with_aux(text_a, mention, id2info[kb_id])
				data.append(pos_sample)
			data += neg_samples
	return data

def get_one_batch_aux(batch, max_seq_len):
	batch_size = len(batch)
	token_ids = np.zeros((batch_size, max_seq_len), dtype='int64')
	segment_ids = np.zeros((batch_size, max_seq_len), dtype='int64')
	labels = np.zeros(batch_size, dtype='int64')
	aux_lables = np.zeros(batch_size, dtype='int64')
	# print(batch)
	for idx, one in enumerate(batch):
		token_ids[idx, :len(one[0])] = one[0]
		segment_ids[idx, :len(one[1])] = one[1]
		labels[idx] = one[2]
		aux_lables[idx] = one[3]
	return (token_ids, segment_ids, labels, aux_lables)

def batchify_aux(features, batch_size, max_seq_len):
	# for batching
	batch_num = len(features) // batch_size
	if batch_num * batch_size != len(features):
		batch_num += 1
	data = [get_one_batch_aux(features[idx*batch_size:(idx+1)*batch_size], max_seq_len) for idx in range(batch_num)]
	# for row in data[:5]:
	# 	print(row)
	return data

# ------------------------------ranking---------------------------------------
def build_ranking_data(train, sub2id, id2info, topk):
	# data generation process for train_matching.py
	data = []
	for row in train:
		text_a = row['text']
		for mention in row['mention_data']:
			kb_id = mention['kb_id']
			if 'NIL' not in kb_id:
				rnd = random.random()
				pos_sample = get_pos_sample(text_a, mention, id2info[kb_id])
				neg_samples = get_neg_samples(text_a, mention, sub2id, id2info, negative_sampling=2)
				# padding_sample = (text_a, 'padding')
				# if len(neg_samples) == 1:
				# 	if rnd < 0.33:
				# 		data.append({'cand0': pos_sample[:2], 'cand1': neg_samples[0][:2], 'cand2': padding_sample, 'label': 0})
				# 	elif 0.33 <= rnd < 0.67:
				# 		data.append({'cand0': neg_samples[0][:2], 'cand1': pos_sample[:2], 'cand2': padding_sample, 'label': 1})
				# 	else:
				# 		data.append({'cand0': neg_samples[0][:2], 'cand1': padding_sample, 'cand2': pos_sample[:2], 'label': 2})
				# elif len(neg_samples) == 2:
				# 	if rnd < 0.33:
				# 		data.append({'cand0': pos_sample[:2], 'cand1': neg_samples[0][:2], 'cand2': neg_samples[1][:2], 'label': 0})
				# 	elif 0.33 <= rnd < 0.67:
				# 		data.append({'cand0': neg_samples[0][:2], 'cand1': pos_sample[:2], 'cand2': neg_samples[1][:2], 'label': 1})
				# 	else:
				# 		data.append({'cand0': neg_samples[0][:2], 'cand1': neg_samples[1][:2], 'cand2': pos_sample[:2], 'label': 2})
				if len(neg_samples) > 0:
					if rnd < 0.5:
						data.append({'cand0': pos_sample[:2], 'cand1': neg_samples[0][:2], 'label': 0})
					else:
						data.append({'cand0': neg_samples[0][:2], 'cand1': pos_sample[:2], 'label': 1})				
	return data	

def get_one_batch_ranking(batch, max_seq_len):
	batch_size = len(batch)
	token_ids_a = np.zeros((batch_size, max_seq_len), dtype='int64')
	segment_ids_a = np.zeros((batch_size, max_seq_len), dtype='int64')
	token_ids_b = np.zeros((batch_size, max_seq_len), dtype='int64')
	segment_ids_b = np.zeros((batch_size, max_seq_len), dtype='int64')
	# token_ids_c = np.zeros((batch_size, max_seq_len), dtype='int64')
	# segment_ids_c = np.zeros((batch_size, max_seq_len), dtype='int64')
	labels = np.zeros(batch_size, dtype='int64')
	# print(batch)
	for idx, one in enumerate(batch):
		token_ids_a[idx, :len(one[0][0])] = one[0][0]
		segment_ids_a[idx, :len(one[0][1])] = one[0][1]
		token_ids_b[idx, :len(one[1][0])] = one[1][0]
		segment_ids_b[idx, :len(one[1][1])] = one[1][1]
		# token_ids_c[idx, :len(one[2][0])] = one[2][0]
		# segment_ids_c[idx, :len(one[2][1])] = one[2][1]
		#labels[idx] = one[3]
		labels[idx] = one[2]
	# return (token_ids_a, segment_ids_a, token_ids_b, segment_ids_b, token_ids_c, segment_ids_c, labels)
	return (token_ids_a, segment_ids_a, token_ids_b, segment_ids_b, labels)

def batchify_ranking(features, batch_size, max_seq_len):
	# for batching
	batch_num = len(features) // batch_size
	if batch_num * batch_size != len(features):
		batch_num += 1
	data = [get_one_batch_ranking(features[idx*batch_size:(idx+1)*batch_size], max_seq_len) for idx in range(batch_num)]
	# for row in data[:5]:
	# 	print(row)
	return data

