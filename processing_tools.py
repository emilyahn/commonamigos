import math
import csv
import json
from collections import defaultdict


# read in txt file of 3 columns (LID tsv, qual tsv, chat json)
# will combine across batches
# returns all_data: 	dict[chat_id] = dicts...
# each entry in qual tsv will be direct key in each chat dict
def load_all_data(master_filelist):
	all_data = defaultdict(dict)

	# read filenames from list
	file_names = defaultdict(list)
	master_reader = csv.DictReader(open(master_filelist, 'r'), delimiter='\t')
	for row in master_reader:
		for file_type in master_reader.fieldnames:  # iterate thru keys (header)
			if row[file_type]:
				file_names[file_type].append(row[file_type])

	# read all qual TSVs
	for qual_file in file_names['qual_tsv']:
		reader = csv.DictReader(open(qual_file, 'r'), delimiter='\t')
		for row in reader:
			chat_id = row['chat_id']
			for category in reader.fieldnames:  # iterate thru keys (header)
				if category == 'chat_id':
					continue
				all_data[chat_id][category] = row[category]
	print('read all qual tsvs')

	# read chat jsons
	# grab agents, events, styles (in case not present in qual surveys)
	for chat_json in file_names['chat_json']:
		chat_list = json.load(open(chat_json))
		# store agents and events
		for chat in chat_list:
			chat_id = chat['uuid']
			all_data[chat_id]['agents'] = chat['agents']
			all_data[chat_id]['style'] = chat['scenario']['styles']

			all_data[chat_id]['all_chat'] = [(d['agent'], d['data']) for d in chat['events'] if d['action'] == 'message']

	print('read all chat jsons')

	# read LID TSVs
	# creates ['txt_dict'] and ['lbl_dict'] entries per chat in all_data
	# each of those are dict[utt_num] = list(words or lbl)
	for lid_file in file_names['lid_tsv']:
		with open(lid_file) as f:
			for line in f.readlines():
				all_info = line.replace('\n', '').split('\t')

				# concat chat_id with utt-number
				# utt-number always has two digits (i.e. 3 -> 03)
				chat_id = all_info[1]
				utt_num = all_info[2].zfill(2)
				txt = all_info[3]  # single token
				lbl = all_info[4]
				if len(all_info) > 5:
					if all_info[5] != '':
						# use manually fixed LID tag instead, if applicable
						lbl = all_info[5]

				if 'txt_dict' not in all_data[chat_id]:
					all_data[chat_id]['txt_dict'] = defaultdict(list)
					all_data[chat_id]['lbl_dict'] = defaultdict(list)
					all_data[chat_id]['uttids'] = defaultdict(list)

				all_data[chat_id]['txt_dict'][utt_num].append(txt)
				all_data[chat_id]['lbl_dict'][utt_num].append(int(lbl))

				uttid = 'co_{}_{}'.format(chat_id, utt_num)
				all_data[chat_id]['uttids'][utt_num] = uttid

	print('read all lid tsvs')

	# calc m and i per chat
	for chat_id in all_data:
		if not is_valid_chat(all_data, chat_id):
			continue

		lbl_lst = []
		lbl_dict = all_data[chat_id]['lbl_dict']
		for utt_num in sorted(lbl_dict.keys()):  # utt_num = '00', '01', ...
			words_01 = lbl_dict[utt_num]
			lbl_lst.append(words_01)

		all_data[chat_id]['m_idx'] = calc_m_idx(lbl_lst, one_user=True)
		all_data[chat_id]['i_idx'] = calc_i_idx(lbl_lst, one_user=True)

	return all_data


# ensure chat is present, has survey AND contains text
def is_valid_chat(full_data, chat_id):
	if chat_id not in full_data:
		return False

	return 'outcome' in full_data[chat_id] and 'txt_dict' in full_data[chat_id]


# multilingual index across all words across users
# default: chat_lid_lsts = dict[user] = list(list): [utterance x lid tags]
# if one_user: chat_lid_lsts = list(list): [utterance x lid tags]
def calc_m_idx(chat_lid_lsts, one_user=False):
	num_spa_eng = [0, 0, 0]  # 3rd spot will be ignored

	if one_user:  # process 1 dialogue only (no dict)
		for utt_list in chat_lid_lsts:
			for x in utt_list:
				# printtype(x)
				num_spa_eng[x] += 1

	else:
		for user, lid_lists in chat_lid_lsts.items():
			for utt_list in lid_lists:
				for x in utt_list:
					num_spa_eng[x] += 1

	num_total_01 = float(num_spa_eng[0] + num_spa_eng[1])

	try:
		sigma = math.pow(num_spa_eng[0]/num_total_01, 2) + math.pow(num_spa_eng[1]/num_total_01, 2)
		return (1 - sigma) / sigma

	except ZeroDivisionError:
		return 0


# integration index, averaged per user (for multiple dialogues)
# chat_lid_lsts = dict[user] = list(list): [utterance x lid tags]
# if one_user: chat_lid_lsts = list(list): [utterance x lid tags]
def calc_i_idx(chat_lid_lsts, one_user=False):
	scores = []
	if one_user:
		lid_lists_all = [chat_lid_lsts]

	else:
		lid_lists_all = [lsts for user, lsts in chat_lid_lsts.items()]

	for lid_lists in lid_lists_all:
		user_01_list = []
		for utt_list in lid_lists:
			user_01_list.extend(utt_list)

		# remove LID tag = 2 (neither eng nor spa)
		flat_list = [utt for utt in user_01_list if utt < 2]

		if len(flat_list) < 2:
			score = 0.

		else:
			num_switches = 0
			for i, lid in enumerate(flat_list[1:]):
				if flat_list[i - 1] != lid:
					num_switches += 1

			score = float(num_switches) / (len(flat_list) - 1)

		scores.append(score)  # if one_user, only 1 score to append

	# printscores
	try:
		return sum(scores) / len(scores)

	except ZeroDivisionError:
		return 0
