from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import os
import numpy as np
from typing import List
from operator import itemgetter
from utils.trainer_utils import get_tokenizer
from utils.utils import TRAINING_PATH, TESTING_PATH
from utils.text_utils import check_sentence_contain_entities, extract_sentence_contain_both_2_entities, extract_nearest_sentences_contain_2_entities
from data_loaders.sequence_padding import PadSequenceALPSDataset
from models.tokenization import FullTokenizer
from transformers import XLNetTokenizer, BertTokenizer, ElectraTokenizer
from tqdm import tqdm

class ALPS_Sample():
    def __init__(self, text_list: List, tokenize):
        self.tokenize = tokenize
        self.title = text_list[0].split('|')[-1]
        self.content = text_list[1].split('|')[-1]
        self.text = self.title + self.content
        self.text = self.text.lower()
        self.entities_list_text = [text_list[i] for i in range(2, len(text_list)) if 'CID' not in text_list[i]]
        self.ca_list_text = [text_list[i] for i in range(2, len(text_list)) if 'CID' in text_list[i]]
        self.entities_list = {}
        self.entities = {}
        self.correct_answers = []

        self.extract_entities()
        self.extract_correct_answers()

    def extract_entities(self):
        entities_list = [t.replace('\n', '').split('\t')[5] for t in self.entities_list_text if t.replace('\n', '').split('\t')[5] != '-1']
        for e in entities_list:
            if '|' not in e:
                self.entities[e] = []
        for i, e in enumerate(self.entities_list_text):
            t = e.replace('\n', '').split('\t')
            entity = t[3]
            start = t[1]
            end = t[2]
            entity_type = t[4]
            entity_code = t[5]
            if entity_code != '-1' and '|' not in entity_code:
                self.entities[entity_code].append({
                    'start': start,
                    'end': end
                })
                self.entities_list[entity_code] = {
                    'text': entity.lower(),
                    'entity_type': entity_type
                }
            
    
    def extract_correct_answers(self):
        for t in self.ca_list_text:
            e1 = t.replace('\n', '').split('\t')[2]
            e2 = t.replace('\n', '').split('\t')[3]
            if '|' not in e1 and '|' not in e2 and e1 in self.entities.keys() and e2 in self.entities.keys():
                self.correct_answers.append((e1, e2))
        # self.correct_answers = [(t.replace('\n', '').split('\t')[2], t.replace('\n', '').split('\t')[3]) for t in self.ca_list_text]

    def check_distance_CA(self):
        count = 0
        for entity_pair in self.correct_answers:
            if entity_pair[1] not in self.entities:
                print('self.entities : ', self.entities)
                continue
            if check_sentence_contain_entities(self.text, self.entities[entity_pair[0]], self.entities[entity_pair[1]]):
                count += 1
        return count

    def get_list_code_by_type(self, type='Chemical'):
        codes = []
        for entity_code in self.entities_list:
            if self.entities_list[entity_code]['entity_type'] == type:
                codes.append(entity_code)
        return codes

    def make_example(self, use_entity_token: bool = True):
        from sklearn import preprocessing
        final_sample = []
        masked_entities = []
        text = self.text
        entities_pos = []
        for entity_code in self.entities:
            for pos in self.entities[entity_code]:
                start = int(pos['start'])
                end = int(pos['end'])
                entities_pos.append({
                    'code': entity_code,
                    'start': start,
                    'end': end
                })
        entities_pos_sorted = sorted(entities_pos, key=itemgetter('start')) 
        for pos in reversed(entities_pos_sorted):
            start = int(pos['start'])
            end = int(pos['end'])
            new_text = text[:start] + ' <e> ' + text[start: end] + ' </e> ' + text[end:]
            text = new_text
        
        # print('new_text: ', text)

        e_start_ids = self.tokenize.convert_tokens_to_ids('<e>')
        e_end_ids = self.tokenize.convert_tokens_to_ids('</e>')

        # print('e_start_ids: ', e_start_ids)

        text_tokenized = self.tokenize.encode(text)

        if len(text_tokenized) > 512:
            subset = text_tokenized[0:512]
            e_end_last_idx = [i for i in range(len(subset)) if subset[i] == e_end_ids]
            end_idx = e_end_last_idx[-1] + 1
            text_tokenized = text_tokenized[0: end_idx]
        # print('text_tokenized: ', text_tokenized)
        # print('text_tokenized_tokens: ', self.tokenize.convert_ids_to_tokens(text_tokenized))
        ids = 0
        entity_check = 0
        while ids < len(text_tokenized):
            if text_tokenized[ids] != e_start_ids:
                masked_entities.append('O')
                ids += 1
            else:
                if use_entity_token:
                    masked_entities.append(entities_pos_sorted[entity_check]['code'])
                ids += 1
                while text_tokenized[ids] != e_end_ids:
                    masked_entities.append(entities_pos_sorted[entity_check]['code'])
                    ids += 1
                if use_entity_token:
                    masked_entities.append(entities_pos_sorted[entity_check]['code'])
                entity_check += 1
                ids += 1

        if not use_entity_token:
            text_tokenized = list(filter(lambda a: a != e_start_ids and a != e_end_ids, text_tokenized))
        
        # print('text_tokenized_2: ', text_tokenized)
        if len(masked_entities) != len(text_tokenized):
            print('Ops ! Something wrong, len of masked_entities is {}, while len of text_tokenized is {}'.format(len(masked_entities), len(text_tokenized)))
        le = preprocessing.LabelEncoder()
        # print(' masked_entities: ', masked_entities)
        masked_entities_encoded = le.fit_transform(masked_entities)
        # print(' masked_entities: ', masked_entities)
        chemical_code_list = self.get_list_code_by_type(type='Chemical')
        disease_code_list = self.get_list_code_by_type(type='Disease')
        other_code_list = self.get_list_code_by_type(type='Other')
        # print('len_chemical_code_list: ', len(chemical_code_list))
        # print('len_disease_code_list: ', len(disease_code_list))
        # print('len_other_code_list: ', len(other_code_list))
        check_pair_code = []
        for chemical_code in chemical_code_list:
            if chemical_code not in masked_entities:
                continue
            for disease_code in disease_code_list:
                if disease_code not in masked_entities:
                    continue
                for other_code in other_code_list:
                    if other_code not in masked_entities:
                        continue
                    if ((chemical_code, disease_code) not in check_pair_code) and ((disease_code, chemical_code) not in check_pair_code):
                        # print('chemical_code: ', chemical_code)
                        # print('disease_code: ', disease_code)
                        # print('check_pair_code: ', check_pair_code)
                        # print(((chemical_code, disease_code) not in check_pair_code) or ((disease_code, chemical_code) not in check_pair_code))
                        check_pair_code.append((chemical_code, disease_code))
                        
                        if (chemical_code, disease_code) in self.correct_answers or (disease_code, chemical_code) in self.correct_answers:
                            # print('check')
                            
                            final_sample.append({
                                'text_tokenized': text_tokenized,
                                'masked_entities': masked_entities_encoded,
                                'chemical_code': le.transform([chemical_code])[0],
                                'disease_code': le.transform([disease_code])[0],
                                'other_code': -1,
                                'label': 1
                            })
                        else:
                            # print('check')
                            final_sample.append({
                                'text_tokenized': text_tokenized,
                                'masked_entities': masked_entities_encoded,
                                'chemical_code': le.transform([chemical_code])[0],
                                'disease_code': le.transform([disease_code])[0],
                                'other_code': -1,
                                'label': 0
                            })
                        
                    if ((other_code, disease_code) not in check_pair_code) and ((other_code, disease_code) not in check_pair_code):
                        check_pair_code.append((other_code, disease_code))
                        if (other_code, disease_code) in self.correct_answers or (disease_code, other_code) in self.correct_answers:
                            # print('check')
                            # print('other_code: ', other_code)
                            # print('disease_code: ', disease_code)
                            final_sample.append({
                                'text_tokenized': text_tokenized,
                                'masked_entities': masked_entities,
                                'chemical_code': -1,
                                'disease_code': le.transform([disease_code])[0],
                                'other_code': le.transform([other_code])[0],
                                'label': 1
                            })
                        else:
                            # print('check')
                            final_sample.append({
                                'text_tokenized': text_tokenized,
                                'masked_entities': masked_entities,
                                'chemical_code': -1,
                                'disease_code': le.transform([disease_code])[0],
                                'other_code': le.transform([other_code])[0],
                                'label': 0
                            })
                    if ((other_code, chemical_code) not in check_pair_code) and ((chemical_code, other_code) not in check_pair_code):
                        check_pair_code.append((other_code, chemical_code))
                        if (other_code, chemical_code) in self.correct_answers or (chemical_code, other_code) in self.correct_answers:
                            # print('check')
                            # print('other_code: ', other_code)
                            # print('chemical_code: ', chemical_code)
                            final_sample.append({
                                'text_tokenized': text_tokenized,
                                'masked_entities': masked_entities,
                                'chemical_code': le.transform([chemical_code])[0],
                                'disease_code': -1,
                                'other_code': le.transform([other_code])[0],
                                'label': 1
                            })
                        else:
                            # print('check')
                            final_sample.append({
                                'text_tokenized': text_tokenized,
                                'masked_entities': masked_entities,
                                'chemical_code': le.transform([chemical_code])[0],
                                'disease_code': -1,
                                'other_code': le.transform([other_code])[0],
                                'label': 0
                            })
        # print('check_pair_code: ', check_pair_code)
        # if len(text_tokenized) > 512:
        #     subset = masked_entities[511:]
        #     subset = list(set(subset))
            
        return final_sample




    
def gen_samples(data_text_raw):
    data = []
    data_cluster = []
    for ids in range(len(data_text_raw)):
        if data_text_raw[ids] != '\n': 
            data_cluster.append(data_text_raw[ids])
        else:
            if len(data_cluster) != 0:
                data.append(data_cluster)
            data_cluster = []
    return data

    

def make_cdr_non_global_dataset(path, use_entity_token=True, batch_size=16, shuffle=True, num_workers=0, extract_type='intra'):
    data = []
    tokenizer = XLNetTokenizer.from_pretrained('model_sentence_piece/wiki-ja.model')
    tokenizer.add_tokens('<e>')
    tokenizer.add_tokens('</e>')
    with open(path, 'r', encoding="utf8") as f:
        raw_data = f.readlines()
    data_raw_sample = gen_samples(raw_data)
    for text_block in tqdm(data_raw_sample):
        sample = ALPS_Sample(text_list=text_block, tokenize=tokenizer)
        final_sample = sample.make_example(use_entity_token=use_entity_token)
        data += final_sample
    PS = PadSequenceALPSDataset(token_pad_value=tokenizer.convert_tokens_to_ids('[PAD]'))
    dataset = ALPSDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=PS,
                             pin_memory=False)
    return data, data_loader

class ALPSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tokenized = self.data[idx]['text_tokenized']
        masked_entities_encoded = self.data[idx]['masked_entities']
        chemical_code = self.data[idx]['chemical_code']
        disease_code = self.data[idx]['disease_code']
        other_code = self.data[idx]['other_code']
        label = self.data[idx]['label']
        return torch.tensor(text_tokenized), torch.tensor(masked_entities_encoded), torch.tensor(chemical_code), torch.tensor(disease_code), torch.tensor(other_code), torch.tensor(label, dtype=torch.long)

