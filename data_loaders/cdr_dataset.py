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
from data_loaders.sequence_padding import PadSequenceCDRDataset, PadSequenceCDRSentenceDataset


class CDR_Sample():
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
            new_text = text[:start] + ' [E] ' + text[start: end] + ' [/E] ' + text[end:]
            text = new_text
        
        e_start_ids = self.tokenize.convert_tokens_to_ids('[e]')
        e_end_ids = self.tokenize.convert_tokens_to_ids('[/e]')

        text_tokenized = self.tokenize.encode(text)
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
        
        if len(masked_entities) != len(text_tokenized):
            print('Ops ! Something wrong, len of masked_entities is {}, while len of text_tokenized is {}'.format(len(masked_entities), len(text_tokenized)))
        le = preprocessing.LabelEncoder()
        masked_entities = le.fit_transform(masked_entities)
        chemical_code_list = self.get_list_code_by_type(type='Chemical')
        disease_code_list = self.get_list_code_by_type(type='Disease')
        for chemical_code in chemical_code_list:
            for disease_code in disease_code_list:
                if (chemical_code, disease_code) in self.correct_answers:
                    final_sample.append({
                        'text_tokenized': text_tokenized,
                        'masked_entities': masked_entities,
                        'chemical_code': le.transform([chemical_code])[0],
                        'disease_code': le.transform([disease_code])[0],
                        'label': 1
                    })
                else:
                    final_sample.append({
                        'text_tokenized': text_tokenized,
                        'masked_entities': masked_entities,
                        'chemical_code': le.transform([chemical_code])[0],
                        'disease_code': le.transform([disease_code])[0],
                        'label': 0
                    })
        
        # if len(text_tokenized) > 512:
        #     subset = masked_entities[511:]
        #     subset = list(set(subset))
            
        return final_sample, text_tokenized

    def check_entity_in_CA(self, entity):
        ca_entities = []
        for e in self.correct_answers:
            ca_entities.append(e[0])
            ca_entities.append(e[1])
        if entity in ca_entities:
            return True
        return False

    # def samples_append(self, final_sample, chemical_code, disease_code):
    #     if (chemical_code, disease_code) in self.correct_answers:
    #         final_sample.append({
    #             'text_tokenized': text_tokenized,
    #             'masked_entities': masked_entities,
    #             'chemical_code': le.transform([chemical_code])[0],
    #             'disease_code': le.transform([disease_code])[0],
    #             'label': 1
    #         })
    #     else:
    #         final_sample.append({
    #             'text_tokenized': text_tokenized,
    #             'masked_entities': masked_entities,
    #             'chemical_code': le.transform([chemical_code])[0],
    #             'disease_code': le.transform([disease_code])[0],
    #             'label': 0
    #         })
    #     return final_sample

    def process_masked_entities(self, entities_pos_sorted, text_tokenized, use_entity_token):
        masked_entities = []
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
        
        if len(masked_entities) != len(text_tokenized):
            print('Ops ! Something wrong, len of masked_entities is {}, while len of text_tokenized is {}'.format(len(masked_entities), len(text_tokenized)))
        return masked_entities

    def extract_intra_inter_sentence(self, extract_inter=False):
        from sklearn import preprocessing
        final_sample_intra = []
        final_sample_inter = []
        global_sample = []
        text = self.text
        chemical_code_list = self.get_list_code_by_type(type='Chemical')
        disease_code_list = self.get_list_code_by_type(type='Disease')
        for chemical_code in chemical_code_list:
            for disease_code in disease_code_list:
                # print(text)
                # print(chemical_code)
                # print(disease_code)
                # print('self.entities: ', self.entities)
                # print('self.entities_list: ', self.entities_list)
                # print('self.correct_answers: ', self.correct_answers)
                data, intra_check = extract_nearest_sentences_contain_2_entities(text, chemical_code, disease_code, self.entities, self.entities_list, self.correct_answers, extract_inter=extract_inter)
                if data != None:
                    if intra_check:
                        final_sample_intra.append(data)
                    elif extract_inter:
                        final_sample_inter.append(data)
                    elif not extract_inter:
                        global_sample.append(data)
            
        return final_sample_intra, final_sample_inter, global_sample


    def make_example_non_global(self, use_entity_token: bool = True, extract_type: str = 'inter'):
        from sklearn import preprocessing
        final_sample = []
        text = self.text
        entities_pos = []
        chemical_code_list = self.get_list_code_by_type(type='Chemical')
        disease_code_list = self.get_list_code_by_type(type='Disease')
        data_intra, data_inter, _ = self.extract_intra_inter_sentence(extract_inter=True)
        e_start_ids = self.tokenize.convert_tokens_to_ids('[e]')
        e_end_ids = self.tokenize.convert_tokens_to_ids('[/e]')

        if extract_type == 'intra':
            data = data_intra
        elif extract_type == 'inter':
            data = data_inter

        for sample in data:
            re_sample = sample
            masked_entities = []
            chemical_start = int(sample['chemical_pos']['start']) - sample['sent_pos']
            chemical_end = int(sample['chemical_pos']['end']) - sample['sent_pos']
            disease_start = int(sample['disease_pos']['start']) - sample['sent_pos']
            disease_end = int(sample['disease_pos']['end']) - sample['sent_pos']
            first, second = '', ''
            if chemical_start > disease_start:
                new_text = sample['sentence'][:chemical_start] + ' [E] ' + sample['sentence'][chemical_start: chemical_end] + ' [/E] ' + sample['sentence'][chemical_end:]
                new_text = new_text[:disease_start] + ' [E] ' + new_text[disease_start: disease_end] + ' [/E] ' + new_text[disease_end:]
                first = sample['entity_disease']
                second = sample['entity_chemical']
            else:
                new_text = sample['sentence'][:disease_start] + ' [E] ' + sample['sentence'][disease_start: disease_end] + ' [/E] ' + sample['sentence'][disease_end:]
                new_text = new_text[:chemical_start] + ' [E] ' + new_text[chemical_start: chemical_end] + ' [/E] ' + new_text[chemical_end:]
                first = sample['entity_chemical']
                second = sample['entity_disease']
            
            # print('first: ', first)
            # print('second: ', second)
            # print('text: ', new_text)
            text_tokenized = self.tokenize.encode(new_text)
            if len(text_tokenized) > 512:
                continue
            ids = 0
            entity_check = 0
            while ids < len(text_tokenized):
                if text_tokenized[ids] != e_start_ids:
                    masked_entities.append('O')
                    ids += 1
                else:
                    if use_entity_token:
                        if entity_check == 0:
                            masked_entities.append(first)
                        else:
                            masked_entities.append(second)
                    ids += 1
                    while text_tokenized[ids] != e_end_ids:
                        if entity_check == 0:
                            masked_entities.append(first)
                        else:
                            masked_entities.append(second)
                        ids += 1
                    if use_entity_token:
                        if entity_check == 0:
                            masked_entities.append(first)
                        else:
                            masked_entities.append(second)
                    entity_check += 1
                    ids += 1
            
            # print('masked_entities: ', masked_entities)

            if not use_entity_token:
                text_tokenized = list(filter(lambda a: a != e_start_ids and a != e_end_ids, text_tokenized))

            if len(masked_entities) != len(text_tokenized):
                print('Ops ! Something wrong, len of masked_entities is {}, while len of text_tokenized is {}'.format(len(masked_entities), len(text_tokenized)))
            le = preprocessing.LabelEncoder()
            masked_entities = le.fit_transform(masked_entities)

            re_sample['entity_chemical'] = le.transform([re_sample['entity_chemical']])[0]
            re_sample['entity_disease'] = le.transform([re_sample['entity_disease']])[0]
            re_sample['sentence_tokenized_ids'] = text_tokenized
            re_sample['sentence_tokenized'] = self.tokenize.convert_ids_to_tokens(text_tokenized)
            re_sample['masked_entities'] = masked_entities
            final_sample.append(re_sample)
        
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




def test_extract_data(path):
    data = []
    tokenizer = get_tokenizer()
    list_data_intra, list_data_inter, list_data_global = [], [], []
    # count = 0
    with open(path, 'r') as f:
        raw_data = f.readlines()
    data_raw_sample = gen_samples(raw_data)
    list_samples = []
    for text_block in data_raw_sample:
        sample = CDR_Sample(text_list=text_block, tokenize=tokenizer)
        # count += sample.check_distance_CA()
        data_intra, data_inter, data_global = sample.extract_intra_inter_sentence(extract_inter=True)
        list_data_intra += data_intra
        list_data_inter += data_inter
        list_data_global += data_global
    return list_data_intra, list_data_inter, list_data_global
    

def make_cdr_train_non_global_dataset(train_path, dev_path, use_entity_token=False, batch_size=16, shuffle=True, num_workers=0, extract_type='intra'):
    data = []
    tokenizer = get_tokenizer()
    # count = 0
    with open(train_path, 'r') as f:
        raw_data_train = f.readlines()
    with open(dev_path, 'r') as f:
        raw_data_dev = f.readlines()
    data_raw_sample_train = gen_samples(raw_data_train)
    data_raw_sample_dev = gen_samples(raw_data_dev)
    for text_block in data_raw_sample_train:
        sample = CDR_Sample(text_list=text_block, tokenize=tokenizer)
        final_sample = sample.make_example_non_global(use_entity_token=use_entity_token, extract_type=extract_type)
        data += final_sample
    for text_block in data_raw_sample_dev:
        sample = CDR_Sample(text_list=text_block, tokenize=tokenizer)
        final_sample = sample.make_example_non_global(use_entity_token=use_entity_token, extract_type=extract_type)
        data += final_sample
    PS = PadSequenceCDRSentenceDataset(token_pad_value=tokenizer.pad_token_id)
    dataset = CDRIntraDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=PS,
                             pin_memory=False)
    return data, data_loader

def make_cdr_non_global_dataset(path, use_entity_token=False, batch_size=16, shuffle=True, num_workers=0, extract_type='intra'):
    data = []
    tokenizer = get_tokenizer()
    # count = 0
    with open(path, 'r') as f:
        raw_data = f.readlines()
    data_raw_sample = gen_samples(raw_data)
    for text_block in data_raw_sample:
        sample = CDR_Sample(text_list=text_block, tokenize=tokenizer)
        final_sample = sample.make_example_non_global(use_entity_token=use_entity_token, extract_type=extract_type)
        data += final_sample
    PS = PadSequenceCDRSentenceDataset(token_pad_value=tokenizer.pad_token_id)
    dataset = CDRIntraDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=PS,
                             pin_memory=False)
    return data, data_loader

class CDRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tokenized = self.data[idx]['text_tokenized']
        masked_entities_encoded = self.data[idx]['masked_entities']
        chemical_code = self.data[idx]['chemical_code']
        disease_code = self.data[idx]['disease_code']
        label = self.data[idx]['label']
        return torch.tensor(text_tokenized), torch.tensor(masked_entities_encoded), torch.tensor(chemical_code), torch.tensor(disease_code), torch.tensor(label)

class CDRIntraDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tokenized = self.data[idx]['sentence_tokenized_ids']
        masked_entities_encoded = self.data[idx]['masked_entities']
        chemical_code = self.data[idx]['entity_chemical']
        disease_code = self.data[idx]['entity_disease']
        label = self.data[idx]['label']
        return torch.tensor(text_tokenized), torch.tensor(masked_entities_encoded), torch.tensor(chemical_code), torch.tensor(disease_code), torch.tensor(label)