import nltk
import numpy as np
import os
from typing import List, Dict


def check_position(start, sent_start, sent):
    if start >= sent_start and start <= sent_start + len(sent) - 1:
        return True
    return False

def compute_token_distance(entity_chemical, entity_chemical_start, entity_disease, entity_disease_start, sent, sent_start):
    pos_start = (entity_chemical_start + len(entity_chemical) if entity_chemical_start < entity_disease_start else entity_disease_start + len(entity_disease))
    pos_end = (entity_chemical_start if entity_chemical_start > entity_disease_start else entity_disease_start)
    tokens_between_2_entities = sent[pos_start - sent_start: pos_end - sent_start].split(' ')
    return len(tokens_between_2_entities)

# def compute_sentence_distance(entity_chemical, entity_chemical_start, entity_disease, entity_disease_start, sent, sent_start):

    

def check_multiple_positions(entity_chemical, entity_disease, entities_pos, sent_start, sent, correct_answers):
    entity_chemical_positions = entities_pos[entity_chemical]
    entity_disease_positions = entities_pos[entity_disease]
    for chemical_pos in entity_chemical_positions:
        for disease_pos in entity_disease_positions:
            chemical_start = chemical_pos['start']
            disease_start = disease_pos['start']
            if check_position(int(chemical_start), sent_start, sent) and check_position(int(disease_start), sent_start, sent):
                block = {
                    'sentence': sent,
                    'sent_pos': sent_start,
                    'entity_chemical': entity_chemical,
                    'entity_disease': entity_disease,
                    'chemical_pos': chemical_pos,
                    'disease_pos': disease_pos
                }
                if (entity_chemical, entity_disease) in correct_answers:
                    block.update({'label': 1})
                else:
                    block.update({'label': 0})
                return block
    return False

def check_sentence_contain_entities(text, entity_chemical: Dict, entity_disease: Dict):
    text_sents = nltk.tokenize.sent_tokenize(text)
    for i, sent in enumerate(text_sents):
        for entity_c in entity_chemical:
            for entity_d in entity_disease:
                if check_position(int(entity_c['start']), text.find(sent), sent) and check_position(int(entity_d['start']), text.find(sent), sent):
                    return True
    return False

def extract_nearest_sentences_contain_2_entities(text, entity_chemical: str, entity_disease: str, entities_pos: Dict, entities_texts: List, correct_answers: List, extract_inter: bool):
    def block_data(sent, sent_start, entity_chemical, entity_disease, chemical_pos, disease_pos, correct_answers):
        block = {
                'sentence': sent,
                'sent_pos': sent_start,
                'entity_chemical': entity_chemical,
                'entity_disease': entity_disease,
                'chemical_pos': chemical_pos,
                'disease_pos': disease_pos
            }
        if (entity_chemical, entity_disease) in correct_answers:
            block.update({'label': 1})
        else:
            block.update({'label': 0})
        return block

    # def check_entity_position():


    text_sents = nltk.tokenize.sent_tokenize(text)
    distances = []
    check = False
    data = []
    data_non_token_distance = []
    distances_2 = []
    chemical_sent_index = []
    disease_sent_index = []
    chemical_sent_start = []
    disease_sent_start = []
    entity_chemical_positions = entities_pos[entity_chemical]
    entity_disease_positions = entities_pos[entity_disease]
    for i, sent in enumerate(text_sents):
        sent_start = text.find(sent)
        for chemical_pos in entity_chemical_positions:
            for disease_pos in entity_disease_positions:
                chemical_start = int(chemical_pos['start'])
                disease_start = int(disease_pos['start'])
                if check_position(int(chemical_start), sent_start, sent) and check_position(int(disease_start), sent_start, sent):
                    
                    token_distance = compute_token_distance(entity_chemical, chemical_start, entity_disease, disease_start, sent, sent_start)
                    if token_distance < 10:
                        distances.append(token_distance)
                        data.append(block_data(sent, sent_start, entity_chemical, entity_disease, chemical_pos, disease_pos, correct_answers))
                    else:
                        check = True
                        distances_2.append(token_distance)
                        data_non_token_distance.append(block_data(sent, sent_start, entity_chemical, entity_disease, chemical_pos, disease_pos, correct_answers))
                else:
                    if check_position(int(chemical_start), sent_start, sent):
                        chemical_sent_index.append(i)
                    elif check_position(int(disease_start), sent_start, sent):
                        disease_sent_index.append(i)


    if len(data) == 0:
        if extract_inter:
            if check:
                index = distances_2.index(min(distances_2))
                return data_non_token_distance[index], False
            else:
                chemical_sent_index = list(set(chemical_sent_index))
                disease_sent_index = list(set(disease_sent_index))
                min_distance = 10000
                start_sentence = 0
                end_sentence = 0
                for chemical_sent_i in chemical_sent_index:
                    for disease_sent_i in disease_sent_index:
                        if abs(chemical_sent_i - disease_sent_i) < min_distance:
                            min_distance = abs(chemical_sent_i - disease_sent_i)
                            start_sentence = (chemical_sent_i if chemical_sent_i < disease_sent_i else disease_sent_i)
                            end_sentence = (chemical_sent_i if chemical_sent_i > disease_sent_i else disease_sent_i)

                if min_distance <= 3:
                    start_pos_sentence = text.find(text_sents[start_sentence])
                    end_pos_sentence = text.find(text_sents[end_sentence]) + len(text_sents[end_sentence])
                    new_text = text[start_pos_sentence: end_pos_sentence]
                    
                    entity_chemical_text = entities_texts[entity_chemical]['text']
                    entity_disease_text = entities_texts[entity_disease]['text']
                    chemical_start = text_sents[start_sentence].find(entity_chemical_text)
                    disease_start = text_sents[end_sentence].find(entity_disease_text)

                    if chemical_start == -1 or disease_start == -1:
                        chemical_start = text_sents[end_sentence].find(entity_chemical_text) + text.find(text_sents[end_sentence])
                        disease_start = text_sents[start_sentence].find(entity_disease_text) + start_pos_sentence
                    else:
                        chemical_start += start_pos_sentence
                        disease_start += text.find(text_sents[end_sentence])
                    
                    return block_data(new_text, start_pos_sentence, entity_chemical, entity_disease, {'start': str(chemical_start), 'end': str(chemical_start + len(entity_chemical_text))}, {'start': str(disease_start), 'end': str(disease_start + len(entity_disease_text))}, correct_answers), False
                else:
                    return None, False
        else:
            return block_data(text, None, entity_chemical, entity_disease, None, None, correct_answers), False
    else:
        index = distances.index(min(distances))
        return data[index], True

def extract_sentence_contain_both_2_entities(text, entities_chemical: List, entities_disease: List, entities_pos: Dict, correct_answers: List):
    text_sents = nltk.tokenize.sent_tokenize(text)
    data = []
    for entity_c in entities_chemical:
        for entity_d in entities_disease:
            for i, sent in enumerate(text_sents):
                data_e = check_multiple_positions(entity_c, entity_d, entities_pos, text.find(sent), sent, correct_answers)
                if data_e:
                    data.append(data_e)
                    break
    return data

# def correct_answer_analysis()