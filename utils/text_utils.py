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

def check_multiple_positions(entity_chemical, entity_disease, entities_pos, sent_start, sent, correct_answers):
    entity_chemical_positions = entities_pos[entity_chemical]
    entity_disease_positions = entities_pos[entity_disease]
    for chemical_pos in entity_chemical_positions:
        for disease_pos in entity_disease_positions:
            chemical_start = chemical_pos['start']
            disease_start = disease_pos['start']
            if check_position(int(chemical_start), sent_start, sent) and check_position(int(disease_start), sent_start, sent):

                if (entity_chemical, entity_disease) in correct_answers:
                    return {
                        'sentence': sent,
                        'sent_pos': sent_start,
                        'entity_chemical': entity_chemical,
                        'entity_disease': entity_disease,
                        'chemical_pos': chemical_pos,
                        'disease_pos': disease_pos,
                        'label': 1
                    }
                else:
                    return {
                        'sentence': sent,
                        'sent_pos': sent_start,
                        'entity_chemical': entity_chemical,
                        'entity_disease': entity_disease,
                        'chemical_pos': chemical_pos,
                        'disease_pos': disease_pos,
                        'label': 0
                    }
            
    return False

def check_sentence_contain_entities(text, entity_chemical: Dict, entity_disease: Dict):
    
    text_sents = nltk.tokenize.sent_tokenize(text)
    for i, sent in enumerate(text_sents):
        for entity_c in entity_chemical:
            for entity_d in entity_disease:
                if check_position(int(entity_c['start']), text.find(sent), sent) and check_position(int(entity_d['start']), text.find(sent), sent):
                    return True
    return False

def extract_nearest_sentences_contain_2_entities(text, entity_chemical: str, entity_disease: str, entities_chemical: List, entities_disease: List, entities_pos: Dict, correct_answers: List):
    text_sents = nltk.tokenize.sent_tokenize(text)
    distances = []
    check = False
    data = []
    for i, sent in enumerate(text_sents):
        entity_chemical_positions = entities_pos[entity_chemical]
        entity_disease_positions = entities_pos[entity_disease]
        for chemical_pos in entity_chemical_positions:
            for disease_pos in entity_disease_positions:
                chemical_start = chemical_pos['start']
                disease_start = disease_pos['start']
                token_distance = compute_token_distance(entity_chemical, entity_chemical_start, entity_disease, entity_disease_start, sent, sent_start)
                if check_position(int(chemical_start), sent_start, sent) and check_position(int(disease_start), sent_start, sent):
                    if token_distance < 10:
                        distances.append(token_distance)
                        if (entity_chemical, entity_disease) in correct_answers:
                            data.append({
                                'sentence': sent,
                                'sent_pos': sent_start,
                                'entity_chemical': entity_chemical,
                                'entity_disease': entity_disease,
                                'chemical_pos': chemical_pos,
                                'disease_pos': disease_pos,
                                'label': 1
                            })
                        else:
                            data.append({
                                'sentence': sent,
                                'sent_pos': sent_start,
                                'entity_chemical': entity_chemical,
                                'entity_disease': entity_disease,
                                'chemical_pos': chemical_pos,
                                'disease_pos': disease_pos,
                                'label': 0
                            })
    if len(data) = 0:
        return None
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