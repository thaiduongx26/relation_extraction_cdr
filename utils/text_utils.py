import nltk
import numpy as np
import os
from typing import List, Dict


def check_position(start, sent_start, sent):
    if start >= sent_start and start <= sent_start + len(sent) - 1:
        return True
    return False

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

def extract_sentence_contain_both_2_entities(text, entities_chemical: List, entities_disease: List, entities_pos: Dict, correct_answers: List):
    text_sents = nltk.tokenize.sent_tokenize(text)
    data = []
    for entity_c in entities_chemical:
        for entity_d in entities_disease:
            for i, sent in enumerate(text_sents):
                # if check_position(int(entity_c['start']), text.find(sent), sent) and check_position(int(entity_d['start']), text.find(sent), sent):
                #     return True
                data_e = check_multiple_positions(entity_c, entity_d, entities_pos, text.find(sent), sent, correct_answers)
                if data_e:
                    data.append(data_e)
                    break
    return data

# def correct_answer_analysis()