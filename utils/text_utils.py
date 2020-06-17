import nltk
import numpy as np
import os
from typing import List, Dict

def check_sentence_contain_entities(text, entity_chemical: Dict, entity_disease: Dict):
    def check_position(start, sent_start, sent):
        if start >= sent_start and start <= sent_start + len(sent) - 1:
            return True
        return False
    text_sents = nltk.tokenize.sent_tokenize(text)
    for i, sent in enumerate(text_sents):
        for entity_c in entity_chemical:
            for entity_d in entity_disease:
                if check_position(int(entity_c['start']), text.find(sent), sent) and check_position(int(entity_d['start']), text.find(sent), sent):
                    return True
    return False

# def correct_answer_analysis()