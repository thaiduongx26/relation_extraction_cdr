import pandas as pd
import pickle
import unicodedata
import copy


def get_entities(entities_file):
    with open(entities_file, 'rb') as f:
        all_entities = pickle.load(f)
    all_entities = [[unicodedata.normalize('NFKC', x) for x in sample] for sample in all_entities]
    return all_entities


def read_data(excel_path, all_label_file, all_cause_file, all_effect_file):
    all_entities = get_entities(all_label_file)
    all_entities_flatten = list(set([x for entities in all_entities for x in entities]))
    entity_dict = {x: i for i, x in enumerate(all_entities_flatten)}
    all_cause = get_entities(all_cause_file)
    all_effect = get_entities(all_effect_file)

    df = pd.read_excel(excel_path)
    sample_index = df['Sample'].values
    texts = df['Japanese'].values
    current_sample_index = 0
    current_text = ''
    all_samples = []
    for i in range(len(texts)):
        index = str(sample_index[i]).lower()

        if index != 'nan' and len(current_text) != 0:
            assert int(float(index)) == current_sample_index + 1
            current_text = unicodedata.normalize('NFKC', current_text)
            current_sample = {'text': current_text, 'entities': all_entities[current_sample_index],
                              'cause': all_cause[current_sample_index], 'effect': all_effect[current_sample_index]}
            all_samples.append(current_sample)
            current_sample_index += 1
            current_text = texts[i]
        else:
            current_text += texts[i]
    return all_samples, entity_dict


def write_to_text(file_cursor, sample, index):
    title = str(index) + '|t|' + '\n'
    abstract = str(index) + '|a|' + sample['text'] + '\n'
    file_cursor.write(title)
    file_cursor.write(abstract)
    for i in range(len(sample['entities'])):
        entity = sample['entities'][i]
        text = str(index) + '\t' + str(entity['start'] + 1) + '\t' + str(
            entity['start'] + 1 + len(entity['text'])) + '\t' + entity['text'] + '\t' + entity['label'] + '\t' + str(
            entity['code']) + '\n'
        file_cursor.write(text)
    for i in range(len(sample['relations'])):
        relation = sample['relations'][i]
        text = str(index) + '\t' + 'CID' + '\t' + str(relation[0]) + '\t' + str(relation[1]) + '\n'
        file_cursor.write(text)
    file_cursor.write('\n')
    pass


def convert_sample_to_text(sample, entity_dict, file_cursor, index):
    text = sample['text']
    entities = sample['entities']
    entities = sorted(entities, key=lambda x: len(x), reverse=True)  # sort by len
    new_entities = []
    for entity in entities:
        a_entity = {'text': entity, 'code': entity_dict[entity]}
        if entity in sample['cause']:
            a_entity['label'] = 'Chemical'
        elif entity in sample['effect']:
            a_entity['label'] = 'Disease'
        else:
            a_entity['label'] = 'Other'
        new_entities.append(a_entity)
    entities = new_entities
    entities_processed_list = []
    relation_list = []
    i = 0
    while i < len(text):
        for entity in entities:
            if text[i:].startswith(entity['text']):
                entity['is_checked'] = True
                a_entity = copy.deepcopy(entity)
                a_entity['start'] = i
                entities_processed_list.append(a_entity)
                i += len(entity['text'])
        i += 1
    not_checked_entity = []
    for entity in entities:
        if 'is_checked' not in entity:
            not_checked_entity.append(entity['code'])
            print(entity)
            print("text: ", text)
    for i in range(len(sample['cause'])):
        for j in range(len(sample['effect'])):
            if entity_dict[sample['cause'][i]] not in not_checked_entity and entity_dict[
                sample['effect'][j]] not in not_checked_entity:
                relation_list.append((entity_dict[sample['cause'][i]], entity_dict[sample['effect'][j]]))
    sample_processed = {'text': text, 'entities': entities_processed_list, 'relations': relation_list}
    write_to_text(file_cursor, sample_processed, index)


if __name__ == '__main__':
    data = 'D:\\relation_extraction_cdr\\data\\all_data_single_ca_translate_columns.xlsx'
    all_label_file = 'D:\\relation_extraction_cdr\\data\\all_label_all.l'
    all_cause_file = 'D:\\relation_extraction_cdr\\data\\cause_label_all.l'
    all_effect_file = 'D:\\relation_extraction_cdr\\data\\effect_label_all.l'
    all_samples, entity_dict = read_data(data, all_label_file, all_cause_file, all_effect_file)
    print("all_samples: ", all_samples)
    train_file = 'D:\\relation_extraction_cdr\\data/alps_train.txt'
    test_file = 'D:\\relation_extraction_cdr\\data/alps_test.txt'

    train_file_cursor = open(train_file, 'w', encoding='utf-8')
    test_file_cursor = open(test_file, 'w', encoding='utf-8')

    all_indexes = range(len(all_samples))
    import random

    test_index = random.sample(all_indexes, int(0.25 * len(all_indexes)))
    train_index = [x for x in all_indexes if x not in test_index]
    index = 22836123
    from tqdm import tqdm

    for i in tqdm(range(len(all_samples))):
        if i in train_index:
            f = train_file_cursor
        else:
            f = test_file_cursor
        convert_sample_to_text(all_samples[i], entity_dict, f, index + i)
