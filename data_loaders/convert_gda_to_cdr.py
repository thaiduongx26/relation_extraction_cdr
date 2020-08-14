from typing import *


class GDAExample(object):
    def __init__(self, abstract, entities_lines, relation, idx):
        self.astract = abstract
        self.entities_lines = entities_lines
        self.relation = relation
        self.id = idx

    def write_example(self, f):
        abstract = self.id + '|a|' + self.astract + '\n'
        f.write(abstract)
        for line in self.entities_lines:
            f.write(line + '\n')
        f.write(self.relation + '\n')
        f.write('\n')


def load_abstract(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    abstracts = []
    is_index = True
    current_sample = None
    for line in lines:
        if len(line.strip()) == 0:
            if not current_sample:
                continue
            assert current_sample['abstract']
            assert not is_index
            abstracts.append(current_sample)
            current_sample = None
            is_index = True
        elif is_index:
            assert len(line.strip().split()) == 1, line
            assert not current_sample
            current_sample = {'id': line.strip(), 'abstract': None}
            is_index = False
        else:
            current_sample['abstract'] = line.strip()
    return abstracts


def load_anns(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    anns = {}
    current_ann = None
    for line in lines:
        if len(line.strip()) == 0:
            continue
        else:
            line_splitted = line.split()
            id = line_splitted[0]
            s = line_splitted[1]
            e = line_splitted[2]
            type = line_splitted[-2]
            e_id = line_splitted[-1]

            entity = line_splitted[3:-2]
            # assert int(e) - int(s) == len(' '.join(entity)), f"{s}, {e}, {len(' '.join(entity))}, {id}, {' '.join(entity)}"
            entity = {'start': s, 'end': e, 'text': ' '.join(entity), 'type': type, 'entity_id': e_id}
            if id not in anns:
                anns[id] = []
            anns[id].append(entity)
    return anns


def load_labels(file):
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
    labels = {}
    for line in lines:
        line_splited = line.strip().split(',')
        id = line_splited[0]
        geneId = line_splited[1]
        diseaseId = line_splited[2]
        label = line_splited[3]
        if id not in labels:
            labels[id] = []
        labels[id].append({"geneId": geneId, 'diseaseId': diseaseId, 'label': label})
    return labels


def write_cdr_file(abstracts, anns, labels, file):
    with open(file, 'w') as f:
        for abstract in abstracts:
            id = abstract['id']
            f.write(id + '|a|' + abstract['abstract']+'\n')
            ann = anns[id]
            for entity in ann:
                f.write(id + '\t' + entity['start'] + '\t' + entity['end'] + '\t' + entity['text'] + '\t' + entity[
                    'type'] + '\t' + entity['entity_id'] + '\n')
            for label in labels[id]:
                f.write(id + '\t' + 'CID' + '\t' + label['geneId'] + '\t' + label['diseaseId']+'\n')
            f.write('\n')



if __name__ == '__main__':
    file = 'D:\\relation_extraction_cdr\\data\\GDA\\testing_data\\abstracts.txt'
    abstracts = load_abstract(file)
    anns_file = 'D:\\relation_extraction_cdr\\data\\GDA\\testing_data\\anns.txt'
    anns = load_anns(anns_file)
    labels_file = 'D:\\relation_extraction_cdr\\data\\GDA\\testing_data\\labels.csv'
    labels = load_labels(labels_file)
    assert len(labels) == len(abstracts)
    assert len(abstracts) == len(anns)
    file ='D:\\relation_extraction_cdr\\data\\GDA\\gda2cda/test.txt'
    write_cdr_file(abstracts, anns, labels, file)
    print("anns: ", anns.keys())
