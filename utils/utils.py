import pickle
import numpy as np

TRAINING_PATH = 'data/cdr/CDR_TrainingSet.PubTator.txt'
TESTING_PATH = 'data/cdr/CDR_TestSet.PubTator.txt'

def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('save successful!!!!!!')