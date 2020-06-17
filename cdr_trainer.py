from torch import nn
import torch
from typing import *

from models.huggingface_models.modeling_electra import ElectraModel
from data_loaders.cdr_dataset import make_cdr_dataset
from tqdm import tqdm
from utils.trainer_utils import get_tokenizer

from sklearn.metrics import classification_report

import os


cuda = torch.cuda.is_available()



def evaluate(net, test_loader, tokenizer):
    net.eval()
    pad_id = tokenizer.pad_token_id
    labels = []
    preds = []
    ner_preds = []
    ner_labels = []

    for i, batch in tqdm(enumerate(test_loader)):
        x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
        label = torch.squeeze(label, 1).to('cpu')
        label = label.data
        labels.append(label)
        attention_mask = (x != pad_id).float()
        attention_mask = (1. - attention_mask) * -10000.
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if cuda:
            x = x.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        
        prediction = net(x, token_type_ids=token_type_ids, attention_masks=attention_mask,
                                  used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                                  chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs)
        prediction.to('cpu')
        pred_label = prediction.argmax(dim=1).to('cpu')
        
        preds.append(pred_label)
    labels = torch.cat(labels, dim=-1)
    preds = torch.cat(preds, dim=-1)
    print("report: ", classification_report(labels, preds))


def train(num_epochs=4):

    train_loader = make_cdr_dataset('data/cdr/CDR_TrainingSet.PubTator.txt')
    test_loader = make_cdr_dataset('data/cdr/CDR_TestSet.PubTator.txt')

    tokenizer = get_tokenizer()
    net = ElectraModel.from_pretrained('google/electra-small-discriminator')

    if cuda:
        net.cuda()

    criteria = torch.nn.CrossEntropyLoss(reduction='mean')

    pad_id = tokenizer.pad_token_id

    def train_model(optimizer=None, tokenizer=None):
        net.train()
        epoch_loss = 0
        all_labels = []
        all_preds = []
        for i, batch in tqdm(enumerate(train_loader)):
            x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
            print('label = ', label)
            # label = torch.squeeze(label, 1)
            attention_mask = (x != pad_id).float()
            attention_mask = (1. - attention_mask) * -10000.
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            if cuda:
                x = x.cuda()
                label = label.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()


            prediction = net(x, token_type_ids=token_type_ids, 
                                # attention_masks=attention_mask,
                                  used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                                  chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs)
            
            loss = criteria(prediction, label)
            pred = prediction.argmax(dim=-1)
            all_labels.append(label.data.to('cpu'))
            all_preds.append(pred.to('cpu'))
            
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        average_loss = epoch_loss / i
        
        print("average RE loss : ", average_loss)
        print("train_cls report: ", classification_report(torch.cat(all_labels, dim=-1), torch.cat(all_preds, dim=-1)))
        evaluate(net, test_loader, tokenizer)

    for epoch in range(num_epochs):
        print("EPOCH: ", epoch)
        optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.0001}])
        train_model(optimizer=optimizer, tokenizer=tokenizer)


if __name__ == '__main__':
    train()