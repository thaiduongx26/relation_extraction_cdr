from torch import nn
import torch
from typing import *

from transformers import AdamW

from models.electra_model import ElectraModelClassification, ElectraModelSentenceClassification, ElectraModelEntitySentenceClassification, ElectraModelEntityTokenClassification
from transformers import ElectraConfig
from data_loaders.cdr_dataset import make_cdr_train_non_global_dataset, make_cdr_non_global_dataset, make_train_pretrain_ner_dataset, make_pretrain_ner_dataset, make_train_joinlabel_dataset
from tqdm import tqdm
from utils.trainer_utils import get_tokenizer
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from optim import optim4GPU
# from torchsummary import summary
from sklearn.metrics import confusion_matrix
import numpy as np

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(0)
print('cuda is_available: ', cuda)


def evaluate(net, test_loader, tokenizer):
    net.eval()
    pad_id = tokenizer.pad_token_id
    labels = []
    preds = []

    for i, batch in tqdm(enumerate(test_loader)):
        x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
        # label = torch.squeeze(label, 1).to('cpu')
        label = label.data
        labels.append(label)
        attention_mask = (x != pad_id).float()
        attention_mask = (1. - attention_mask) * -10000.
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if cuda:
            x = x.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        
        prediction = net(x, token_type_ids=token_type_ids, 
                                # attention_masks=attention_mask,
                                  used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                                  chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs)
        prediction.to('cpu')
        pred_label = prediction.argmax(dim=1).to('cpu')
        
        preds.append(pred_label)
    
    new_all_labels = []
    new_all_preds = []
    for i in range(len(labels)):
        new_all_labels += labels[i].tolist()
        new_all_preds += preds[i].tolist()
    
    # labels = torch.cat(labels, dim=-1)
    # preds = torch.cat(preds, dim=-1)
    from sklearn.metrics import classification_report
    print("Testing report: ", classification_report(new_all_labels, new_all_preds))
    print("Testing Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))
    return classification_report(new_all_labels, new_all_preds, output_dict=True)['1']


def train(num_epochs=100):
    best_test_results = None
    train_loader = make_cdr_train_dataset(train_path='data/cdr/CDR_TrainingSet.PubTator.txt', dev_path='data/cdr/CDR_DevelopmentSet.PubTator.txt')
    test_loader = make_cdr_dataset('data/cdr/CDR_TestSet.PubTator.txt')

    tokenizer = get_tokenizer()
    electra_config = ElectraConfig()
    # net = ElectraModelClassification(electra_config)
    net = ElectraModelClassification.from_pretrained('google/electra-small-discriminator')
    # summary(net)
    # for param in net.
    for name, param in net.named_parameters():
        # if 'encoder' in name:
            # param.requires_grad = False
        print("name: {}, unfrozen:{}, size: {}".format(name, param.requires_grad, param.size()))
    # for layer in net:
    #     x = layer(x)
    #     print(x.size())
    if cuda:
        net.cuda()

    criteria = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    pad_id = tokenizer.pad_token_id

    def train_model(optimizer=None, scheduler=None, tokenizer=None, do_eval=False):
        net.train()
        epoch_loss = []
        all_labels = []
        all_preds = []
        for i, batch in tqdm(enumerate(train_loader)):
            x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
            # print('label = ', label)
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
            # print('learned before = {}'.format(net.projection.weight.data))
            loss = criteria(prediction, label)
            pred = prediction.argmax(dim=-1)
            all_labels.append(label.data.to('cpu'))
            all_preds.append(pred.to('cpu'))
            
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        # scheduler.step()
            
        average_loss = np.mean(epoch_loss)
        new_all_labels = []
        new_all_preds = []
        for i in range(len(all_labels)):
            new_all_labels += all_labels[i].tolist()
            new_all_preds += all_preds[i].tolist()

        from sklearn.metrics import classification_report
        print("average RE loss : ", average_loss)
        print("train_cls report: \n", classification_report(new_all_labels, new_all_preds))
        print("Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))
        if do_eval:
            evaluate(net, test_loader, tokenizer)

    # optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.01}])
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4, eps=1e-8)
    # optimizer = optim.SGD(net.parameters(), lr=0.05)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        do_eval = False
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            do_eval = True
        res_test = train_model(net, loss_fn=criteria, optimizer=optimizer, scheduler=None, tokenizer=tokenizer, do_eval=do_eval)
        if best_test_results == None or res_test['f1-score'] > best_test_results['f1-score']:
            best_test_results = res_test
        print('Best result on test data: Precision: {}, Recall: {}, F1: {}'.format(best_test_results['precision'], best_test_results['recall'], best_test_results['f1-score']))

def evaluate_sentence(net, test_loader, tokenizer):
    net.eval()
    pad_id = tokenizer.pad_token_id
    labels = []
    preds = []

    for i, batch in (enumerate(test_loader)):
        x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
        label = label.data
        labels.append(label)
        attention_mask = (x != pad_id).float()
        attention_mask = (1. - attention_mask) * -10000.
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if cuda:
            x = x.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        
        prediction = net(x, token_type_ids=token_type_ids, 
                                  attention_mask=attention_mask,
                                  used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                                  chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs)
        prediction.to('cpu')
        pred_label = prediction.argmax(dim=1).to('cpu')
        
        preds.append(pred_label)
    
    new_all_labels = []
    new_all_preds = []
    for i in range(len(labels)):
        new_all_labels += labels[i].tolist()
        new_all_preds += preds[i].tolist()
    
    # labels = torch.cat(labels, dim=-1)
    # preds = torch.cat(preds, dim=-1)
    from sklearn.metrics import classification_report
    print("Testing report: \n", classification_report(new_all_labels, new_all_preds))
    print("Testing Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))
    return classification_report(new_all_labels, new_all_preds, output_dict=True)['1']



def train_sentence(num_epochs=100, use_entity_token=False, train_with_full_sample = False):
    best_test_results = None
    best_epoch = None
    from models import src_path

    if train_with_full_sample:
        train_data_full, train_loader_full = make_train_joinlabel_dataset(src_path + '/data/cdr/CDR_TrainingSet.PubTator.txt',
                                                                dev_path='data/cdr/CDR_DevelopmentSet.PubTator.txt',
                                                                use_entity_token=True)
    _, train_loader = make_cdr_train_non_global_dataset(train_path='data/cdr/CDR_TrainingSet.PubTator.txt', dev_path='data/cdr/CDR_DevelopmentSet.PubTator.txt', use_entity_token=use_entity_token, extract_type='intra', batch_size=8)
    _, test_loader = make_cdr_non_global_dataset('data/cdr/CDR_TestSet.PubTator.txt', use_entity_token=use_entity_token, extract_type='intra', batch_size=8)
    # _, train_loader = make_cdr_non_global_dataset('data/cdr/CDR_TrainingSet.PubTator.txt', use_entity_token=use_entity_token, extract_type='inter')

    tokenizer = get_tokenizer()
    # electra_config = ElectraConfig.from_pretrained('google/electra-small-discriminator')
    # electra_config.vocab_size = electra_config.vocab_size + 2
    # net = ElectraModelEntitySentenceClassification(electra_config)ElectraModelEntityTokenClassification

    net_ner_pretrained = ElectraModelEntityTokenClassification.from_pretrained('models_saved/electra_token_model')
    net = ElectraModelEntitySentenceClassification.from_pretrained('google/electra-base-discriminator')
    net.resize_token_embeddings(len(tokenizer))
    net_ner_pretrained_encoder_params = net_ner_pretrained.encoder.named_parameters()
    net_encoder_params = net.encoder.named_parameters()

    dict_params_ner = dict(net_ner_pretrained_encoder_params)

    for name1, param1 in net_ner_pretrained_encoder_params:
        if name1 in net_encoder_params:
            dict_params_ner[name1].data.copy_(param1.data)

    net.encoder.load_state_dict(dict_params_ner)

    # summary(net)
    # for param in net.
    for name, param in net.named_parameters():
        print("name: {}, unfrozen:{}, size: {}".format(name, param.requires_grad, param.size()))

    if cuda:
        net.cuda()

    criteria = torch.nn.CrossEntropyLoss().cuda()

    pad_id = tokenizer.pad_token_id

    def train_model(model, loss_fn=None, optimizer=None, scheduler=None, tokenizer=None, do_eval=False):
        model.train()
        epoch_loss = []
        all_labels = []
        all_preds = []
        for i, batch in enumerate(train_loader):
            x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
            # print('label = ', label)
            # label = torch.squeeze(label, 1)
            # print('x: ', x)
            attention_mask = (x != pad_id).float()
            # attention_mask = (1. - attention_mask) * -10000.
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            if cuda:
                x = x.cuda()
                label = label.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            prediction = model(x, token_type_ids=token_type_ids, 
                                  attention_mask=attention_mask,
                                  used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                                  chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs)
            loss = loss_fn(prediction.view(-1, 2), label.view(-1))
            # if (i % 100 == 0):
            #     print('label: ', label)
            #     print('pred: ', prediction)
            #     print('loss: ', loss)
            
            pred = prediction.argmax(dim=-1)
            all_labels.append(label.data.to('cpu'))
            all_preds.append(pred.to('cpu'))
            
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            
        # scheduler.step()
            
        average_loss = np.mean(epoch_loss)
        new_all_labels = []
        new_all_preds = []
        for i in range(len(all_labels)):
            new_all_labels += all_labels[i].tolist()
            new_all_preds += all_preds[i].tolist()

        from sklearn.metrics import classification_report
        print("average RE loss : ", average_loss)
        print("train_cls report: \n", classification_report(new_all_labels, new_all_preds))
        print("Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))
        if do_eval:
            res = evaluate_sentence(model, test_loader, tokenizer)
            return res

    def train_full_sample(model, loss_fn=None, optimizer=None, scheduler=None, tokenizer=None):
        model.train()
        epoch_loss = []
        all_labels = []
        all_preds = []
        label_pad_id = -1
        for i, batch in enumerate(train_loader_full):
            x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch

            print("label shape: ", label.size())
            print("chemical_code_seqs ", chemical_code_seqs.size())
            print("disease_code_seqs ", disease_code_seqs.size())
            1/0
            # label = torch.squeeze(label, 1)
            # print('x: ', x)
            attention_mask = (x != pad_id).float()
            label_mask = (label != label_pad_id).float()
            # attention_mask = (1. - attention_mask) * -10000.
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            if cuda:
                x = x.cuda()
                label = label.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            prediction = model(x, token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs,
                               chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs,
                               is_full_sample= True)
            loss = loss_fn(prediction.view(-1, 2), label.view(-1))
            # if (i % 100 == 0):
            #     print('label: ', label)
            #     print('pred: ', prediction)
            #     print('loss: ', loss)

            pred = prediction.argmax(dim=-1)
            all_labels.append(label.data.to('cpu'))
            all_preds.append(pred.to('cpu'))

            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # scheduler.step()

        average_loss = np.mean(epoch_loss)
        new_all_labels = []
        new_all_preds = []
        for i in range(len(all_labels)):
            new_all_labels += all_labels[i].tolist()
            new_all_preds += all_preds[i].tolist()

        from sklearn.metrics import classification_report
        print("average RE loss : ", average_loss)
        print("train_cls report: \n", classification_report(new_all_labels, new_all_preds))
        print("Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))

    # optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.01}])
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4, eps=1e-8)
    
    # optimizer = optim.SGD(net.parameters(), lr=0.05)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        do_eval = False
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            do_eval = True
        train_full_sample(net, loss_fn=criteria, optimizer=optimizer, scheduler=None, tokenizer=tokenizer)
        res_test = train_model(net, loss_fn=criteria, optimizer=optimizer, scheduler=None, tokenizer=tokenizer, do_eval=do_eval)
        if best_test_results == None or res_test['f1-score'] > best_test_results['f1-score']:
            best_test_results = res_test
            best_epoch = epoch
        print('Best result on test data: Precision: {}, Recall: {}, F1: {}'.format(best_test_results['precision'], best_test_results['recall'], best_test_results['f1-score']))

def evaluate_ner(net, test_loader, tokenizer):
    net.eval()
    pad_id = tokenizer.pad_token_id
    labels = []
    preds = []

    for i, batch in (enumerate(test_loader)):
        x, entity_token_ids, label = batch
        label = label.data
        labels.append(label)
        attention_mask = (x != pad_id).float()
        attention_mask = (1. - attention_mask) * -10000.
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if cuda:
            x = x.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        
        prediction = net(x, token_type_ids=token_type_ids, 
                                # attention_masks=attention_mask,
                                  entity_token_ids=entity_token_ids)
        prediction.to('cpu')
        pred_label = prediction.argmax(dim=1).to('cpu')
        
        preds.append(pred_label)
    
    new_all_labels = []
    new_all_preds = []
    for i in range(len(labels)):
        new_all_labels += labels[i].tolist()
        new_all_preds += preds[i].tolist()
    
    # labels = torch.cat(labels, dim=-1)
    # preds = torch.cat(preds, dim=-1)
    from sklearn.metrics import classification_report
    print("Testing report: \n", classification_report(new_all_labels, new_all_preds))
    print("Testing Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))
    return classification_report(new_all_labels, new_all_preds, output_dict=True)['macro avg']

def train_ner(num_epochs=100, use_entity_token=False):
    best_test_results = None
    best_epoch = None
    _, train_loader = make_train_pretrain_ner_dataset(train_path='data/cdr/CDR_TrainingSet.PubTator.txt', dev_path='data/cdr/CDR_DevelopmentSet.PubTator.txt', use_entity_token=use_entity_token, batch_size=4)
    _, test_loader = make_pretrain_ner_dataset('data/cdr/CDR_TestSet.PubTator.txt', use_entity_token=use_entity_token, batch_size=4)
    # _, train_loader = make_cdr_non_global_dataset('data/cdr/CDR_TrainingSet.PubTator.txt', use_entity_token=use_entity_token, extract_type='inter')

    tokenizer = get_tokenizer()

    net = ElectraModelEntityTokenClassification.from_pretrained('google/electra-base-discriminator')
    net.resize_token_embeddings(len(tokenizer))
    # summary(net)
    # for param in net.
    for name, param in net.named_parameters():
        print("name: {}, unfrozen:{}, size: {}".format(name, param.requires_grad, param.size()))
    
    if cuda:
        net.cuda()

    criteria = torch.nn.CrossEntropyLoss().cuda()

    pad_id = tokenizer.pad_token_id

    def train_model(model, loss_fn=None, optimizer=None, scheduler=None, tokenizer=None, do_eval=False):
        model.train()
        epoch_loss = []
        all_labels = []
        all_preds = []
        for i, batch in enumerate(train_loader):
            x, entity_token_ids, label = batch
            # print('x: ', x)
            attention_mask = (x != pad_id).float()
            attention_mask = (1. - attention_mask) * -10000.
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            if cuda:
                x = x.cuda()
                label = label.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            prediction = model(x, token_type_ids=token_type_ids, 
                                # attention_masks=attention_mask,
                                  entity_token_ids=entity_token_ids)
            loss = loss_fn(prediction.view(-1, 2), label.view(-1))
            
            pred = prediction.argmax(dim=-1)
            all_labels.append(label.data.to('cpu'))
            all_preds.append(pred.to('cpu'))
            
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            
        average_loss = np.mean(epoch_loss)
        new_all_labels = []
        new_all_preds = []
        for i in range(len(all_labels)):
            new_all_labels += all_labels[i].tolist()
            new_all_preds += all_preds[i].tolist()

        from sklearn.metrics import classification_report
        print("average RE loss : ", average_loss)
        print("train_cls report: \n", classification_report(new_all_labels, new_all_preds))
        print("Confusion matrix report: \n", confusion_matrix(new_all_labels, new_all_preds))
        if do_eval:
            res = evaluate_ner(model, test_loader, tokenizer)
            return res

    # optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.01}])
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4, eps=1e-8)

    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        do_eval = False
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            do_eval = True
        res_test = train_model(net, loss_fn=criteria, optimizer=optimizer, scheduler=None, tokenizer=tokenizer, do_eval=do_eval)
        if best_test_results == None or res_test['f1-score'] > best_test_results['f1-score']:
            best_test_results = res_test
            best_epoch = epoch
            net.save_pretrained('models_saved/electra_token_model')
        print('Best result on test data: Precision: {}, Recall: {}, F1: {}'.format(best_test_results['precision'], best_test_results['recall'], best_test_results['f1-score']))
        print('Best epoch = ', best_epoch)

if __name__ == '__main__':
    train_sentence(num_epochs=1000, use_entity_token=True, train_with_full_sample= True)
