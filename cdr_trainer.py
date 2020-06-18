from torch import nn
import torch
from typing import *

from models.electra_model import ElectraModelClassification
from transformers import ElectraConfig
from data_loaders.cdr_dataset import make_cdr_dataset, make_cdr_train_dataset
from tqdm import tqdm
from utils.trainer_utils import get_tokenizer
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from optim import optim4GPU
from torchsummary import summary


import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
cuda = torch.cuda.is_available()
torch.cuda.set_device(2)
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
    print("report: ", classification_report(new_all_labels, new_all_preds))


def train(num_epochs=100):

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
        epoch_loss = 0
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
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss
            
            # print('learned after = {}'.format(net.projection.weight.data))
            # print("learned A = {}".format(list(net.parameters())[14].data[0]))
            # print("learned b = {}".format(list(net.parameters())[1].data[0]))

        # scheduler.step()
            
        average_loss = epoch_loss / i
        new_all_labels = []
        new_all_preds = []
        for i in range(len(all_labels)):
            new_all_labels += all_labels[i].tolist()
            new_all_preds += all_preds[i].tolist()

        from sklearn.metrics import classification_report
        print("average RE loss : ", average_loss)
        print("train_cls report: ", classification_report(new_all_labels, new_all_preds))
        if do_eval:
            evaluate(net, test_loader, tokenizer)

    optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.01}])
    
    # optimizer = optim.SGD(net.parameters(), lr=0.05)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        # print('Epoch:', epoch,'LR:', scheduler.get_lr())
        do_eval = False
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            do_eval = True
        train_model(optimizer=optimizer, scheduler=None, tokenizer=tokenizer, do_eval=do_eval)

# def pretrain_model(num_epochs=100):

#     train_loader = make_cdr_dataset('data/cdr/CDR_TrainingSet.PubTator.txt')
#     test_loader = make_cdr_dataset('data/cdr/CDR_TestSet.PubTator.txt')

#     tokenizer = get_tokenizer()
#     electra_config = ElectraConfig()
#     electra_config.vocab_size = tokenizer.vocab_size
#     net = ElectraModelClassification(electra_config)
#     # for name, param in net.named_parameters():
#     #     print("name: {}, unfrozen:{}".format(name, param.requires_grad))
#     if cuda:
#         net.cuda()

#     criteria = torch.nn.CrossEntropyLoss(reduction='sum')

#     pad_id = tokenizer.pad_token_id

#     def train_model(optimizer=None, scheduler=None, tokenizer=None, do_eval=False):
#         net.train()
#         epoch_loss = 0
#         all_labels = []
#         all_preds = []
#         for i, batch in tqdm(enumerate(train_loader)):
#             x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label = batch
#             # print('label = ', label)
#             # label = torch.squeeze(label, 1)
#             attention_mask = (x != pad_id).float()
#             attention_mask = (1. - attention_mask) * -10000.
#             token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
#             if cuda:
#                 x = x.cuda()
#                 label = label.cuda()
#                 attention_mask = attention_mask.cuda()
#                 token_type_ids = token_type_ids.cuda()


#             prediction = net(x, token_type_ids=token_type_ids, 
#                                 # attention_masks=attention_mask,
#                                   used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
#                                   chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs)
            
#             loss = criteria(prediction, label)
#             pred = prediction.argmax(dim=-1)
#             all_labels.append(label.data.to('cpu'))
#             all_preds.append(pred.to('cpu'))
#             # print('all_labels: ', all_labels)
#             # print('all_preds: ', all_preds)
#             epoch_loss += loss
#             loss.backward()
#             optimizer.zero_grad()
#             optimizer.step()
            
            
#         # scheduler.step()
            
#         average_loss = epoch_loss / i
#         new_all_labels = []
#         new_all_preds = []
#         for i in range(len(all_labels)):
#             new_all_labels += all_labels[i].tolist()
#             new_all_preds += all_preds[i].tolist()

#         from sklearn.metrics import classification_report
#         print("average RE loss : ", average_loss)
#         print("train_cls report: ", classification_report(new_all_labels, new_all_preds))
#         if do_eval:
#             evaluate(net, test_loader, tokenizer)

#     optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.01}])
#     # optimizer = optim4GPU(net)
#     # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)
#     for epoch in range(num_epochs):
#         print('Epoch:', epoch)
#         # print('Epoch:', epoch,'LR:', scheduler.get_lr())
#         do_eval = False
#         if epoch % 3 == 0 or epoch == num_epochs - 1:
#             do_eval = True
#         train_model(optimizer=optimizer, scheduler=None, tokenizer=tokenizer, do_eval=do_eval)

if __name__ == '__main__':
    train()