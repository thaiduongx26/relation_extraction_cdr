from torch import nn
import torch
from typing import *

from transformers import AdamW, XLNetTokenizer

from models.electra_model import ElectraModelClassificationALPS
from transformers import ElectraConfig
from data_loaders.alps_dataset import make_cdr_non_global_dataset
from tqdm import tqdm
from utils.trainer_utils import get_tokenizer
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
# from optim import optim4GPU
# from torchsummary import summary
from sklearn.metrics import confusion_matrix
import numpy as np

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
cuda = torch.cuda.is_available()
torch.cuda.set_device(0)
print('cuda is_available: ', cuda)


def evaluate(net, test_loader, tokenizer):
    net.eval()
    pad_id = tokenizer.pad_token_id
    labels = []
    preds = []

    for i, batch in tqdm(enumerate(test_loader)):
        x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, other_code_seqs, label = batch
        # label = torch.squeeze(label, 1).to('cpu')
        label = label.data
        labels.append(label)
        attention_mask = (x != pad_id).float()
        attention_mask = (1. - attention_mask) * -10000.
        token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
        if cuda:
            x = x.cuda()
        
        prediction = net(x, 
                        # token_type_ids=token_type_ids, 
                        # attention_masks=attention_mask,
                            used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                            chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs, other_code_list=other_code_seqs)
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
    best_epoch = None
    _, train_loader = make_cdr_non_global_dataset('data/alps/alps_train.txt')
    _, test_loader = make_cdr_non_global_dataset('data/alps/alps_test.txt')

    tokenizer = XLNetTokenizer.from_pretrained('model_sentence_piece/wiki-ja.model')
    tokenizer.add_tokens('<e>')
    tokenizer.add_tokens('</e>')
    # electra_config = ElectraConfig()
    # net = ElectraModelClassification(electra_config)
    net = ElectraModelClassificationALPS.from_pretrained('models_saved/Electra_converted_pytorch')
    net.resize_token_embeddings(len(tokenizer))

    for name, param in net.named_parameters():
        # if 'encoder' in name:
            # param.requires_grad = False
        print("name: {}, unfrozen:{}, size: {}".format(name, param.requires_grad, param.size()))
    if cuda:
        net.cuda()

    

    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def train_model(model, loss_fn, optimizer, scheduler, tokenizer, do_eval):
        net.train()
        epoch_loss = []
        all_labels = []
        all_preds = []
        for i, batch in tqdm(enumerate(train_loader)):
            x, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, other_code_seqs, label = batch
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

            prediction = model(x, 
                                # token_type_ids=token_type_ids, 
                                # attention_masks=attention_mask,
                                  used_entity_token=False, masked_entities_list=masked_entities_encoded_seqs, 
                                  chemical_code_list=chemical_code_seqs, disease_code_list=disease_code_seqs, other_code_list=other_code_seqs)
            # print('learned before = {}'.format(net.projection.weight.data))
            loss = criteria(prediction, label)
            pred = prediction.argmax(dim=-1)
            all_labels.append(label.data)
            all_preds.append(pred)
            
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
            return evaluate(model, test_loader, tokenizer)

    optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": 0.000000001}])
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        criteria = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'))
        do_eval = False
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            do_eval = True
        res_test = train_model(net, loss_fn=criteria, optimizer=optimizer, scheduler=None, tokenizer=tokenizer, do_eval=do_eval)
        if best_test_results == None or res_test['f1-score'] > best_test_results['f1-score']:
            best_test_results = res_test
            best_epoch = epoch
        print('Best result on test data: Precision: {}, Recall: {}, F1: {}'.format(best_test_results['precision'], best_test_results['recall'], best_test_results['f1-score']))
        print('Best epoch = ', best_epoch)


if __name__ == "__main__":
    train(num_epochs=1000)    