import torch
from torch.nn.utils.rnn import pad_sequence

class PadSequenceCDRDataset():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, token_pad_value, masked_entities_pad_value=-1):
        self.token_pad_value = token_pad_value
        self.masked_entities_pad_value = masked_entities_pad_value
        

    def __call__(self, batch):
        seqs = [x[0] for x in batch]
        masked_entities_encoded_seqs = [x[1] for x in batch]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.token_pad_value)
        masked_entities_encoded_seqs = pad_sequence(masked_entities_encoded_seqs, batch_first=True, padding_value=self.masked_entities_pad_value)
        chemical_code_seqs = torch.stack(list(map(lambda x: x[2], batch)))
        disease_code_seqs = torch.stack(list(map(lambda x: x[3], batch)))
        label = torch.stack(list(map(lambda x: x[4], batch)))
        return seqs, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label

class PadSequenceCDRSentenceDataset():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, token_pad_value, masked_entities_pad_value=-1):
        self.token_pad_value = token_pad_value
        self.masked_entities_pad_value = masked_entities_pad_value
        

    def __call__(self, batch):
        seqs = [x[0] for x in batch]
        masked_entities_encoded_seqs = [x[1] for x in batch]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.token_pad_value)
        masked_entities_encoded_seqs = pad_sequence(masked_entities_encoded_seqs, batch_first=True, padding_value=self.masked_entities_pad_value)
        chemical_code_seqs = torch.stack(list(map(lambda x: x[2], batch)))
        disease_code_seqs = torch.stack(list(map(lambda x: x[3], batch)))
        label = torch.stack(list(map(lambda x: x[4], batch)))
        return seqs, masked_entities_encoded_seqs, chemical_code_seqs, disease_code_seqs, label

class PadSequenceNERCDRDataset():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, token_pad_value, masked_entities_pad_value=-1):
        self.token_pad_value = token_pad_value
        self.masked_entities_pad_value = masked_entities_pad_value
        

    def __call__(self, batch):
        seqs = [x[0] for x in batch]  
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.token_pad_value)
        code_index = torch.stack(list(map(lambda x: x[1], batch)))
        label = torch.stack(list(map(lambda x: x[2], batch)))
        return seqs, code_index, label


class PadSequenceCDRFulltextJoinLabelDataset():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(self, token_pad_value, masked_entities_pad_value=-1, masked_chemical_code_list_encoded=-1, masked_disease_code_list_encoded=-1, masked_label_list=-1):
        self.token_pad_value = token_pad_value
        self.masked_entities_pad_value = masked_entities_pad_value
        self.masked_chemical_code_list_encoded = masked_chemical_code_list_encoded
        self.masked_disease_code_list_encoded = masked_disease_code_list_encoded
        self.masked_label_list = masked_label_list
        

    def __call__(self, batch):
        seqs = [x[0] for x in batch]
        masked_entities_encoded_seqs = [x[1] for x in batch]
        chemical_code_list_encoded_seqs = [x[2] for x in batch]
        disease_code_list_encoded_seqs = [x[3] for x in batch]
        label_list_seqs = [x[4] for x in batch]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.token_pad_value)
        masked_entities_encoded_seqs = pad_sequence(masked_entities_encoded_seqs, batch_first=True, padding_value=self.masked_entities_pad_value)
        chemical_code_list_encoded_seqs = pad_sequence(chemical_code_list_encoded_seqs, batch_first=True, padding_value=self.masked_chemical_code_list_encoded)
        disease_code_list_encoded_seqs = pad_sequence(disease_code_list_encoded_seqs, batch_first=True, padding_value=self.masked_disease_code_list_encoded)
        label_list_seqs = pad_sequence(label_list_seqs, batch_first=True, padding_value=self.masked_label_list)
        return seqs, masked_entities_encoded_seqs, chemical_code_list_encoded_seqs, disease_code_list_encoded_seqs, label_list_seqs