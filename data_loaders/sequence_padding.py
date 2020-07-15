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
        code_index = torch.stack(list(map(lambda x: x[1], batch)))
        label = torch.stack(list(map(lambda x: x[2], batch)))
        return seqs, code_index, label