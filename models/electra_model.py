import logging
import os
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import ElectraModel, ElectraConfig


class ElectraModelClassification(ElectraModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.projection = nn.Linear(1024, 2)
        # self.embeddings = ElectraEmbeddings(config)

        # if config.embedding_size != config.hidden_size:
        #     self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        # self.encoder = BertEncoder(config)
        # self.config = config
        # self.init_weights()

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value

    # def _prune_heads(self, heads_to_prune):
    #     """ Prunes heads of the model.
    #         heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
    #         See base class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        used_entity_token=False,
        masked_entities_list=None,
        chemical_code_list=None,
        disease_code_list=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import ElectraModel, ElectraTokenizer
        import torch

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = ElectraModel.from_pretrained('google/electra-small-discriminator')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            # output_attentions=output_attentions
        )

        batch_size = chemical_code_list.shape[0]
        token_embedding_output = hidden_states[0]
        
        def get_entity_embedding(token_embedding, masked_entities, code):
            count = 0
            embedding = torch.zeros(token_embedding.shape[1])
            for i, mask in enumerate(masked_entities):
                if mask == code:
                    count += 1
                    embedding += token_embedding[i]
            embedding = embedding / count
            return embedding

        batch_embedding = []

        if not used_entity_token:
            for i in range(batch_size):
                masked_entities = masked_entities_list[i]
                chemical_code = chemical_code_list[i]
                disease_code = disease_code_list[i]
                token_embedding = token_embedding_output[i]
                chemical_embedding = get_entity_embedding(token_embedding, masked_entities, chemical_code)
                disease_embedding = get_entity_embedding(token_embedding, masked_entities, disease_code)
                entity_embedding = torch.cat((chemical_embedding, disease_embedding), 1)
                print(entity_embedding.shape)
                batch_embedding.append(entity_embedding)
        output = self.projection(torch.tensor(batch_embedding))
        return output
