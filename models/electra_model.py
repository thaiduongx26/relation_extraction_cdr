import logging
import os
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import ElectraModel, ElectraConfig
from transformers.modeling_electra import ElectraEmbeddings, ElectraPreTrainedModel
from transformers.modeling_bert import BertEncoder
from transformers.activations import get_activation

ELECTRA_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.ElectraConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.ElectraTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""
def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

class ElectraModelClassification(ElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.dense = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, 2)
        # 1/0
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
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
        disease_code_list=None,
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
            # output_attentions=output_attentions,
        )

        batch_size = chemical_code_list.shape[0]
        token_embedding_output = hidden_states[0]
        
        def get_entity_embedding(token_embedding, masked_entities, code):
            count = 0
            embedding = torch.zeros(token_embedding.shape[1]).cuda()
            check = True
            for i, mask in enumerate(masked_entities):
                if mask == code:
                    if check:
                        count += 1
                        check = False
                    embedding += token_embedding[i]
                else:
                    check = True

            embedding = embedding / count
            return embedding

        # def get_entity_token_embedding(token_embedding, masked_entities, code):
        #     count = 0
        #     embedding = torch.zeros(token_embedding.shape[1]).cuda()
        #     check = True
        #     for i, mask in enumerate(masked_entities):
        #         if mask == code and check:
        #             count += 1
        #             embedding += token_embedding[i]
        #         else:
        #             if 
        #     # embedding = embedding / count
        #     return embedding

        batch_embedding = []

        if not used_entity_token:
            for i in range(batch_size):
                masked_entities = masked_entities_list[i]
                chemical_code = chemical_code_list[i]
                disease_code = disease_code_list[i]
                token_embedding = token_embedding_output[i]
                chemical_embedding = get_entity_embedding(token_embedding, masked_entities, chemical_code)
                disease_embedding = get_entity_embedding(token_embedding, masked_entities, disease_code)
                # print('chemical_embedding shape: ', chemical_embedding.shape)
                # print('disease_embedding shape: ', disease_embedding.shape)
                entity_embedding = torch.cat((chemical_embedding, disease_embedding), 0)
                # print(entity_embedding.shape)
                batch_embedding.append(entity_embedding.tolist())
        # else:
        #     for i in range(batch_size):
        #         masked_entities = masked_entities_list[i]
        #         chemical_code = chemical_code_list[i]
        #         disease_code = disease_code_list[i]
        #         token_embedding = token_embedding_output[i]
        #         chemical_embedding = get_entity_embedding(token_embedding, masked_entities, chemical_code)
        #         disease_embedding = get_entity_embedding(token_embedding, masked_entities, disease_code)
        #         # print('chemical_embedding shape: ', chemical_embedding.shape)
        #         # print('disease_embedding shape: ', disease_embedding.shape)
        #         entity_embedding = torch.cat((chemical_embedding, disease_embedding), 0)
        #         # print(entity_embedding.shape)
        #         batch_embedding.append(entity_embedding.tolist())
        batch_embedding = torch.tensor(batch_embedding).cuda()
        # print('batch_embedding shape: ', batch_embedding.shape)
        sequence_output_cls = batch_embedding
        x = self.dropout(sequence_output_cls)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraModelClassificationALPS(ElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.dense = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(512, 2)
        # 1/0
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        used_entity_token=True,
        masked_entities_list=None,
        chemical_code_list=None,
        disease_code_list=None,
        other_code_list=None
    ):
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
            # output_attentions=output_attentions,
        )

        batch_size = chemical_code_list.shape[0]

        token_embedding_output = hidden_states[0]
        
        def get_entity_embedding(token_embedding, masked_entities, code):
            count = 0
            embedding = None
            check = True
            print(token_embedding.shape)
            for i, mask in enumerate(masked_entities):
                if mask == code:
                    if check:
                        count += 1
                        check = False
                    if embedding == None:
                        embedding = token_embedding[i]
                    else:
                        embedding += token_embedding[i]
                    # print('embedding shape: ', embedding.shape)
                else:
                    check = True

            embedding = embedding / count
            return embedding


        batch_embedding = []

        if not used_entity_token:
            for i in range(batch_size):
                masked_entities = masked_entities_list[i]
                chemical_code = chemical_code_list[i]
                disease_code = disease_code_list[i]
                other_code = other_code_list[i]
                token_embedding = token_embedding_output[i]
                if chemical_code == -1:
                    other_embedding = get_entity_embedding(token_embedding, masked_entities, other_code)
                    disease_embedding = get_entity_embedding(token_embedding, masked_entities, disease_code)
                    entity_embedding = torch.cat((disease_embedding, other_embedding), 0)
                elif disease_code == -1:
                    chemical_embedding = get_entity_embedding(token_embedding, masked_entities, chemical_code)
                    other_embedding = get_entity_embedding(token_embedding, masked_entities, other_code)
                    entity_embedding = torch.cat((chemical_embedding, other_embedding), 0)
                elif other_code == -1:
                    chemical_embedding = get_entity_embedding(token_embedding, masked_entities, chemical_code)
                    disease_embedding = get_entity_embedding(token_embedding, masked_entities, disease_code)
                    entity_embedding = torch.cat((chemical_embedding, disease_embedding), 0)

                batch_embedding.append(entity_embedding.tolist())
        batch_embedding = torch.tensor(batch_embedding).cuda()
        sequence_output_cls = batch_embedding
        x = self.dropout(sequence_output_cls)
        x = self.dense(x)
        x = get_activation("gelu")(x) 
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraModelSentenceClassification(ElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.dense = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(256, 2)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
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
            # output_attentions=output_attentions,
        )
        sequence_output = hidden_states[0]
        sequence_output_cls = sequence_output[:, 0, :]
        x = self.dropout(sequence_output_cls)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraModelEntitySentenceClassification(ElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.dense = nn.Linear(config.embedding_size * 2, config.embedding_size * 2)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.embedding_size * 2, 2)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
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
        disease_code_list=None,
        is_full_sample= False,
        label_length = 0,
    ):
    

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
            # output_attentions=output_attentions,
        )
        sequence_output = hidden_states[0]

        batch_size = chemical_code_list.shape[0]
        def get_entity_embedding(token_embedding, masked_entities, code):
            embedding = None
            for i, mask in enumerate(masked_entities):
                if mask == code:
                    embedding = token_embedding[i]
                    break
            return embedding

        def get_all_entity_embedding(token_embedding, masked_entities, code):
            embedding_size = list(token_embedding.size())[-1]
            embedding = []
            current_idx= 0
            for i, mask in enumerate(masked_entities):
                if mask == code:
                    if i!= current_idx-1: #get first embedding
                        embedding.append(token_embedding[i])
                    current_idx = i
            if len(embedding) == 0:
                embedding= [torch.zeros(embedding_size)]
                if torch.cuda.is_available():
                    embedding =torch.stack( embedding).cuda()
            else:
                embedding = torch.stack(embedding)
            return embedding

        def generate_code_pairs_list(chemical_code_list_encoded, disease_code_list_encoded, label_len):
            chemical_codes = []
            disease_codes = []
            chemical_code_size = list(chemical_code_list_encoded.size())
            disease_code_size = list(disease_code_list_encoded.size())
            tensor_size = chemical_code_size[0]*disease_code_size[0]
            for i in range(chemical_code_size[0]):
                if chemical_code_list_encoded[i] == -1:
                    break
                for j in range(disease_code_size[0]):
                    if disease_code_list_encoded[j] == -1:
                        break
                    chemical_codes.append(chemical_code_list_encoded[i])
                    disease_codes.append(disease_code_list_encoded[j])
            for i in range(len(chemical_codes), label_len):
                chemical_codes.append(-1)
                disease_codes.append(-1)
            return chemical_codes, disease_codes



        # def get_entity_embedding(token_embedding, masked_entities, code):
        #     count = 0
        #     embedding = torch.zeros(token_embedding.shape[1]).cuda()
        #     check = True
        #     for i, mask in enumerate(masked_entities):
        #         if mask == code:
        #             if check:
        #                 count += 1
        #                 check = False
        #             embedding += token_embedding[i]
        #         else:
        #             check = True

        #     embedding = embedding / count
        #     return embedding

        # def get_entity_embedding_use_e_token(token_embedding, masked_entities, code):
        #     embedding = None
        #     for i, mask in enumerate(masked_entities):
        #         if mask == code:
        #             embedding = token_embedding[i]
                    
        #     return embedding

        batch_embedding = []

        if not is_full_sample:
            for i in range(batch_size):
                masked_entities = masked_entities_list[i]
                chemical_code = chemical_code_list[i]
                disease_code = disease_code_list[i]
                token_embedding = sequence_output[i]
                chemical_embedding = get_entity_embedding(token_embedding, masked_entities, chemical_code)
                disease_embedding = get_entity_embedding(token_embedding, masked_entities, disease_code)
                # print('chemical_embedding shape: ', chemical_embedding.shape)
                # print('disease_embedding shape: ', disease_embedding.shape)
                entity_embedding = torch.cat((chemical_embedding, disease_embedding), 0)
                # print(entity_embedding.shape)
                batch_embedding.append(entity_embedding.tolist())
            batch_embedding = torch.tensor(batch_embedding).cuda()
            sequence_output_cls = batch_embedding
            x = self.dropout(sequence_output_cls)
            x = self.dense(x)
            x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
            x = self.dropout(x)
            x = self.out_proj(x)
            return x
        else:
            batch_embedding = []
            for i in range(batch_size):
                masked_entities = masked_entities_list[i]
                chemical_codes, disease_codes = generate_code_pairs_list(chemical_code_list[i], disease_code_list[i], label_length)
                token_embedding = sequence_output[i]
                current_output = []
                for j in range(len(chemical_codes)):
                    chemical_embeddings = get_all_entity_embedding(token_embedding, masked_entities, chemical_codes[j])
                    disease_embeddings = get_all_entity_embedding(token_embedding, masked_entities, disease_codes[j])
                    chemical_embedding = torch.mean(chemical_embeddings, dim=0)
                    disease_embedding = torch.mean(disease_embeddings, dim=0)
                    r_rep = torch.cat([chemical_embedding, disease_embedding], 0)
                    current_output.append(r_rep)
                current_output_stacked = torch.stack(current_output).unsqueeze(0)
                batch_embedding.append(current_output_stacked)
            batch_embedding = torch.cat(batch_embedding, 0)
            sequence_output_cls = batch_embedding
            x = self.dropout(sequence_output_cls)
            x = self.dense(x)
            x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
            x = self.dropout(x)
            x = self.out_proj(x)
            return x


class ElectraModelEntityTokenClassification(ElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.dense = nn.Linear(config.embedding_size, config.embedding_size * 2)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.embedding_size * 2, 2)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        entity_token_ids=None
    ):
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
            head_mask=head_mask
            # output_attentions=output_attentions,
        )
        sequence_output = hidden_states[0]
        batch_size = sequence_output.shape[0]

        batch_embedding = []

        for i in range(batch_size):
            entity_embedding = sequence_output[i][entity_token_ids[i]]
            batch_embedding.append(entity_embedding.tolist())
        
        batch_embedding = torch.tensor(batch_embedding).cuda()
        sequence_output_cls = batch_embedding
        x = self.dropout(sequence_output_cls)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x