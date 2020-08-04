import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertOnlyMLMHead



class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BertModelPoolerLogit(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)
        self._init_weights_bert(self.cls_layer)

    def forward(self, seq, token_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=token_ids)

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(pooler_output)

        return logits

class BertModelSupplementaryPoolerComputers(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)
        # computers_new
        self.load_state_dict(torch.load('saved/models/BERT-computers-pretrain/0708_185939/model_with_head.bin'))


    def forward(self, seq, token_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=token_ids)

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(pooler_output)

        return logits

class BertModelSupplementaryPooler4Cat(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, num_classes)
        # computers_new
        self.load_state_dict(torch.load('saved/models/BERT-4cat-pretrain/0710_094316/model_with_head.bin'))


    def forward(self, seq, token_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=token_ids)

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(pooler_output)

        return logits

class BertModelSupplementaryPoolerMLM(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls = nn.Linear(768, num_classes)
        # computers_new + mlm
        self.load_state_dict(torch.load('saved/models/BERT-computers-pretrain-mlm/0627_114018/model_with_head.bin'))


    def forward(self, seq, token_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # Feeding the input to BERT model to obtain contextualized representations
        _, pooler_output = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=token_ids)

        # Feeding cls_rep to the classifier layer
        logits = self.cls(pooler_output)

        return logits

class BertModelSupplementaryMLM(BaseModel):
    def __init__(self, num_classes=1, freeze_bert=False):
        super().__init__()
        self.config = BertConfig()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.mlm = BertOnlyMLMHead(self.config)
        self.cls = nn.Linear(768, num_classes)
        self._init_weights_bert(self.mlm)
        self._init_weights_bert(self.cls)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

    def forward(self, seq, token_ids, attn_masks):

        outputs = self.bert_layer(
            seq,
            attention_mask=attn_masks,
            token_type_ids=token_ids
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.mlm(sequence_output)
        seq_relationship_score = self.cls(pooled_output)

        outputs = seq_relationship_score, prediction_scores

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)