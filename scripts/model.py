from TorchCRF import CRF
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from transformers import MT5EncoderModel, T5Tokenizer
from cnn import *
import torch
import torch.nn as nn
import utils

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.softmax = nn.LogSoftmax(2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = x * self.tanh(self.softplus(x))

        x = self.linear2(x)
        return x, self.softmax(x)


class JointEntityRelation_cnn_four(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=len(utils.ENTITIES), rdim=len(utils.RELATIONS),
                 device='cuda', max_length=256):
        super(JointEntityRelation_cnn_four, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        self.pretrained_model_path = pretrained_model_path

        self.cnn = IDCNN(768*4,768)



        # BETO
        if 'mt5' in pretrained_model_path:
            # google/mt5-base
            self.beto = MT5EncoderModel.from_pretrained(pretrained_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
            self.beto = AutoModel.from_pretrained(pretrained_model_path)

        self.linear_layer = nn.Linear(2 * (hdim + edim), hdim)
        self.tanh = nn.Tanh()
        self.beto.config.output_hidden_states = True
        # linear projections
        self.entity_classifier = Classifier(768 * 2, edim)

        # self.related_classifier = Classifier(hdim, rdim)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        all_hidden_states = self.beto(**tokens).hidden_states
        embeddings = torch.cat(

            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1

        )
        embeddings,_ = self.cnn(embeddings)
        # part 2
        logits, entity = self.entity_classifier(embeddings)

        return entity #, related


class JointEntityRelation_rnn_four(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=len(utils.ENTITIES), rdim=len(utils.RELATIONS),
                 device='cuda', max_length=256):
        super(JointEntityRelation_rnn_four, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        self.pretrained_model_path = pretrained_model_path


        self.rnn = nn.GRU(768*4, 768, 2,bidirectional=True)  # utilize the GRU model in torch.nn

        # BETO
        if 'mt5' in pretrained_model_path:
            # google/mt5-base
            self.beto = MT5EncoderModel.from_pretrained(pretrained_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
            self.beto = AutoModel.from_pretrained(pretrained_model_path)

        self.linear_layer = nn.Linear(2 * (hdim + edim), hdim)
        self.tanh = nn.Tanh()
        self.beto.config.output_hidden_states = True
        # linear projections
        self.entity_classifier = Classifier(768 * 2, edim)

        # self.related_classifier = Classifier(hdim, rdim)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        all_hidden_states = self.beto(**tokens).hidden_states
        embeddings = torch.cat(

            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1

        )
        embeddings,_ = self.rnn(embeddings)
        # part 2
        logits, entity = self.entity_classifier(embeddings)

        return entity #, related



class JointEntityRelation_crf(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=len(utils.ENTITIES), rdim=len(utils.RELATIONS),
                 device='cuda', max_length=256):
        super(JointEntityRelation_crf, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        self.pretrained_model_path = pretrained_model_path

        # BETO
        if 'mt5' in pretrained_model_path:
            # google/mt5-base
            self.beto = MT5EncoderModel.from_pretrained(pretrained_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
            self.beto = AutoModel.from_pretrained(pretrained_model_path)
        self.beto.config.output_hidden_states = True
        self.linear_layer = nn.Linear(2 * (hdim + edim), hdim)
        self.tanh = nn.Tanh()

        self.rnn = nn.GRU(768 * 4, 768, 2, bidirectional=True)  # utilize the GRU model in torch.nn
        # linear projections
        self.entity_classifier = Classifier(hdim * 2, edim)
        self.crf = CRF(num_labels=5)


        # self.related_classifier = Classifier(hdim, rdim)

    def forward(self, texts,labels=None):
        # part 1
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        all_hidden_states = self.beto(**tokens).hidden_states
        embeddings = torch.cat(

            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1

        )
        embeddings, _ = self.rnn(embeddings)
        # part 2
        logits, entity = self.entity_classifier(embeddings)
        if labels != None:
            loss = self.crf(logits,labels,labels.gt(-1)) * (-1)
            return entity,loss  #, related
        return entity  #, related




class JointEntityRelation_cnn(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=len(utils.ENTITIES), rdim=len(utils.RELATIONS),
                 device='cuda', max_length=256):
        super(JointEntityRelation_cnn, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        self.pretrained_model_path = pretrained_model_path

        self.cnn = IDCNN(768,100)

        # BETO
        if 'mt5' in pretrained_model_path:
            # google/mt5-base
            self.beto = MT5EncoderModel.from_pretrained(pretrained_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
            self.beto = AutoModel.from_pretrained(pretrained_model_path)

        self.linear_layer = nn.Linear(2 * (hdim + edim), hdim)
        self.tanh = nn.Tanh()

        # linear projections
        self.entity_classifier = Classifier(100, edim)

        # self.related_classifier = Classifier(hdim, rdim)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        embeddings = self.beto(**tokens)['last_hidden_state']  # torch.Size([1, 59, 768])
        embeddings = self.cnn(embeddings)
        # part 2
        logits, entity = self.entity_classifier(embeddings)

        return entity #, related


class JointEntityRelation(nn.Module):
    def __init__(self, pretrained_model_path='dccuchile/bert-base-spanish-wwm-cased',
                 hdim=768, edim=len(utils.ENTITIES), rdim=len(utils.RELATIONS),
                 device='cuda', max_length=256):
        super(JointEntityRelation, self).__init__()
        self.hdim = hdim
        self.edim = edim
        self.rdim = rdim
        self.device = device
        self.max_length = max_length
        self.pretrained_model_path = pretrained_model_path
        
        # BETO
        if 'mt5' in pretrained_model_path:
            # google/mt5-base
            self.beto = MT5EncoderModel.from_pretrained(pretrained_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
            self.beto = AutoModel.from_pretrained(pretrained_model_path)

        self.linear_layer = nn.Linear(2 * (hdim + edim), hdim)
        self.tanh = nn.Tanh()

        # linear projections
        self.entity_classifier = Classifier(hdim, edim)

        # self.related_classifier = Classifier(hdim, rdim)

    def forward(self, texts):
        # part 1
        tokens = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        embeddings = self.beto(**tokens)['last_hidden_state']


        # part 2  torch.Size([1, 59, 768])
        logits, entity = self.entity_classifier(embeddings)

        return entity #, related

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, entity_loss, relation_loss):

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*entity_loss + self.log_vars[0]

        # precision1 = torch.exp(-self.log_vars[1])
        # loss1 = precision1*relation_loss + self.log_vars[1]
        
        return loss0  #+loss1