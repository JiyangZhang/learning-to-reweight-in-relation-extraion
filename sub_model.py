
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import torchvision
from torch.autograd import Variable
#import itertools
import torch.nn.init as weight_init

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class getEmbeddings(nn.Module):
    def __init__(self, word_size, word_length, feature_size, feature_length, Wv, pf1, pf2):
        super(getEmbeddings, self).__init__()
        self.x_embedding = nn.Embedding(word_length, word_size, padding_idx=0)
        self.ldist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)
        self.rdist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)
        self.x_embedding.weight.data.copy_(torch.from_numpy(Wv))
        self.ldist_embedding.weight.data.copy_(torch.from_numpy(pf1))
        self.rdist_embedding.weight.data.copy_(torch.from_numpy(pf2))

    def forward(self, x, ldist, rdist):
        x_embed = self.x_embedding(x)
        ldist_embed = self.ldist_embedding(ldist)
        rdist_embed = self.rdist_embedding(rdist)
        concat = torch.cat([x_embed, ldist_embed, rdist_embed], x_embed.dim() - 1)
        return concat.unsqueeze(1)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
       for name, param in self.named_params(self):
            yield param
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self, curr_module=None, memo=None, prefix=''):       
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    #print(name)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    #print(name)
                    yield prefix + ('.' if prefix else '') + name, p
                    
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                #print(name)
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                if "embeddings" in name_t:
                    continue
                #print(name_t, type(tmp))
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            #print('name has .')
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    #print('yesss')
                    self.set_param(mod, rest, param)
                    break
        else:
            #print(name, type(param))
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaConv2d, self).__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        ignore.bias.data.copy_(weight_init.constant(ignore.bias.data,0.))
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class CNNwithPool(MetaModule):
    def __init__(self, cnn_layers, kernel_size):

        super(CNNwithPool,self).__init__()
        self.cnn = MetaConv2d(1, cnn_layers, kernel_size)
    
    def forward(self, x, entity_pos):
        cnn = self.cnn(x)
        concat_list = []
        for index, entity in enumerate(entity_pos):
            elem = cnn.narrow(0,index,1)
            if entity[0] > 78:
                entity[0] = 78
            if entity[1] > 78:
                entity[1] = 78
            if entity[0] == entity[1]:
                entity[1] += 1
            pool1 = F.max_pool2d(elem.narrow(2,0,entity[0]),(entity[0],1))
            pool2 = F.max_pool2d(elem.narrow(2,entity[0],entity[1]-entity[0]),(entity[1]-entity[0],1))
            pool3 = F.max_pool2d(elem.narrow(2,entity[1],cnn.size(2)-entity[1]),(cnn.size(2)-entity[1],1))
            concat_pool = torch.cat((pool1, pool2, pool3), cnn.dim()-1)
            concat_list.append(concat_pool)
        concat_all = torch.cat(concat_list,0)
        return concat_all

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaLinear, self).__init__()
        ignore = nn.Linear(*args, **kwargs)
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class PCNN(MetaModule):
    def __init__(self, word_length, feature_length, cnn_layers, Wv, pf1, pf2, kernel_size, word_size=50, feature_size=5, dropout=0.5, num_classes=53, num_words=82):
        
        super(PCNN, self).__init__()
        self.word_length = word_length
        self.feature_length = feature_length
        self.cnn_layers = cnn_layers
        self.kernel_size = kernel_size
        self.word_size = word_size
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.embeddings = getEmbeddings(self.word_size, self.word_length, self.feature_size, self.feature_length, Wv, pf1, pf2)
        self.cnn = CNNwithPool(self.cnn_layers, self.kernel_size)
        self.drop = nn.Dropout(dropout)
        self.linear = MetaLinear(self.cnn_layers*3, self.num_classes)

    def forward(self, x, ldist, rdist, pool):
        embeddings = self.embeddings(x,ldist,rdist)
        cnn = self.cnn(embeddings,pool).view((embeddings.size(0),-1))  # (bs, )
        cnn_dropout = self.drop(cnn)
        probabilities = self.linear(cnn_dropout)
        return probabilities, []
