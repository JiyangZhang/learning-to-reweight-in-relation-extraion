from sub_model import *
import os, sys
import pickle
import math
import time
sys.path.insert(0, '/home/mall/.local/lib/python2.7/site-packages/')
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as Fh
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
import gc
import operator
import random
from sklearn.metrics import average_precision_score
import getopt
from logger import Logger

# the directory to store the tensorboard data 
log = Logger('./lerw/tblog')

# check the GPU
print("Please Note That CUDA is required for the model to run")
use_cuda = torch.cuda.is_available()
print(use_cuda)

class DocumentContainer(object):
    """
    the container to store the data
    """
    def __init__(self, entity_pair, sentences, label, l_dist, r_dist, entity_pos):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos

def train_lre(train, test, dev, epochs, directory, Wv, pf1, 
    pf2, batch=50, num_classes=53, max_sentences=5, img_h=82, to_train=1, test_epoch=0):
    """
    train: the file name of training data
    test: the file name of test data
    dev: the file name of dev data
    epochs: the # iterations
    directory: the directory to store the model
    Wv: the file name of Word Vector data
    batch: batch_size
    num_classes: # classes of the dataset
    img_h: the length of one sentence
    to_train: the training flag
    test_epoch: no use param
    """

    # initialize the model
    model = PCNN(word_length=len(Wv), feature_length=len(pf1), cnn_layers=230, kernel_size=(3,60), 
        Wv=Wv, pf1=pf1, pf2=pf2, num_classes=num_classes)
    if torch.cuda.is_available():
            model.cuda()  # use CUDA
    # wrap the data
    [test_label, test_sents, test_pos, test_ldist, test_rdist, test_entity, test_epos] = bags_decompose(test)
    [dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist, dev_entity, dev_epos] = bags_decompose(dev)
    [train_label, train_sents, train_pos, train_ldist, train_rdist, train_entity, train_epos] = bags_decompose(train)
    optimizer = optim.SGD(model.params(), lr=0.1)
    
    # start the training
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # prepare the training data
    F1 = 0
    for epoch in range(0,epochs):
        if to_train == 1:
            train_data, train_labels, train_poss, train_ldists, train_rdists, train_eposs = select_instance3(train_label, train_sents, train_pos, train_ldist, train_rdist, train_epos, img_h, num_classes, max_sentences, model, batch=2000)
            print "Training:",str(now)
            
            total_loss = 0.
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            samples = train_data.shape[0]
            batches = _make_batches(samples, batch)
            index_array = np.arange(samples)
            random.shuffle(index_array)
            print str(now),"\tStarting Epoch",(epoch+1),"\tBatches:",len(batches)
            model.train()
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_slice = torch.from_numpy(_slice_arrays(train_data, batch_ids)).long().cuda()
                l_slice = torch.from_numpy(_slice_arrays(train_ldists, batch_ids)).long().cuda()
                r_slice = torch.from_numpy(_slice_arrays(train_rdists, batch_ids)).long().cuda()
                e_slice = torch.from_numpy(_slice_arrays(train_eposs, batch_ids)).long().cuda()
                train_labels_slice = torch.from_numpy(_slice_arrays(train_labels, batch_ids)).long().cuda()
                # put the data into variable
                x_batch = autograd.Variable(x_slice, requires_grad=False)
                l_batch = autograd.Variable(l_slice, requires_grad=False)
                r_batch = autograd.Variable(r_slice, requires_grad=False)
                e_batch = e_slice
                train_labels_batch = autograd.Variable(train_labels_slice, requires_grad=False).squeeze(1) 
                # initialize a dummy network for the meta learning of the weights
                meta_model = PCNN(word_length=len(Wv), feature_length=len(pf1), cnn_layers=230, kernel_size=(3,60), 
                    Wv=Wv, pf1=pf1, pf2=pf2, num_classes=num_classes)
                if torch.cuda.is_available():
                    meta_model.cuda()
                meta_model.load_state_dict(model.state_dict())
                # Lines 4 - 5 initial forward pass to compute the initial weighted loss
                results_batch, attention_scores = meta_model(x_batch, l_batch, r_batch, e_batch)
                loss = F.cross_entropy(results_batch, train_labels_batch, reduce=False)
                eps = nn.Parameter(torch.zeros(loss.size()).cuda())
                l_f_meta = torch.sum(loss * eps)  # lf = sigma(eps*loss)
                meta_model.zero_grad()
                # Line 6 perform a parameter update
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
                meta_model.update_params(0.1, source_params=grads)
                # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
                meta_model.eval()
                # prepare the dev data
                dev_data, dev_labels, dev_poss, dev_ldists, dev_rdists, dev_eposs, dev_entity = select_dev3(dev_label, dev_entity,
                    dev_sents, dev_pos, dev_ldist, dev_rdist,  dev_epos, img_h, num_classes, max_sentences, model, batch=batch)
                # put the dev data into variable
                x_dev = autograd.Variable(torch.from_numpy(dev_data).long().cuda(), requires_grad=False)
                l_dev = autograd.Variable(torch.from_numpy(dev_ldists).long().cuda(), requires_grad=False)
                r_dev = autograd.Variable(torch.from_numpy(dev_rdists).long().cuda(), requires_grad=False)
                e_dev = torch.from_numpy(dev_eposs).long().cuda()
                dev_labels = autograd.Variable(torch.from_numpy(dev_labels).long().cuda(), requires_grad=False).squeeze(1)
                # get the result
                results_dev, attention_dev = meta_model(x_dev, l_dev, r_dev, e_dev)
                l_g_meta = F.cross_entropy(results_dev, dev_labels) 
                # calculate the gradients of eps
                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0]
                # Line 11 computing and normalizing the weights
                w_tilde = torch.clamp(-grad_eps,min=0)
                norm_c = torch.sum(w_tilde)
                # calculate the weight
                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde
                # Lines 12 - 14 computing for the loss with the computed weights
                # and then perform a parameter update
                model.train()
                results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
                loss = F.cross_entropy(results_batch, train_labels_batch, reduce=False)
                l_f = torch.sum(loss * w)
                
                total_loss += l_f.data
                # backprop
                optimizer.zero_grad()
                l_f.backward()
                optimizer.step()

                print("epoch {} batch {} training loss: {}" \
                .format(epoch+1, batch_index+1, l_f.data))
                #break
            # validate part
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now),"\tDone Epoch",(epoch+1),"\nLoss:",total_loss
            #torch.save({'epoch': epoch ,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, directory+"modules/model_"+str(epoch))

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
            
            # 1. Log scalar values (scalar summary)
            info = { 'loss': total_loss.item()/samples, 'accuracy': 1.0}
            
            for tag, value in info.items():
                log.scalar_summary(tag, value, epoch+1)
            
            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                log.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                log.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
            print("Now validation starts")
            dev_data, dev_labels, dev_poss, dev_ldists, dev_rdists, dev_eposs = select_instance3(dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist, dev_epos, img_h, num_classes, max_sentences, model, batch=2000)
            samples = dev_data.shape[0]
            batches = _make_batches(samples, batch)
            index_array = np.arange(samples)
            random.shuffle(index_array)
            results = []
            labels = []
            results = np.zeros((samples, num_classes), dtype='float32')
            labels = np.zeros((samples,), dtype='float32')
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_slice = torch.from_numpy(_slice_arrays(dev_data, batch_ids)).long().cuda()
                l_slice = torch.from_numpy(_slice_arrays(dev_ldists, batch_ids)).long().cuda()
                r_slice = torch.from_numpy(_slice_arrays(dev_rdists, batch_ids)).long().cuda()
                e_slice = torch.from_numpy(_slice_arrays(dev_eposs, batch_ids)).long().cuda()
                dev_labels_slice = torch.from_numpy(_slice_arrays(dev_labels, batch_ids)).long().cuda()
                # put the data into variable
                x_batch = autograd.Variable(x_slice, requires_grad=False)
                l_batch = autograd.Variable(l_slice, requires_grad=False)
                r_batch = autograd.Variable(r_slice, requires_grad=False)
                e_batch = e_slice
                dev_labels_batch = autograd.Variable(dev_labels_slice, requires_grad=False).squeeze(1)
                results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
                results[batch_start:batch_end,:] = F.softmax(results_batch, dim=-1).data.cpu().numpy()
                labels[batch_start:batch_end] = dev_labels_batch.data.cpu().numpy()
            # predict the label
            rel_type_arr = np.argmax(results,axis=-1)  # (num_dev, )
            predict_y_dist = np.asarray(np.copy(results))  # (num_dev, num_class)
            # calcualate the precision and recall
            dev_pr = pr(rel_type_arr, labels, dev_entity)
            #accuracy(predict_y_dist, labels)

            one_hot = []
            results = predict_y_dist
            for labels in dev_labels:
                arr = np.zeros(shape=(num_classes-1,),dtype='int32')
                for label in labels:
                    if label != 0:
                        arr[label-1] = 1
                one_hot.append(arr)
            one_hot = np.array(one_hot)
            results = results[:,1:]
            score = average_precision_score(one_hot, results, average='micro')

            
            F1_temp = 2 *(dev_pr[0]*dev_pr[1])/(dev_pr[0]+dev_pr[1])
            if F1_temp > F1:
                F1 = F1_temp
                print("saving the best model....")
                torch.save({'epoch': epoch+1 ,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, directory+"modules/saved_model")
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(str(now) + '\t epoch ' + str(epoch+1) + "\tValidate\tScore:"+str(score)+"\t Precision : "+str(dev_pr[0]) + "\t Recall: "+str(dev_pr[1])+ "\t F1score: "+ str(F1_temp) + '\n')
            f_log = open(directory + 'logs/valid_log.txt', 'a+', 1)
            f_log.write(str(now) + '\t epoch ' + str(epoch+1) + "\tValidate\tScore:"+str(score)+
                "\t Precision : "+str(dev_pr[0]) + "\t Recall: "+str(dev_pr[1])+ "\t F1score: "+ str(F1_temp) + '\n')
            f_log.close()
        else:
            print("Loading:","the saved model") 
            checkpoint = torch.load(directory+"modules/saved_model", map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("start testing")
            test_data, test_labels, test_poss, test_ldists, test_rdists, test_eposs = select_instance3(test_label, test_sents, test_pos, test_ldist, test_rdist, test_epos, img_h, num_classes, max_sentences, model, batch=2000)
            samples = test_data.shape[0]
            batches = _make_batches(samples, batch)
            index_array = np.arange(samples)
            random.shuffle(index_array)
            results = []
            labels = []
            results = np.zeros((samples, num_classes), dtype='float32')
            labels = np.zeros((samples,), dtype='float32')
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                x_slice = torch.from_numpy(_slice_arrays(test_data, batch_ids)).long().cuda()
                l_slice = torch.from_numpy(_slice_arrays(test_ldists, batch_ids)).long().cuda()
                r_slice = torch.from_numpy(_slice_arrays(test_rdists, batch_ids)).long().cuda()
                e_slice = torch.from_numpy(_slice_arrays(test_eposs, batch_ids)).long().cuda()
                test_labels_slice = torch.from_numpy(_slice_arrays(test_labels, batch_ids)).long().cuda()
                # put the data into variable
                x_batch = autograd.Variable(x_slice, requires_grad=False)
                l_batch = autograd.Variable(l_slice, requires_grad=False)
                r_batch = autograd.Variable(r_slice, requires_grad=False)
                e_batch = e_slice
                test_labels_batch = autograd.Variable(test_labels_slice, requires_grad=False).squeeze(1)
                results_batch, attention_scores = model(x_batch, l_batch, r_batch, e_batch)
                results[batch_start:batch_end,:] = F.softmax(results_batch, dim=-1).data.cpu().numpy()
                labels[batch_start:batch_end] = test_labels_batch.data.cpu().numpy()
            # predict the label
            rel_type_arr = np.argmax(results,axis=-1)  # (num_test, )
            predict_y_dist = np.asarray(np.copy(results))  # (num_test, num_class)
            # calcualate the precision and recall
            test_pr = pr(rel_type_arr, labels, test_entity)
            #accuracy(predict_y_dist, test_labels)

            one_hot = []
            results = predict_y_dist
            for labels in test_labels:
                arr = np.zeros(shape=(num_classes-1,),dtype='int32')
                for label in labels:
                    if label != 0:
                        arr[label-1] = 1
                one_hot.append(arr)
            one_hot = np.array(one_hot)
            results = results[:,1:]
            score = average_precision_score(one_hot, results, average='micro')

            
            F1_temp = 2 *(test_pr[0]*test_pr[1])/(test_pr[0]+test_pr[1])
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(str(now) + "\tTest\tScore:"+str(score)+"\t Precision : "+str(test_pr[0]) + "\t Recall: "+str(test_pr[1])+ "\t F1score: "+ str(F1_temp) + '\n')
            f_log = open(directory + 'logs/test_log.txt', 'a+', 1)
            f_log.write(str(now) + "\tTest\tScore:"+str(score)+
                "\t Precision : "+str(test_pr[0]) + "\t Recall: "+str(test_pr[1])+ "\t F1score: "+ str(F1_temp) + '\n')
            f_log.close()
            break

def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_label = [data_bag.label for data_bag in data_bags]
    bag_pos = [data_bag.pos for data_bag in data_bags]
    bag_ldist = [data_bag.l_dist for data_bag in data_bags]
    bag_rdist = [data_bag.r_dist for data_bag in data_bags]
    bag_entity = [data_bag.entity_pair for data_bag in data_bags]
    bag_epos = [data_bag.entity_pos for data_bag in data_bags]
    return [bag_label, bag_sent, bag_pos, bag_ldist, bag_rdist, bag_entity, bag_epos]

def accuracy(predict_y, true_y):
    correct = 0
    count = 0
    for i,label in enumerate(true_y):
        if len(true_y[i]) ==1 and true_y[i][0] == 0:
            continue
        else:
            count += 1
            if np.argmax(predict_y[i]) in true_y[i]:
                correct += 1
    print "accuracy: ",float(correct)/count, correct, count


def pr(predict_y, true_y, entity_pair):
    """
    predict_y: (#instance, )
    true_y: (#instance, 1)
    """
    #total = np.shape(true_y)[0]
    total = 0
    for label in true_y:
        if 0 == label:
            continue
        else:
            total += 1
    print "Total:",total  # exclude the label 0
    
    p_p = 0.0  # true postive
    p_n = 0.0  # false postive
    n_p = 0.0  # false negative
    pr = []
    prec = 0.0
    rec = 0.0
    p_p_final = 0.0
    p_n_final = 0.0
    n_p_final = 0.0
    prev = -1
    for real,pred in zip(true_y,predict_y):
        if real == 0:
            if pred == 0:
                temp = 1  # true negative
            else:
                n_p += 1  # false positve
        else:
            if pred == real:
                p_p += 1  # true postive
            else:
                p_n += 1  # false negative
        try:
            pr.append([(p_p)/(p_p+n_p+p_n), (p_p)/total])  # pr = [precision, recall]
        except:
            pr.append([1.0,(p_p)/total])
        
        try:
            prec = (p_p)/(p_p+n_p)  # precision = (relevant)/total
        except: 
            prec = 1.0
        rec = (p_p)/total  # recall = (relevant)/total
        p_p_final = p_p
        p_n_final = p_n
        n_p_final = n_p

    print ("p_p:",p_p_final,"n_p:",n_p_final,"p_n:",p_n_final)
    return [prec,rec,total,pr]

def _make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def _slice_arrays(arrays, start=None, stop=None):
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]

def get_test3(label, sents, pos, ldist, rdist, epos, img_h , numClasses, maxSentences, testModel, filterSize = 3, batch = 1000):
    numBags = len(label)
    predict_y = np.zeros((numBags), dtype='int32')
    predict_y_prob = np.zeros((numBags), dtype='float32')
    predict_y_dist = np.zeros((numBags, numClasses), dtype='float32')
    # y = np.asarray(rels, dtype='int32')
    numSentences = 0
    for ind in range(len(sents)):
        numSentences += len(sents[ind])
    #print "Num Sentences:", numSentences
    insX = np.zeros((numSentences, img_h), dtype='int32')
    insPf1 = np.zeros((numSentences, img_h), dtype='int32')
    insPf2 = np.zeros((numSentences, img_h), dtype='int32')
    insPool = np.zeros((numSentences, 2), dtype='int32')
    currLine = 0
    for bagIndex, insRel in enumerate(label):
        insNum = len(sents[bagIndex])
        for m in range(insNum):
            insX[currLine] = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
            insPf1[currLine] = np.asarray(ldist[bagIndex][m], dtype='int32').reshape((1, img_h))
            insPf2[currLine] = np.asarray(rdist[bagIndex][m], dtype='int32').reshape((1, img_h))
            epos[bagIndex][m] = sorted(epos[bagIndex][m])
            if epos[bagIndex][m][0] > 79:
                epos[bagIndex][m][0] = 79
            if epos[bagIndex][m][1] > 79:
                epos[bagIndex][m][1] = 79
            if epos[bagIndex][m][0] == epos[bagIndex][m][1]:
                insPool[currLine] = np.asarray([epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2) + 1], dtype='int32').reshape((1, 2))
            else:
                insPool[currLine] = np.asarray([epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2)], dtype='int32').reshape((1, 2))
            currLine += 1
    insX = np.array(insX.tolist())
    insPf1 = np.array(insPf1.tolist())
    insPf2 = np.array(insPf2.tolist())
    insPool = np.array(insPool.tolist())
    results = []
    totalBatches = int(math.ceil(float(insX.shape[0])/batch))
    results = np.zeros((numSentences, numClasses), dtype='float32')
    #print "totalBatches:",totalBatches
    samples = insX.shape[0]
    batches = _make_batches(samples, batch)
    index_array = np.arange(samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        # print batch_index, (batch_start, batch_end)
        batch_ids = index_array[batch_start:batch_end]
        x_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insX, batch_ids)).long().cuda(gpu), volatile=True)
        l_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insPf1, batch_ids)).long().cuda(gpu), volatile=True)
        r_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insPf2, batch_ids)).long().cuda(gpu), volatile=True)
        # e_batch = autograd.Variable(torch.from_numpy(_slice_arrays(insPool, batch_ids)).long().cuda(gpu), volatile=True)
        e_batch = torch.from_numpy(_slice_arrays(insPool, batch_ids)).long().cuda(gpu)
        results_batch, attention_scores = testModel(x_batch, l_batch, r_batch, e_batch)
        results[batch_start:batch_end,:] = F.softmax(results_batch, dim=-1).data.cpu().numpy()
    # print results
    rel_type_arr = np.argmax(results,axis=-1)
    max_prob = np.amax(results, axis=-1)
    currLine = 0
    for bagIndex, insRel in enumerate(label):
        insNum = len(sents[bagIndex])
        maxP = -1
        pred_rel_type = 0
        max_pos_p = -1
        positive_flag = False
        max_vec = []

        for m in range(insNum):
            rel_type = rel_type_arr[currLine]  # the type of this line
            if positive_flag and rel_type == 0:
                currLine += 1
                continue
            else:
                # at least one instance is positive
                tmpMax = max_prob[currLine]  # the max predicted prob of this sentence
                if rel_type > 0:
                    positive_flag = True
                    if tmpMax > max_pos_p:
                        max_pos_p = tmpMax
                        pred_rel_type = rel_type
                        max_vec = np.copy(results[currLine])  # the result prob distribution
                else:
                    if tmpMax > maxP:
                        maxP = tmpMax
                        max_vec = np.copy(results[currLine])
                currLine += 1
        if positive_flag:
            predict_y_prob[bagIndex] = max_pos_p
        else:
            predict_y_prob[bagIndex] = maxP
        predict_y_dist[bagIndex] =  np.asarray(np.copy(max_vec), dtype='float32').reshape((1,numClasses))
        predict_y[bagIndex] = pred_rel_type
    return [predict_y, predict_y_prob, label, predict_y_dist]

def select_instance3(label, sents, pos, ldist, rdist, epos, img_h, numClasses, maxSentences, testModel, filterSize = 3,batch=1000):
    """
    preprocess the data: take every sentence as an example
    """
    numBags = len(label)
    bagIndexX = 0
    totalSents = 0
    for bagIndex, insNum in enumerate(sents):
        totalSents += len(insNum)

    x = np.zeros((totalSents, img_h), dtype='int32')
    p = np.zeros((totalSents, img_h), dtype='int32')
    l = np.zeros((totalSents, img_h), dtype='int32')
    r = np.zeros((totalSents, img_h), dtype='int32')
    e = np.zeros((totalSents, 2), dtype='int32')
    lab = np.zeros((totalSents, 1), dtype='int32')
    curr = 0
    for bagIndex, insNum in enumerate(sents):
        if len(insNum) > 0:
            for m in range(len(insNum)):
                x[curr,:] = sents[bagIndex][m]
                l[curr,:] = ldist[bagIndex][m]
                r[curr,:] = rdist[bagIndex][m]
                lab[curr, :] = [label[bagIndex][0]]
                epos[bagIndex][m] = sorted(epos[bagIndex][m])
                if epos[bagIndex][m][0] > 79:
                    epos[bagIndex][m][0] = 79
                if epos[bagIndex][m][1] > 79:
                    epos[bagIndex][m][1] = 79
                if epos[bagIndex][m][0] == epos[bagIndex][m][1]:
                    e[curr,:] = [epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2) + 1]
                else:
                    e[curr,:] = [epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2)]
                curr += 1
    x = np.array(x.tolist())
    l = np.array(l.tolist())
    r = np.array(r.tolist())
    e = np.array(e.tolist())
    
    return [x, lab, p, l, r, e]

def select_dev3(label, entity, sents, pos, ldist, rdist, epos, img_h, numClasses, maxSentences, testModel, filterSize = 3,batch=1000):
    """
    preprocess the data: take every sentence as an example
    """
    numBags = len(label)
    bagIndexX = 0
    totalSents = 0
    for bagIndex, insNum in enumerate(sents):
        totalSents += len(insNum)

    x = np.zeros((totalSents, img_h), dtype='int32')
    p = np.zeros((totalSents, img_h), dtype='int32')
    l = np.zeros((totalSents, img_h), dtype='int32')
    r = np.zeros((totalSents, img_h), dtype='int32')
    e = np.zeros((totalSents, 2), dtype='int32')
    lab = np.zeros((totalSents, 1), dtype='int32')
    dev_entity = []
    curr = 0
    for bagIndex, insNum in enumerate(sents):
        if len(insNum) > 0:
            for m in range(len(insNum)):
                x[curr,:] = sents[bagIndex][m]
                l[curr,:] = ldist[bagIndex][m]
                r[curr,:] = rdist[bagIndex][m]
                lab[curr, :] = [label[bagIndex][0]]
                dev_entity.append(entity[bagIndex])
                epos[bagIndex][m] = sorted(epos[bagIndex][m])
                if epos[bagIndex][m][0] > 79:
                    epos[bagIndex][m][0] = 79
                if epos[bagIndex][m][1] > 79:
                    epos[bagIndex][m][1] = 79
                if epos[bagIndex][m][0] == epos[bagIndex][m][1]:
                    e[curr,:] = [epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2) + 1]
                else:
                    e[curr,:] = [epos[bagIndex][m][0]+int(filterSize/2), epos[bagIndex][m][1]+ int(filterSize/2)]
                curr += 1
    sample_index = np.random.randint(0, totalSents, batch)
    x = np.array(x.tolist())[sample_index]
    l = np.array(l.tolist())[sample_index]
    r = np.array(r.tolist())[sample_index]
    e = np.array(e.tolist())[sample_index]
    lab = np.array(lab.tolist())[sample_index]
    
    return [x, lab, p, l, r, e, dev_entity]


if __name__ == "__main__":

    if len(sys.argv) < 6:
        print "Please enter the arguments correctly!"
        print(len(sys.argv))
        sys.exit()

    inputdir = sys.argv[1] + "/"
    resultdir = inputdir
    resultdir = "lerw/"
    print 'result dir='+resultdir
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)


    dataType = "_features_all_6Months"
    test = pickle.load(open(inputdir+sys.argv[3]))
    train = pickle.load(open(inputdir+sys.argv[2]))
    dev = pickle.load(open(inputdir+sys.argv[4]))
    print 'load Wv ...'
    Wv = np.array(pickle.load(open(inputdir+sys.argv[5])))

    # Wv = np.random.random((10,50))
    # Wv[0] = Wv[0]*0
    #print(Wv[0])
    # rng = np.random.RandomState(3435)
    PF1 = np.asarray(np.random.uniform(low=-1, high=1, size=[101, 5]), dtype='float32')
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(np.random.uniform(low=-1, high=1, size=[101, 5]), dtype='float32')
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))
    #print PF1[0]
    #print PF2[0]

    train_lre(train,
                    test,
                    dev,
                    50,
                    resultdir,
                    Wv,
                    PF1,
                    PF2,batch=50, test_epoch=0, to_train=0, num_classes=5)

        

        
