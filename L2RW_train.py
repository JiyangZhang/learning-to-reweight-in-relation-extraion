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


hyperparameters = {
    'lr' : 1e-3,
    'momentum' : 0.9,
    'batch_size' : 100,
    'num_iterations' : 8000,
}


print("Please Note That CUDA is required for the model to run")
use_cuda = torch.cuda.is_available()
print(use_cuda)
gpu = 0

def parse_argv(argv):
    opts, args = getopt.getopt(sys.argv[1:], "he:i:o:m:",
                               ['epoch','input','output','mode'])
    epochs = 50
    output = './'
    mode = 1
    inputs = ""
    for op, value in opts:
        #print op,value
        if op == '-e':
            epochs = int(value)
        elif op == '-o':
            output = value
        elif op == '-m':
            mode =int(value)
        elif op == '-i':
            inputs = value
        elif op == '-h':
            #TODO
            #usage()
            sys.exit()
    return [epochs, inputs, output, mode]

class DocumentContainer(object):
    def __init__(self, entity_pair, sentences, label,l_dist,r_dist,entity_pos):
        self.entity_pair = entity_pair
        self.sentences = sentences
        self.label = label
        self.pos = pos
        self.l_dist = l_dist
        self.r_dist = r_dist
        self.entity_pos = entity_pos

def train_lre(train, test, dev, epochs, directory, Wv, pf1, 
    pf2, batch=50, num_classes=53, max_sentences=5, img_h=82, to_train=1, test_epoch=0):
    # initialize the model
    model = PCNN(word_length=len(Wv), feature_length=len(pf1), cnn_layers=230, kernel_size=(3,60), 
        Wv=Wv, pf1=pf1, pf2=pf2, num_classes=num_classes)
    if torch.cuda.is_available():
            model.cuda()
    # prepare the train, dev, test data
    [test_label, test_sents, test_pos, test_ldist, test_rdist, test_entity, test_epos] = bags_decompose(test)
    [dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist, dev_entity, dev_epos] = bags_decompose(dev)
    [train_label, train_sents, train_pos, train_ldist, train_rdist, train_entity, train_epos] = bags_decompose(train)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # start the training
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print "Training:",str(now)
    # start iterations
    train_data, train_labels, train_poss, train_ldists, train_rdists, train_eposs = select_instance3(train_label, train_sents, train_pos, train_ldist, train_rdist, train_epos, img_h, num_classes, max_sentences, model, batch=2000)
    for epoch in range(test_epoch,epochs):
        if to_train == 1:
            total_loss = 0.
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            samples = train_data.shape[0]
            batches = _make_batches(samples, batch)
            index_array = np.arange(samples)
            random.shuffle(index_array)
            #print str(now),"\tStarting Epoch",(epoch),"\tBatches:",len(batches)
            model.train()
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[0:50]
                #batch_ids = index_array[batch_start:batch_end]
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
                #print(eps.volatile)
                meta_model.update_params(0.1, source_params=grads)
                # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
                #meta_model.eval()
                dev_data, dev_labels, dev_poss, dev_ldists, dev_rdists, dev_eposs = select_instance3(dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist,  dev_epos, img_h, num_classes, max_sentences, model, batch=2000)
                # put the data into variable
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
            # test part
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now),"\tDone Epoch",(epoch),"\nLoss:",total_loss
            torch.save({'epoch': epoch ,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, directory+"model_"+str(epoch))

            model.eval()
            dev_predict = get_test3(dev_label, dev_sents, dev_pos, dev_ldist, dev_rdist, dev_epos, img_h, num_classes, max_sentences, model, batch=2000)

            if to_train == 1:
                pickle.dump(dev_predict,open(directory+"predict_prob_dev_"+str(epoch),"wb"))
            else:
                pickle.dump(dev_predict,open(directory+"predict_prob_dev_temp_"+str(epoch),"wb"))
            print("Test")

            dev_pr = pr(dev_predict[3], dev_predict[2], dev_entity)
            accuracy(dev_predict[3], dev_predict[2])
            one_hot = []
            results = dev_predict[3]
            for labels in dev_label:
                arr = np.zeros(shape=(num_classes-1,),dtype='int32')
                for label in labels:
                    if label != 0:
                        arr[label-1] = 1
                one_hot.append(arr)
            one_hot = np.array(one_hot)
            results = results[:,1:]
            score = average_precision_score(one_hot, results, average='micro')
            if to_train == 1:
                out = open(directory+"pr_dev_"+str(epoch),"wb")
            else:
                out = open(directory+"pr_dev_temp_"+str(epoch),"wb")
            for res in dev_pr[3]:
                out.write(str(res[0])+"\t"+str(res[1])+"\n")
            out.close()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            precision = -1
            recall = -1
            print(str(now) + '\t epoch ' + str(epoch) + "\tTest\tScore:"+str(score)+"\t Precision : "+str(dev_pr[0]) + "\t Recall: "+str(dev_pr[1])+ "\t Total: "+ str(dev_pr[2]) + '\n')
            f_log = open('./logs/training_log.txt', 'a+', 1)
            f_log.write(str(now) + '\t epoch ' + str(epoch) + "\tTest\tScore:"+str(score)+
                "\t Precision : "+str(dev_pr[0]) + "\t Recall: "+str(dev_pr[1])+ "\t Total: "+ str(dev_pr[2]) + '\n')
        else:
            print("Loading:","model_"+str(epoch))
            checkpoint = torch.load(directory+"model_"+str(epoch), map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])    
            test_predict = get_test3(test_label, test_sents, test_pos, test_ldist, test_rdist, test_epos, img_h, num_classes, max_sentences, model, batch=2000)
            # test_predict = visualize_attention(test_label, test_sents, test_pos, test_ldist, test_rdist, test_epos, img_h, num_classes, max_sentences, model, batch=2000)
            if to_train == 1:
                pickle.dump(test_predict,open(directory+"predict_prob_"+str(epoch),"wb"))
            else:
                pickle.dump(test_predict,open(directory+"predict_prob_temp_"+str(epoch),"wb"))
            print("Test")

            test_pr = pr(test_predict[3], test_predict[2], test_entity)
            accuracy(test_predict[3], test_predict[2])
            one_hot = []
            results = test_predict[3]
            for labels in test_label:
                arr = np.zeros(shape=(num_classes-1,),dtype='int32')
                for label in labels:
                    if label != 0:
                        arr[label-1] = 1
                one_hot.append(arr)
            one_hot = np.array(one_hot)
            results = results[:,1:]
            score = average_precision_score(one_hot, results, average='micro')
            if to_train == 1:
                out = open(directory+"pr_"+str(epoch),"wb")
            else:
                out = open(directory+"pr_temp_"+str(epoch),"wb")
            for res in test_pr[3]:
                out.write(str(res[0])+"\t"+str(res[1])+"\n")
            out.close()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            precision = -1
            recall = -1
            print(str(now) + '\t epoch ' + str(epoch) + "\tTest\tScore:"+str(score)+"\t Precision : "+str(test_pr[0]) + "\t Recall: "+str(test_pr[1])+ "\t Total: "+ str(test_pr[2]))
            now = time.strftime("%Y-%m-%d %H:%M:%S")

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
    #print "accuracy: ",float(correct)/count, correct, count


def pr(predict_y, true_y,entity_pair):
    final_labels = []
    for label in true_y:
        if 0 in label and len(label) > 1:
            label = [x for x in label if x!=0]
        final_labels.append(label[:])

    total = 0
    for label in final_labels:
        if 0 in label:
            continue
        else:
            total += len(label)
    #print "Total:",total
    results = []
    for i in range(predict_y.shape[0]):
        for j in range(1, predict_y.shape[1]):
            results.append([i,j,predict_y[i][j],entity_pair[i]])
    resultSorted = sorted(results, key=operator.itemgetter(2),reverse=True)
    p_p = 0.0
    p_n = 0.0
    n_p = 0.0
    pr = []
    prec = 0.0
    rec = 0.0
    p_p_final = 0.0
    p_n_final = 0.0
    n_p_final = 0.0
    prev = -1
    for g,item in enumerate(resultSorted):
        prev = item[2]
        if 0 in final_labels[item[0]]:
            if item[1] == 0:
                temp = 1
            else:
                n_p += 1
        else:
            if item[1] in final_labels[item[0]]:
                p_p += 1
            else:
                p_n += 1
        # if g%100 == 0:
            # print "Precision:",(p_p)/(p_p+n_p)
            # print "Recall",(p_p)/total

        try:
            pr.append([(p_p)/(p_p+n_p+p_n), (p_p)/total])
        except:
            pr.append([1.0,(p_p)/total])
        if rec <= 0.3:
            try:
                prec = (p_p)/(p_p+n_p+p_n)
            except:
                prec = 1.0
            rec = (p_p)/total
            p_p_final = p_p
            p_n_final = p_n
            n_p_final = n_p

        if (p_p)/total > 0.7:
            break
    #print "p_p:",p_p_final,"n_p:",n_p_final,"p_n:",p_n_final
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
    print(Wv[0])
    # rng = np.random.RandomState(3435)
    PF1 = np.asarray(np.random.uniform(low=-1, high=1, size=[101, 5]), dtype='float32')
    padPF1 = np.zeros((1, 5))
    PF1 = np.vstack((padPF1, PF1))
    PF2 = np.asarray(np.random.uniform(low=-1, high=1, size=[101, 5]), dtype='float32')
    padPF2 = np.zeros((1, 5))
    PF2 = np.vstack((padPF2, PF2))
    print PF1[0]
    print PF2[0]
    # num_classes??53
    train_lre(train,
                    test,
                    dev,
                    50,
                    resultdir,
                    Wv,
                    PF1,
                    PF2,batch=50, test_epoch=0, to_train=1, num_classes=53)

        

        