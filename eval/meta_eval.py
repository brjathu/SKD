from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import sys, os
from collections import Counter


sys.path.append(os.path.abspath('..'))

from util import accuracy


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=False, is_norm=True, classifier='LR'):
    net = net.eval()
    acc = []

    with torch.no_grad():
        with tqdm(testloader, total=len(testloader)) as pbar:
            for idx, data in enumerate(pbar):
                support_xs, support_ys, query_xs, query_ys = data

                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
                batch_size, _, height, width, channel = support_xs.size()
                support_xs = support_xs.view(-1, height, width, channel)
                query_xs = query_xs.view(-1, height, width, channel)

                
                
#                 batch_size = support_xs.size()[0]
#                 x = support_xs
#                 x_90 = x.transpose(2,3).flip(2)
#                 x_180 = x.flip(2).flip(3)
#                 x_270 = x.flip(2).transpose(2,3)
#                 generated_data = torch.cat((x, x_90, x_180, x_270),0)
#                 support_ys = support_ys.repeat(1,4)
#                 support_xs = generated_data
            
#                 print(support_xs.size())
#                 print(support_ys.size())



                if use_logit:
                    support_features = net(support_xs).view(support_xs.size(0), -1)
                    query_features = net(query_xs).view(query_xs.size(0), -1)
                else:
                    feat_support, _ = net(support_xs, is_feat=True)
                    support_features = feat_support[-1].view(support_xs.size(0), -1)
                    feat_query, _ = net(query_xs, is_feat=True)
                    query_features = feat_query[-1].view(query_xs.size(0), -1)

#                     feat_support, _ = net(support_xs)
#                     support_features = feat_support.view(support_xs.size(0), -1)
#                     feat_query, _ = net(query_xs)
#                     query_features = feat_query.view(query_xs.size(0), -1)


                if is_norm:
                    support_features = normalize(support_features)
                    query_features = normalize(query_features)

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()
                
                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()
                
                
                
                if classifier == 'LR':
                    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                                             multi_class='multinomial')
                    clf.fit(support_features, support_ys)
                    query_ys_pred = clf.predict(query_features)
                elif classifier == 'NN':
                    query_ys_pred = NN(support_features, support_ys, query_features)
                elif classifier == 'Cosine':
                    query_ys_pred = Cosine(support_features, support_ys, query_features)
                else:
                    raise NotImplementedError('classifier not supported: {}'.format(classifier))

                    
#                 bs = query_features.shape[0]//opt.n_aug_support_samples
#                 a = np.reshape(query_ys_pred[:bs], (-1,1))
#                 c = query_ys[:bs]
#                 for i in range(1,opt.n_aug_support_samples):
#                     a = np.hstack([a, np.reshape(query_ys_pred[i*bs:(i+1)*bs], (-1,1))])
                
#                 d = [] 
#                 for i in range(a.shape[0]):
#                     b = Counter(a[i,:])
#                     d.append(b.most_common(1)[0][0])
                
# #                 (values,counts) = np.unique(a,axis=1, return_counts=True)
# #                 print(counts)
# # ind=np.argmax(counts)
# # print values[ind]  # pr


# # #                 a = np.argmax
# #                 print(a.shape)
# #                 print(c.shape)
                    
                acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
                
                pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(metrics.accuracy_score(query_ys, query_ys_pred))})
    
    return mean_confidence_interval(acc)




def meta_test_tune(net, testloader, use_logit=False, is_norm=True, classifier='LR', lamda=0.2):
    net = net.eval()
    acc = []
    
    with tqdm(testloader, total=len(testloader)) as pbar:
        for idx, data in enumerate(pbar):
            support_xs, support_ys, query_xs, query_ys, support_ts, query_ts = data

            support_xs = support_xs.cuda()
            support_ys = support_ys.cuda()
            query_ys = query_ys.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            support_ys = support_ys.view(-1,1)
            query_ys = query_ys.view(-1)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)
               
            y_onehot = torch.FloatTensor(support_ys.size()[0], 5).cuda()

            # In your for loop
            y_onehot.zero_()
            y_onehot.scatter_(1, support_ys, 1)

    
            X = support_features
            XTX = torch.matmul(torch.t(X),X)
            
            B = torch.matmul( (XTX + lamda*torch.eye(640).cuda() ).inverse(), torch.matmul(torch.t(X), y_onehot.float()) )
#             print(B.size())
            m = nn.Sigmoid()
            Y_pred = m(torch.matmul(query_features, B))
                
                
#             print(Y_pred, query_ys)
#             model = nn.Sequential(nn.Linear(64, 10),nn.LogSoftmax(dim=1))
#             optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#             criterion = nn.CrossEntropyLoss()

#             model.cuda()
#             criterion.cuda()
#             model.train()
            
#             for i in range(5):
#                 output = model(support_features)
#                 loss = criterion(output, support_ys)
#                 optimizer.zero_grad()
#                 loss.backward(retain_graph=True) # auto-grad 
#                 optimizer.step() # update  weights 
            
#             model.eval()
#             query_ys_pred = model(query_features)

            acc1, acc5 = accuracy(Y_pred, query_ys, topk=(1, 1))
            
            
#             support_features = support_features.detach().cpu().numpy()
#             query_features = query_features.detach().cpu().numpy()

#             support_ys = support_ys.view(-1).numpy()
#             query_ys = query_ys.view(-1).numpy()

#             if classifier == 'LR':
#                 clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
#                                          multi_class='multinomial')
#                 clf.fit(support_features, support_ys)
#                 query_ys_pred = clf.predict(query_features)
#             elif classifier == 'NN':
#                 query_ys_pred = NN(support_features, support_ys, query_features)
#             elif classifier == 'Cosine':
#                 query_ys_pred = Cosine(support_features, support_ys, query_features)
#             else:
#                 raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(acc1.item()/100.0)

            pbar.set_postfix({"FSL_Acc":'{0:.4f}'.format(np.mean(acc))})
                
                
    return mean_confidence_interval(acc)



def meta_test_ensamble(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    for n in net:
        n = n.eval()
    acc = []

    with torch.no_grad():
        with tqdm(testloader, total=len(testloader)) as pbar:
            for idx, data in enumerate(pbar):
                support_xs, support_ys, query_xs, query_ys = data

                support_xs = support_xs.cuda()
                query_xs = query_xs.cuda()
                batch_size, _, height, width, channel = support_xs.size()
                support_xs = support_xs.view(-1, height, width, channel)
                query_xs = query_xs.view(-1, height, width, channel)

                if use_logit:
                    support_features = net[0](support_xs).view(support_xs.size(0), -1)
                    query_features = net[0](query_xs).view(query_xs.size(0), -1)
                    for n in net[1:]:
                        support_features += n(support_xs).view(support_xs.size(0), -1)
                        query_features += n(query_xs).view(query_xs.size(0), -1)
                else:
                    feat_support, _ = net(support_xs, is_feat=True)
                    support_features = feat_support[-1].view(support_xs.size(0), -1)
                    feat_query, _ = net(query_xs, is_feat=True)
                    query_features = feat_query[-1].view(query_xs.size(0), -1)

                if is_norm:
                    support_features = normalize(support_features)
                    query_features = normalize(query_features)

                support_features = support_features.detach().cpu().numpy()
                query_features = query_features.detach().cpu().numpy()

                support_ys = support_ys.view(-1).numpy()
                query_ys = query_ys.view(-1).numpy()

                if classifier == 'LR':
                    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(support_features, support_ys)
                    query_ys_pred = clf.predict(query_features)
                elif classifier == 'NN':
                    query_ys_pred = NN(support_features, support_ys, query_features)
                elif classifier == 'Cosine':
                    query_ys_pred = Cosine(support_features, support_ys, query_features)
                else:
                    raise NotImplementedError('classifier not supported: {}'.format(classifier))

                acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
                
                pbar.set_postfix({"FSL_Acc":'{0:.2f}'.format(metrics.accuracy_score(query_ys, query_ys_pred))})
                
    return mean_confidence_interval(acc)


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
