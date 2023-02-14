import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



from model import *
from utils import *

fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}

coarse_id_fine_id = {0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92], 3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87], 6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97], 9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38], 12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99], 14: [2, 11, 35, 46, 98], 15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96], 18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89]}

coarse_split={'train': [1,2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19], 'valid': [8, 11, 13, 16], 'test': [0, 7, 12, 14]}

from collections import defaultdict

fine_split=defaultdict(list)

for fine_id,sparse_id in fine_id_coarse_id.items():
    if sparse_id in coarse_split['train']:
        fine_split['train'].append(fine_id)
    elif sparse_id in coarse_split['valid']:
        fine_split['valid'].append(fine_id)  
    else:
        fine_split['test'].append(fine_id)  

fine_split_train_map={class_:i for i,class_ in enumerate(fine_split['train'])}
        
#train_class2id={class_id: i for i, class_id in enumerate(fine_split['train'])}
        
        
import torchvision.transforms as transforms

#FC100
normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

#miniImageNet
mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
normalize = transforms.Normalize(mean=mean_pix,
                                 std=std_pix)
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize
])


# data prep for test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize])


transform_train=transform_test
def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm+1e-9)
    return out
        
        
        
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet12', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='huffpost', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01, 0.0005, 0.005)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    
    
    parser.add_argument('--method', type=str, default='new',
                        help='few-shot or normal')
    parser.add_argument('--mode', type=str, default='few-shot',
                        help='few-shot or normal')
    parser.add_argument('--N', type=int, default=5, help='number of ways')
    parser.add_argument('--K', type=int, default=1, help='number of shots')
    parser.add_argument('--Q', type=int, default=5, help='number of queries')   
    parser.add_argument('--num_train_tasks', type=int, default=500, help='number of meta-training tasks (5)')
    parser.add_argument('--num_test_tasks', type=int, default=10, help='number of meta-test tasks')
    parser.add_argument('--num_true_test_ratio', type=int, default=10, help='number of meta-test tasks (10)')
    parser.add_argument('--fine_tune_steps', type=int, default=10//5, help='number of meta-learning steps (5)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1/5, help='number of meta-learning lr (0.05)')
    parser.add_argument('--meta_lr', type=float, default=0.5/500, help='number of meta-learning lr (0.05)')
    parser.add_argument('--comm_round', type=int, default=5000, help='number of maximum communication roun')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    
    
    parser.add_argument("--bert_cache_dir", default=None, type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))
    parser.add_argument("--wv_path", type=str,
                        default="./",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", type=bool, default=False)
    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))
    
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100' or args.dataset=='FC100' or args.dataset=='miniImageNet':
        total_classes=64 #100
    elif args.dataset == '20newsgroup':
        total_classes=20

    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2

        
    if args.mode=='few-shot':
        n_classes=args.N
        
    if args.mode=='few-shot' and args.method=='MAML':
        if args.normal_model:
            for net_i in range(n_parties):
                if args.model == 'simple-cnn':
                    net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                if device == 'cpu':
                    net.to(device)
                else:
                    net = net.cuda()
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.use_project_head:
                    net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
                else:
                    net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
                if device == 'cpu':
                    net.to(device)
                else:
                    net = net.cuda()
                nets[net_i] = net
    elif args.mode=='few-shot' and args.method=='new':
        for net_i in range(n_parties):
            if args.dataset!='20newsgroup':
                net = ModelFed_Adp(args.model, args.out_dim, n_classes, total_classes, net_configs, args)
            else:
                net = LSTMAtt( args.out_dim, n_classes, total_classes,args)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

            
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net_few_shot_new(net_id, net, n_epoch, lr, args_optimizer, args, X_train_client,y_train_client, X_test, y_test,
                                        device='cpu', test_only=False):
    #net = nn.DataParallel(net)
    #net=nn.parallel.DistributedDataParallel(net)
    #net.cuda()

    #logger.info('Training network %s' % str(net_id))
    #logger.info('n_training: %d' % X_train_client.shape[0])
    #logger.info('n_test: %d' % X_test.shape[0])
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam( net.parameters(), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    loss_ce = nn.CrossEntropyLoss()
    

                   
    def train_epoch(epoch, mode='train'):

        if mode == 'train':
            net.train()
            optimizer.zero_grad()
            X_transform=transform_train
        else:
            net.eval()
            X_transform=transform_test
            
        if mode == 'train':
            if args.dataset=='FC100':
                class_dict = fine_split['train']
            else:
                class_dict=list(range(64))
            X=X_train_client
            y=y_train_client
            #for i in class_dict:  
                #class_dict[i] = class_dict[i][:avail_train_num_per_class]
        elif mode == 'test':
            if args.dataset=='FC100':
                class_dict = fine_split['test']
            else:
                class_dict=list(range(20))
            X=X_test
            y=y_test





        min_size=0
        while min_size<K+Q:
            X_class=[]
            classes = np.random.choice(class_dict, N, replace=False).tolist()
            for i in classes:
                X_class.append(X[y==i])      
            min_size=min([one.shape[0] for one in X_class])




        X_total_sup=[]
        X_total_query=[]
        y_sup=[]
        y_query=[]
        for class_, X_class_i in zip(classes, X_class):
            sample_idx=np.random.choice(list(range(X_class_i.shape[0])), K+Q, replace=False).tolist()
            X_total_sup.append(X_class_i[sample_idx[:K]])
            X_total_query.append(X_class_i[sample_idx[K:]])
            if mode=='train':
                if args.dataset=='FC100':
                    y_sup.append(torch.ones(K)*fine_split_train_map[class_])
                    y_query.append(torch.ones(Q) * fine_split_train_map[class_])
                elif args.dataset=='miniImageNet':
                    y_sup.append(torch.ones(K)*class_)
                    y_query.append(torch.ones(Q) * class_)

                y_total = torch.cat([torch.cat(y_sup, 0), torch.cat(y_query, 0)], 0).long().cuda()
        #y_total=torch.tensor(np.concatenate([np.concatenate(y_sup, 0),np.concatenate(y_query, 0)],0)).cuda()

        
        X_total_sup=np.concatenate(X_total_sup, 0)
        X_total_query=np.concatenate(X_total_query,0)



        X_total_transformed_sup=[]
        X_total_transformed_query=[]
        for i in range(X_total_sup.shape[0]):
            X_total_transformed_sup.append(X_transform(X_total_sup[i]))
        X_total_sup=torch.stack(X_total_transformed_sup,0).cuda()



        for i in range(X_total_query.shape[0]):
            X_total_transformed_query.append(X_transform(X_total_query[i]))
        X_total_query=torch.stack(X_total_transformed_query,0).cuda()




        net_new=copy.deepcopy(net)

        for j in range(args.fine_tune_steps):

            
            X_out_sup,_,out = net_new(X_total_sup)


            loss = loss_ce(out, support_labels)
            
            net_para=net_new.state_dict()
            param_require_grad={}

            for key, param in net_new.named_parameters():
                if mode=='train':
                    if key=='few_classify.weight' or key=='few_classify.bias':
                    #if key !='module.all_classify.weight' and key !='module.all_classify.bias':
                        param_require_grad[key]=param
                else:
                    #if key == 'few_classify.weight' or key == 'few_classify.bias':
                    if key!='all_classify.weight' and key!='all_classify.bias':
                        param_require_grad[key] = param


            grad = torch.autograd.grad(loss, param_require_grad.values())
            #print(grad)
            if torch.any(grad[0].isnan()):
                print(loss)
                print(grad)
                print(1/0)
                       
            for key, grad_ in zip(param_require_grad.keys(), grad):
                net_para[key]=net_para[key]-args.fine_tune_lr*grad_
                
            #net_para = list(
            #                map(lambda p: p[1] - fine_tune_lr * p[0], zip(grad, net_para)))
            #net_para={key:value for key, value in zip(net.state_dict().keys(),net.state_dict().values())}
            
            net_new.load_state_dict(net_para)

        X_out_query,_,out = net_new(X_total_query)


        #net.load_state_dict(net_para_ori)


        #_,_,out_all=net_new(torch.cat([X_total_sup, X_total_query],0), all_classify=True)

        
        

            #print(out[:3])
        if mode == 'train':
            loss = loss_ce(out, query_labels)
            # all_classify update
            X_out_sup, _, out_all = net(torch.cat([X_total_sup, X_total_query], 0), all_classify=True)
            # _, _, out_all = net(X_total_sup, all_classify=True)


            loss_all = loss_ce(out_all, y_total)
            loss_all.backward()

            optimizer.step()

            # few_classify update
            net_para_ori=net.state_dict()
            param_require_grad={}
            for key, param in net_new.named_parameters():
                if key=='few_classify.weight' or key=='few_classify.bias':
                #if key != 'module.all_classify.weight' and key != 'module.all_classify.bias':
                    param_require_grad[key]=param

            grad = torch.autograd.grad(loss, param_require_grad.values())
#


            for key, grad_ in zip(param_require_grad.keys(), grad):
                net_para_ori[key]=net_para_ori[key]-args.meta_lr*grad_
            net.load_state_dict(net_para_ori)

            if np.random.rand() < 0.005:
                print('loss: {:.4f}'.format(loss_all.item()))
            #    print(out)
            #    print('loss: {:.4f}'.format(loss_all.item()))

            acc_train = (torch.argmax(out_all, -1) == y_total).float().mean().item()

            del net_new, X_out_sup, X_out_query, out, out_all
            return acc_train

        else:
            if torch.any(X_out_sup.isnan()) or torch.any(X_out_query.isnan()):
                print(loss)
                print(grad)
                print('1111111111111111111111')
                #print(net_new.named_parameters())
                print(X_total_sup)
                print(X_total_query)
                print(X_out_sup)
                print(X_out_query)
            use_logistic=False

            if use_logistic:
                X_out_sup, _, out = net_new(X_total_sup)
                support_features = l2_normalize(X_out_sup.detach().cpu()).numpy()
                query_features = l2_normalize(X_out_query.detach().cpu()).numpy()


                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_labels.detach().cpu().numpy())





                query_ys_pred = clf.predict(query_features)

                out=torch.tensor(clf.predict_proba(query_features)).cuda()

                acc_train = (torch.argmax(out, -1) == query_labels).float().mean().item()
                max_value, index=torch.max(out,-1)

                del net_new, X_out_sup, X_out_query, out, param_require_grad, grad
                if test_only:
                    return acc_train, max_value, index
                else:
                    return acc_train

                #return metrics.accuracy_score(query_labels.detach().cpu().numpy(), query_ys_pred)

            else:

                acc_train = (torch.argmax(out, -1) == query_labels).float().mean().item()
                with torch.no_grad():
                    max_value, index=torch.max(out,-1)



                del net_new, X_out_sup, X_out_query, out,net_para, param_require_grad, grad, X_total_query, X_total_sup
                if test_only:
                    return acc_train, max_value, index
                else:
                    return acc_train
    
    if not test_only:
        best_acc = 0
        accs_train=[]
        for epoch in range(args.num_train_tasks):
            accs_train.append(train_epoch(epoch))
            if np.random.rand() < 0.005:
                print("Meta-train_Accuracy: {:.4f}".format(np.mean(accs_train)))


        accs=[]
        for epoch_test in range(args.num_test_tasks):
            accs.append(train_epoch(epoch_test, mode='test'))
    else:
        accs=[]
        max_values=[]
        indices=[]
        accs_train=[]

        for epoch in range(args.num_train_tasks):
            accs_train.append(train_epoch(epoch))

        for epoch_test in range(args.num_test_tasks*args.num_true_test_ratio):
            acc, max_value, index=train_epoch(epoch_test, mode='test')
            accs.append(acc)
            max_values.append(max_value)
            indices.append(index)
            del acc, max_value, index

        return np.mean(accs), torch.cat(max_values,0), torch.cat(indices,0)

    if np.random.rand()<0.3:
        print("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))
    #logger.info("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))
    
    return  np.mean(accs)
    
    

def train_net_few_shot(net_id, net, n_epoch, lr, args_optimizer, args, X_train_client,y_train_client, X_test, y_test,
                                        device='cpu', test_only=False):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % X_train_client.shape[0])
    logger.info('n_test: %d' % X_test.shape[0])
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    loss_ce = nn.CrossEntropyLoss()
    
    N=args.N
    K=args.K
    Q=args.Q

                   
    def train_epoch(epoch, mode='train'):
        if mode == 'train':
            net.train()
            optimizer.zero_grad()
            X_transform=transform_train
        else:
            net.eval()
            X_transform=transform_test
            
        if mode == 'train':
            class_dict = fine_split['train']
            X=X_train_client
            y=y_train_client
            #for i in class_dict:  
                #class_dict[i] = class_dict[i][:avail_train_num_per_class]
        elif mode == 'test':
            class_dict = fine_split['test']
            X=X_test
            y=y_test
                      
        
        min_size=0
        while min_size<K+Q:
            X_class=[]
            classes = np.random.choice(class_dict, N, replace=False).tolist()
            for i in classes:
                X_class.append(X[y==i])      
            min_size=min([one.shape[0] for one in X_class])
            
        X_total_sup=[]
        X_total_query=[]
        for X_class_i in X_class:
            sample_idx=np.random.choice(list(range(X_class_i.shape[0])), K+Q, replace=False).tolist()
            X_total_sup.append(X_class_i[sample_idx[:K]])
            X_total_query.append(X_class_i[sample_idx[K:]])
            
        
        X_total_sup=np.concatenate(X_total_sup, 0)
        X_total_query=np.concatenate(X_total_query,0)
        
        
        X_total_transformed_sup=[]
        X_total_transformed_query=[]
        for i in range(X_total_sup.shape[0]):
            X_total_transformed_sup.append(X_transform(X_total_sup[i]))
        X_total_sup=torch.stack(X_total_transformed_sup,0)   
        
        for i in range(X_total_query.shape[0]):
            X_total_transformed_query.append(X_transform(X_total_query[i]))
        X_total_query=torch.stack(X_total_transformed_query,0)   

        
        
        net_para_ori=net.state_dict()
        param_ori={}
        for key, param in net.named_parameters():
            if param.requires_grad:
                param_ori[key]=param    
        
        
        net_new=copy.deepcopy(net)

        for j in range(args.fine_tune_steps):
            
            _,_,out = net_new(X_total_sup)    
            loss = loss_ce(out, support_labels)
            
            net_para=net_new.state_dict()
            param_require_grad={}
            for key, param in net_new.named_parameters():
                if param.requires_grad:
                    param_require_grad[key]=param
                
            grad = torch.autograd.grad(loss, param_require_grad.values())
                       
            for key, grad_ in zip(param_require_grad.keys(), grad):
                net_para[key]=net_para[key]-args.fine_tune_lr*grad_
                
            #net_para = list(
            #                map(lambda p: p[1] - fine_tune_lr * p[0], zip(grad, net_para)))
            #net_para={key:value for key, value in zip(net.state_dict().keys(),net.state_dict().values())}
            
            net_new.load_state_dict(net_para)
              
        _,_,out = net_new(X_total_query)
        
        
        #net.load_state_dict(net_para_ori)
        loss = loss_ce(out, query_labels)
        if epoch==10:
            print('loss: {:.4f}'.format(loss.item()))
            #print(out[:3])
        if mode == 'train':

            param_require_grad={}
            for key, param in net_new.named_parameters():
                if param.requires_grad:
                    param_require_grad[key]=param
                
            grad = torch.autograd.grad(loss, param_require_grad.values())
                            
            for key, grad_ in zip(param_require_grad.keys(), grad):
                net_para_ori[key]=net_para_ori[key]-args.meta_lr*grad_
                
            
            net.load_state_dict(net_para_ori)

            #loss.backward()
            optimizer.step()
            
        acc_train = (torch.argmax(out, -1) == query_labels).float().mean().item()
        return acc_train
    
    if not test_only:
        best_acc = 0
        accs_train=[]
        for epoch in range(args.num_train_tasks):
            accs_train.append(train_epoch(epoch))
        print("Meta-train_Accuracy: {:.4f}".format(np.mean(accs_train))) 

        accs=[]
        for epoch_test in range(args.num_test_tasks):
            accs.append(train_epoch(epoch_test, mode='test'))
    else:
        accs=[]
        for epoch_test in range(args.num_test_tasks*10):
            accs.append(train_epoch(epoch_test, mode='test'))
    

    print("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))
    logger.info("Meta-test_Accuracy: {}".format(np.mean(accs)))
    
    return  np.mean(accs)
    
    
    
    
    
    

def local_train_net_few_shot(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, device="cpu", test_only=False):
    avg_acc = 0.0
    acc_list = []
    max_value_all_clients=[]
    indices_all_clients=[]

    for net_id, net in nets.items():
        print(net_id)

        net.cuda()

        dataidxs = net_dataidx_map[net_id]

        #logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        
    
        n_epoch = args.epochs
        
        #_,_, train_ds, test_ds = get_dataloader(args.dataset, args.datadir, args.batch_size, len(dataidxs), dataidxs)
        
        #X_train_client=train_ds.data
        #y_train_client=train_ds.target
        
        X_train_client=X_train[dataidxs]
        y_train_client=y_train[dataidxs]
        
        #X_test=test_ds.data
        #y_test=test_ds.target


        if args.method=='MAML':
            testacc = train_net_few_shot(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client,y_train_client,X_test, y_test,
                                            device=device, test_only=test_only)
        elif args.method=='new':
            if test_only==False:
                testacc = train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client,y_train_client,X_test, y_test,
                                            device=device, test_only=False)
            else:
                np.random.seed(1)
                testacc, max_values, indices=train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client,y_train_client,X_test, y_test,
                                            device=device, test_only=True)
                max_value_all_clients.append(max_values)
                indices_all_clients.append(indices)
                np.random.seed(int(time.time()))

        #logger.info("net {} final test acc {:.4f}" .format(net_id, testacc))

        
        avg_acc += testacc
        acc_list.append(testacc)

        net.cpu()



    logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
    print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

    if test_only:
        max_value_all_clients=torch.stack(max_value_all_clients,0)
        indices_all_clients=torch.stack(indices_all_clients,0)
        return acc_list, max_value_all_clients, indices_all_clients

    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))

    return nets






if __name__ == '__main__':
    args = get_args()
    print(args)
    
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    test_task_sample_seed=1
    np.random.seed(test_task_sample_seed)
    test_classes=[]
    test_index=[]
    for i in range(args.num_test_tasks):
        test_classes.append(np.random.choice(fine_split['test'], args.N, replace=False).tolist())
        test_index.append(np.random.rand(args.N, args.K+args.Q))






    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    print(X_train.shape)
    print(X_test.shape)
    N=args.N
    K=args.K
    Q=args.Q

    support_labels=torch.zeros(N*K,dtype=torch.long)
    for i in range(N):
        support_labels[i * K:(i + 1) * K] = i
    query_labels=torch.zeros(N*Q,dtype=torch.long)
    for i in range(N):
        query_labels[i * Q:(i + 1) * Q] = i
    if args.device!='cpu':
        support_labels=support_labels.cuda()
        query_labels=query_labels.cuda()
    
    
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    #train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
    #                                                                           args.datadir,
    #                                                                           args.batch_size,
    #                                                                           32)


    #train_dl=None
    #data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'fedavg':
        use_minus=False
        best_acc=0
        best_confident_acc=0
        
        
        for round in range(n_comm_rounds):
            #logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}


            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
            
            for net_id, net in nets_this_round.items():
                if use_minus:
                    net_para = net.state_dict()
                    for key in net_para:
                        net_para[key]=(global_w[key]*total_data_points-net_para[key]*len(net_dataidx_map[net_id]))/(total_data_points+1e-9-len(net_dataidx_map[net_id]))    
                    net.load_state_dict(net_para)
                else:
                    if args.mode=='few-shot' and args.method=='MAML':
                        net.load_state_dict(global_w)
                    elif args.mode=='few-shot' and args.method=='new':
                        net_para = net.state_dict()
                        for key in net_para:
                            if key!='few_classify.weight' and key!='few_classify.bias':
                                net_para[key]=global_w[key]
                        net.load_state_dict(net_para)

                        #net.load_state_dict(global_w)
                    
                
            
            if args.mode=='few-shot':
                global_acc, max_value_all_clients, indices_all_clients=local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device, test_only=True)
                global_acc = max(global_acc)
                if global_acc > best_acc:
                    best_acc = global_acc

                max_values=max_value_all_clients.transpose(0,1).argmax(-1) #[2500]
                labels_predicted=indices_all_clients.transpose(0,1).gather(-1,max_values.unsqueeze(-1)).squeeze()
                query_labels_total=query_labels.repeat([args.num_test_tasks*args.num_true_test_ratio])
                confident_pred=(labels_predicted==query_labels_total).float().mean().item()

                if confident_pred>best_confident_acc:
                    best_confident_acc=confident_pred

                print('>> Global Model Test accuracy: {:.4f} Best Acc: {:.4f} Confident Acc: {:.4f} Confident_Best Acc: {:.4f} '.format(global_acc, best_acc, confident_pred, best_confident_acc))
                logger.info('>> Global Model Test accuracy: {:.4f} Best Acc: {:.4f} Confident Acc: {:.4f} Confident_Best Acc: {:.4f} '.format(global_acc, best_acc, confident_pred, best_confident_acc))

                local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)
            else:
                local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)
            


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


                        
            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            #logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            #train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            #test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Current Round: {}'.format(round))
            
            
            
            #logger.info('>> Global Model Train accuracy: %f' % train_acc)

            #logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            #7global_model.to('cpu')

            if confident_pred > best_confident_acc:
                torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')
