"""
grid search for fg 

"""
import networkx as nx
from cdt.data import load_dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from models.Flow_Regression import FlowGraph
from multiprocessing import Pool
import os 
from loguru import logger 
import random


def train_flow_graph(param):
    
    dataset, flow_depth, lr, max_epoch, w_init_sigma, sample_n = param
    
    LocalProcRandGen = np.random.RandomState()
    
    cuda_device = LocalProcRandGen.choice(range(6))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    
    #logger.info("Start {0}.{1}.{2}.{3}.{4}.log".format(dataset, flow_depth, lr, max_epoch, w_init_sigma))
    logger.info("Start {0}.{1}.{2}.{3}.{4}.{5}.log".format(dataset, flow_depth, lr, max_epoch, w_init_sigma, sample_n))

    obj = FlowGraph(sample_n = sample_n, max_epoch = max_epoch, lr =lr, w_init_sigma=w_init_sigma, flow_depth=flow_depth)
    
    if dataset == "sachs" or dataset.startswith("dream4"):
        data, graph = load_dataset(dataset)
        output = obj.orient_graph(data, nx.Graph(graph))     
        
        total_edge_num = 0
        match_edge_num = 0
        for n1, n2 in graph.edges():
            total_edge_num += 1
            if output.has_edge(n1, n2):
                match_edge_num += 1
                
        acc = match_edge_num/ total_edge_num
    else:
        
        data, labels = load_dataset(dataset)
            
        output = obj.predict(data)
        
        try:
            output = np.array(output)
        except:
            print(type(output))
            raise 

        print(labels.Target.shape)
        pair_n = output.shape[0]

        pred = output.reshape((pair_n,))

        pred[pred>0.0] = 1
        pred[pred<=0.0] = -1

        correct = np.zeros(pair_n)

        print(pred.shape)

        correct[pred == labels.Target] = 1
        acc = 1.0*sum(correct)/pair_n
        
    print('accuracy = {}'.format(acc))
    
    with open("{0}.{1}.{2}.{3}.{4}.{5}.ECDV_M_log.{6}".format(dataset, flow_depth, lr, max_epoch, w_init_sigma, sample_n, acc), "w") as f:
        f.write("{0}.{1}.{2}.{3}.{4}.{5}.{6}\n".format(dataset, flow_depth, lr, max_epoch, w_init_sigma, sample_n, acc))
        f.write(str(acc))

            
def grid_search():

    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)


    flow_depth = [2] # 3], 4
    lr = [0.002] # 0.003]
    max_epoch = [800] #, 1000] #2500]
    w_init_sigma = [0.02] #
    data_sets = ["tuebingen"] #["sachs"] ##["dream4-1"] # "tuebingen", "sachs",  "dream4-2", "dream4-3", "dream4-1"
    sample_ns = [200]

    import itertools

    param_grid = itertools.product(data_sets, flow_depth, lr, max_epoch, w_init_sigma, sample_ns)

    with Pool(12) as pool:
        result = pool.map(train_flow_graph, param_grid)
                
                
if __name__ == "__main__":
                
    grid_search()
        
