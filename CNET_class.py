#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 19:38:16 2018
The class of CNet

@author: shashajin
"""
import numpy as np
from Util import *
from CLT_class import CLT
import time
import sys
import copy
import utilM


class dummy_node:
    
    def __init__(self,size =2):
        self.size = size
        
    
    def computeLL(self, dataset):
        if dataset.shape[0] <= 1:
            return pow(0.5, self.size)
        else:
            return dataset.shape[0] * pow(0.5, self.size)

# The internal node in the cutset network
class CNode:


    def __init__(self, var, weights, ids, id):
        
        self.var = var  # the variable id
        #self.var_assign = 0  # the assignment of var in map tuple
        self.children = []    # only has 2 child
        self.weights = weights 
        #print ('self.weights :', self.weights)
        #self.log_weights = np.log(weights)
        #self.log_inst_weight = np.array([])  # when doing instantiation
        #self.log_value = utilM.LOG_ZERO
        self.value = 0.0
        self.ids = ids
        self.id = id
        
        

    def add_child(self, child):

        self.children.append(child)


    def set_weights(self, weights):
                
        self.weights = weights
        #self.log_weights = np.log(self.weights)
    
    
    def sumout(self, child_value, weights):
        
        #self.log_val = np.logaddexp(self.children[0].log_value + log_weights[0],
                                    #self.children[1].log_value + log_weights[1])
        #self.value = child_value[0] * self.weights[0] + child_value[1] * self.weights[1]
        self.value = np.matmul(child_value, weights)
        
        
#    def maxout(self, log_weights):
#        
#        left = self.children[0].log_value + log_weights[0]
#        right = self.children[1].log_value + log_weights[1]
#        
#        if left >= right:
#            self.log_value = left
#            self.var_assign = 0
#        else:
#            self.log_value = right
#            self.var_assign = 1
            
            
        
        #self.log_value = max(self.children[0].log_value + log_weights[0],
        #                            self.children[1].log_value + log_weights[1])
        #print ('cnet nodes: ', self.var)
        #print ('child_log value:' , self.children[0].log_value,  self.children[1].log_value)
        #print ('log weights: ', log_weights)
        #print ( 'log_val:', self.log_value)
        #print (self.children[0])
        #print (self.children[1])
    
    """    
    def instantiation(self, flag = False, log_weight = None):
        
        if flag == False:
            self.log_inst_weight = np.copy(self.log_weights)
        else:
            self.log_inst_weight = log_weight
    """       




# Code copied from Tahrima.
# Inefficient and will need to speed up one day
class CNET:
    def __init__(self,depth=100, min_rec=10, min_var=5):
        self.nvariables=0
        self.depth=depth
        self.tree=[]
        self.min_rec = min_rec
        self.min_var = min_var
        # for get node and edge potential
        self.internal_list = []
        self.internal_var_list = []
        self.leaf_list = []
        self.leaf_ids_list = []
        
    def learnStructureHelper(self,dataset,ids):
        curr_depth=self.nvariables-dataset.shape[1]
        #print ("curr_dept: ", curr_depth)
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset)
            #clt.get_log_cond_cpt()
            
            #print ("built from count")
            #print ("topo_order: ", clt.topo_order)
            #print ("parents: ", clt.parents)
            #print ("pxy: " )
            #for sszezeaei in xrange (clt.xyprob.shape[0]):
            #    print ("-------X = ", i)
            #    for j in xrange (clt.xyprob.shape[1]):
            #        print ("-------Y = ", j)
            #        print (clt.xyprob[i,j,:,:])
                    
            return clt
        xycounts = Util.compute_xycounts(dataset) + 1  # laplace correction
        xcounts = Util.compute_xcounts(dataset) + 2  # laplace correction
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_edge_weights(xycounts, xcounts)
        #np.fill_diagonal(edgemat, 0) #shasha#
        
        #print ("edgemat: ", edgemat)
        scores = np.sum(edgemat, axis=0)
        #print (scores)
        variable = np.argmax(scores)
        
        #print ("variable: ", ids[variable])
        
        new_dataset1=np.delete(dataset[dataset[:,variable]==1],variable,1)
        p1=float(new_dataset1.shape[0])+1.0
        new_ids=np.delete(ids,variable,0)
        
        #print ("new_ids: ", new_ids)
        new_dataset0 = np.delete(dataset[dataset[:, variable] == 0], variable, 1)
        p0 = float(new_dataset0.shape[0]) +1.0
        
        #print ("p0, p1:", float(p0)/(p0+p1), float(p1)/(p0+p1))
        return [variable,ids[variable],p0,p1,self.learnStructureHelper(new_dataset0,new_ids),
                self.learnStructureHelper(new_dataset1,new_ids)]
        
        
    def learnStructure(self, dataset):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)
        self.tree=self.learnStructureHelper(dataset,ids)
        
        
        #print(self.tree)
    def computeLL(self,dataset):
        prob = 0.0
        for i in range(dataset.shape[0]):
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    prob+=np.log(p1/(p0+p1))
                    node=node1
                else:
                    prob+=np.log(p0/(p0+p1))
                    node = node0
                #print ('a:', prob)
            #print ('ids:', ids)
            #print dataset[i:i+1,ids].shape
            #print node.topo_order
            #print node.parents
            #print node.log_cond_cpt
            prob+=node.computeLL(dataset[i:i+1,ids])
            #print ('a:', prob)
        return prob
    
    def computeLL_each_datapoint(self,dataset):
        probs = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0]):
            prob = 0.0
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    prob+=np.log(p1/(p0+p1))
                    node=node1
                else:
                    prob+=np.log(p0/(p0+p1))
                    node = node0
            prob+=node.computeLL(dataset[i:i+1,ids])
            probs[i] = prob
        return probs
    
    def update(self,dataset_, weights=np.array([])):
        if weights.shape[0]==dataset_.shape[0]:
            norm_weights = Util.normalize(weights)
            indices = np.argwhere(np.random.multinomial(dataset_.shape[0], norm_weights)).ravel()
            dataset = dataset_[indices, :]
        else:
            dataset=dataset_
        for i in range(dataset.shape[0]):
            node=self.tree
            ids=np.arange(self.nvariables)
            while isinstance(node,list):
                id,x,p0,p1,node0,node1=node
                p0_index=2
                p1_index=3
                assignx=dataset[i,x]
                ids=np.delete(ids,id,0)
                if assignx==1:
                    node[p1_index]=p1+1.0
                    node=node1
                else:
                    node[p0_index]=p0+1.0
                    node = node0
            node.update(dataset[i:i+1,ids])



    """
    ---------------------------------------------------------------------------
    - Shasha's code starts here
    ---------------------------------------------------------------------------    
    """
    # the dataset are weighted
    def learn_structure_weight(self, dataset, weights, ids, smooth):
        curr_depth=self.nvariables-dataset.shape[1]
        #print 'curr_depth: ', curr_depth
        #print 'ids:', ids
        
            
        
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset) 
            #print edgemat
            #print 'topo_order: ', clt.topo_order
            # set value to compute
            #clt.xyprob = Util.normalize2d(self.xycounts)
            #clt.xprob = Util.normalize1d(self.xcounts)  
            #clt.get_log_cond_cpt() 
            # reset to save memeroy
            clt.xyprob = np.zeros((1, 1, 2, 2))
            clt.xprob = np.zeros((1, 2))             
            return clt
        
        
        self.xycounts = Util.compute_weighted_xycounts(dataset, weights) + smooth
        self.xcounts = Util.compute_weighted_xcounts(dataset, weights) + 2.0 *smooth
        edgemat = Util.compute_edge_weights(self.xycounts, self.xcounts)
        
        #edgemat[edgemat == 0.0] = 1e-20
        np.fill_diagonal(edgemat, 0)
        
#        if dataset.shape[0] == 0:
#            print 'here'
#            dummy = dummy_node(ids.shape[0])
#            return dummy
        #print ("edgemat: ", edgemat)
        scores = np.sum(edgemat, axis=0)
        #print (scores)
        variable = np.argmax(scores)
        
        #print ("variable: ", ids[variable])
        
        index1 = np.where(dataset[:,variable]==1)[0]
        index0 = np.where(dataset[:,variable]==0)[0]
        #index0 = np.setdiff1d(np.arange(dataset.shape[0]), index1)
        
        new_dataset =  np.delete(dataset, variable, axis = 1)
        
        new_dataset1 = new_dataset[index1]
        new_weights1 = weights[index1]
        p1= np.sum(new_weights1)+smooth
                
        #print ("new_ids: ", new_ids)
        new_dataset0 = new_dataset[index0]
        new_weights0 = weights[index0]
        p0 = np.sum(new_weights0)+smooth
        
        # Normalize
        p0 = p0/(p0+p1)
        p1 = 1.0 - p0
        
        #print p0, p1
        
        new_ids=np.delete(ids,variable,0)
        
        #print ("p0, p1:", float(p0)/(p0+p1), float(p1)/(p0+p1))
        return [variable,ids[variable],p0,p1,self.learn_structure_weight(new_dataset0,new_weights0,new_ids, smooth),
                self.learn_structure_weight(new_dataset1,new_weights1, new_ids, smooth)]
    
    
    def update_parameter(self, node, dataset, weights, ids, smooth):
        
        if dataset.shape[0] == 0:
            return
        
        # internal nodes, not reach the leaf
        if isinstance(node,list):
            id,x,p0,p1,node0,node1 = node
            index1 = np.where(dataset[:,x]==1)[0]
            index0 = np.where(dataset[:,x]==0)[0]
            
            #new_dataset =  np.delete(dataset, variable, axis = 1)
            
            new_weights1 = weights[index1]
            new_weights0 = weights[index0]
            new_dataset1 = dataset[index1]
            new_dataset0 = dataset[index0]
            
            p1 = np.sum(new_weights1) + smooth
            p0 = np.sum(new_weights0) + smooth
            
            # Normalize
            p0 = p0/(p0+p1)
            p1 = 1.0 - p0
            
            
            node[2] = p0
            node[3] = p1
            
            new_ids = np.delete(ids, id)
            
            self.update_parameter(node0, new_dataset0, new_weights0, new_ids, smooth)
            self.update_parameter(node1, new_dataset1, new_weights1, new_ids, smooth)
        
        #elif isinstance(node, CLT):
        else:
            clt_dataset = dataset[:, ids]
            node.update_exact(clt_dataset, weights, structure_update_flag = False)
            return
#        # dummy node
#        else:
#            return
            
            

        

    '''
        Update the CNet using weighted samples, exact update
    '''
    def update_exact(self, dataset, weights, structure_update_flag = False):
        
        if dataset.shape[0] != weights.shape[0]:
            print ('ERROR: weight size not equal to dataset size!!!')
            exit()
        # Perform based on weights
        # assume that dataset_.shape[0] equals weights.shape[0] because each example has a weight
        # try to avoid sum(weights = 0
        smooth = max(np.sum(weights), 1.0) / dataset.shape[0]
        ids = np.arange(dataset.shape[1])
        self.nvariables = dataset.shape[1]
       
        
        if structure_update_flag == True:
            # update the structure as well as parameters
            self.tree = self.learn_structure_weight(dataset, weights, ids, smooth)
        else:
            # only update parameters
            node=self.tree
            self.update_parameter(node, dataset, weights, ids,smooth)
            
        
    def get_prob_each(self, node, samples, row_index, ids, probs):
        
        
        
        if isinstance(node,list):
            #print ('*** in internal nodes ***')
            id,x,p0,p1,node0,node1=node
            p0 = p0 / float(p0+p1)
            p1 = 1.0 - p0
            #print ('x: ', x)
            #print p0, p1
            
            index1 = np.where(samples[:,id]==1)[0]
            index0 = np.where(samples[:,id]==0)[0]
            #print 'index1: ', index1
            #print 'index0: ', index0
            
            row_index1 = row_index[index1]
            row_index0 = row_index[index0]
            
            probs[row_index1] += np.log(p1)
            probs[row_index0] += np.log(p0)
            #print 'probs: ', probs
            
            #new_dataset =  np.delete(dataset, variable, axis = 1)
            
            new_samples =  np.delete(samples, id, axis = 1)
            new_samples1 = new_samples[index1]
            new_samples0 = new_samples[index0]
            
            new_ids = np.delete(ids, id)
            
            if new_samples0.shape[0] > 0:
                self.get_prob_each(node0, new_samples0, row_index0, new_ids, probs)
            if new_samples1.shape[0] > 0:
                self.get_prob_each(node1, new_samples1, row_index1, new_ids, probs)
        
        # leaf node
        else:
            #print ('reach leaf')
            clt_prob = node.getWeights (samples)
            probs[row_index] += clt_prob
            
 
    def getWeights(self, dataset):
        
        probs = np.zeros(dataset.shape[0])
        row_index = np.arange(dataset.shape[0])
        ids=np.arange(self.nvariables)
        node=self.tree
        
        self.get_prob_each(node, dataset, row_index, ids, probs)
        return probs
        
    
    
    '''
    For bags of CNet
    '''
    def learnStructureP_Helper(self,dataset,ids, portion):
        curr_depth=self.nvariables-dataset.shape[1]
        #print ("curr_dept: ", curr_depth)
        if dataset.shape[0]<self.min_rec or dataset.shape[1]<self.min_var or curr_depth >= self.depth:
            clt=CLT()
            clt.learnStructure(dataset)
            #clt.get_log_cond_cpt()
            
            #print ("built from count")
            #print ("topo_order: ", clt.topo_order)
            #print ("parents: ", clt.parents)
            #print ("pxy: " )
            #for sszezeaei in xrange (clt.xyprob.shape[0]):
            #    print ("-------X = ", i)
            #    for j in xrange (clt.xyprob.shape[1]):
            #        print ("-------Y = ", j)
            #        print (clt.xyprob[i,j,:,:])
                    
            return clt
        xycounts = Util.compute_xycounts(dataset) + 1  # laplace correction
        xcounts = Util.compute_xcounts(dataset) + 2  # laplace correction
        # compute mutual information score for all pairs of variables
        # weights are multiplied by -1.0 because we compute the minimum spanning tree
        edgemat = Util.compute_edge_weights(xycounts, xcounts)
        #np.fill_diagonal(edgemat, 0) #shasha#
        
        #print ("edgemat: ", edgemat)
        scores = np.sum(edgemat, axis=0)
        #print (scores)
        ind_portion = np.random.choice(ids.shape[0], int(ids.shape[0] * portion), replace=False )
        #print 'ind_portion: ', ind_portion
        scores_portion = scores[ind_portion]
        #print 'scores_portion: ', scores_portion
        
        #print np.argmax(scores_portion)
        variable = ind_portion[np.argmax(scores_portion)]
        #print 'variable: ', variable
        #print 'ids: ', ids
        
        #print ("variable: ", ids[variable])
        
        new_dataset1=np.delete(dataset[dataset[:,variable]==1],variable,1)
        p1=float(new_dataset1.shape[0])+1.0
        new_ids=np.delete(ids,variable,0)
        
        #print ("new_ids: ", new_ids)
        new_dataset0 = np.delete(dataset[dataset[:, variable] == 0], variable, 1)
        p0 = float(new_dataset0.shape[0]) +1.0
        
        #print ("p0, p1:", float(p0)/(p0+p1), float(p1)/(p0+p1))
        return [variable,ids[variable],p0,p1,self.learnStructureP_Helper(new_dataset0,new_ids, portion),
                self.learnStructureP_Helper(new_dataset1,new_ids, portion)]
        
        
    def learnStructure_portion(self, dataset,portion_percent):
        self.nvariables = dataset.shape[1]
        ids=np.arange(self.nvariables)
        self.tree=self.learnStructureP_Helper(dataset,ids, portion_percent)
        
    
    
    def get_node_list(self, node, ids):

        
        if isinstance(node,list):
            id,x,p0,p1,node0,node1=node
            weights = np.array([p0,p1])
            weights /= float(p1+p0)
            
            cnode = CNode(x, weights, ids, id)
            cnode.add_child(node0)
            cnode.add_child(node1)
            self.internal_list.append(node)
            self.internal_var_list.append(x)
            
            ids=np.delete(ids,id,0)
            self.get_node_list(node0, ids)
            self.get_node_list(node1, ids)
        else:
            self.leaf_list.append(node)
            self.leaf_ids_list.append(ids)
            return
                
    
    def instantiation(self, evid_list):
        
        dupe_internal_list = copy.deepcopy(self.internal_list)
        dupe_leaf_list = copy.deepcopy(self.leaf_list)
        
        # for internal nodes:
        for i in xrange(len(evid_list)):
            evid_var = evid_list[i][0]
            evid_value = evid_list[i][1]
            
            
            ind_cn = np.where(self.internal_var_list == evid_var)[0]            
            for j in ind_cn:
                if evid_value == 0:
                    dupe_internal_list[j].weights[1] =0
                else:
                    dupe_internal_list[j].weights[0] =0
                    
        # for leaf nodes: 
        evid_arr = np.asarray(evid_list)
        evid_arr[evid_arr[:,0].argsort()] # sort based on the first column
        for k, ln in enumerate(dupe_leaf_list):
            comm_var = np.intersect1d(self.leaf_ids_list[k],evid_arr[:,0])
            # find the projected varible in leaf clt tree
            # the varible shown in leaf clt tree
            proj_var = np.searchsorted(self.leaf_ids_list[k],comm_var)
            evid_ind = np.searchsorted(evid_arr[:,0],comm_var)
            ln_evid_list = np.copy(evid_arr[evid_ind])
            ln_evid_list[:,0] = proj_var
            ln.cond_cpt = ln.instantiation(list(ln_evid_list))
        
        return dupe_internal_list, dupe_leaf_list
        

    def get_node_marginal(self, evid_list, var):
        if len(self.internal_list) == 0 or len(self.leaf_list) == 0:
            self.get_node_list(self.tree, np.arange(self.nvariables))
            self.internal_var_list = np.asarray(self.internal_var_list)
        
        #evid_arr = np.asarray(evid_list)
        # Set evidence
        if len(evid_list) > 0:
            dupe_internal_list, dupe_leaf_list = self.instantiation( evid_list)
        else:
            dupe_internal_list = self.internal_list
            dupe_leaf_list = self.leaf_list
            
        for k, ln in enumerate(dupe_leaf_list):
            ln.xprob = 0
            curr_ids = self.leaf_ids_list[k]
            if var in curr_ids:
                ln.xprob = utilM.get_var_prob (ln.topo_order, ln.parents, ln.cond_cpt, np.where(curr_ids ==var)[0[0]])
            else:
                ln.xprob = np.array([utilM.ve_tree_bin(ln.topo_order, ln.parents, ln.cond_cpt)])
         
        node_marginal = np.zeros(2)
        # var = 0
        for i in xrange(len(dupe_internal_list), -1, -1):
            cn = dupe_internal_list[i]
            cn.value = 0.0
            weights = np.copy(cn.weights)
            if var == cn.var:
                weights[1] = 0
            child_val = np.zeros(2)
            if isinstance(cn.children[0], CNode):
                child_val[0] = cn.children[0].value
            elif isinstance(cn.children[0], CLT):
                child_val[0] = cn.children[0].xprob[0]
            cn.sumout(child_val, weights)
        
        node_marginal[0] = dupe_internal_list[0].value
        
        # var = 1
        for i in xrange(len(dupe_internal_list), -1, -1):
            cn = dupe_internal_list[i]
            cn.value = 0.0
            weights = np.copy(cn.weights)
            if var == cn.var:
                weights[0] = 0
            child_val = np.zeros(2)
            if isinstance(cn.children[0], CNode):
                child_val[0] = cn.children[0].value
            elif isinstance(cn.children[0], CLT):
                if cn.children[0].xprob.shape[0] == 2:
                    child_val[0] = cn.children[0].xprob[1]
                else:
                    child_val[0] = cn.children[0].xprob[0]
            cn.sumout(child_val, weights)
        
        node_marginal[1] = dupe_internal_list[0].value
        
        return node_marginal
                    
                
                
                    
            

#def main_cutset():
#    
#    dataset_dir = sys.argv[2]
#    data_name = sys.argv[4]
#    min_depth = int(sys.argv[6])
#    max_depth = int(sys.argv[8])
#    
#    #dataset_dir = '/Users/shashajin/Desktop/TIM/dataset/'
#    #data_name = 'nltcs'
#    
#    #train_filename = sys.argv[1]
#    train_filename = dataset_dir + data_name + '.ts.data'
#    test_filename = dataset_dir + data_name +'.test.data'
#    valid_filename = dataset_dir + data_name + '.valid.data'
#    
#    #train_filename = sys.argv[1]
#    #train_filename = '/Users/shashajin/Desktop/TIM/dataset/nltcs.ts.data'
#    #test_filename = train_filename[:-8] + '.test.data'
#    #valid_filename = train_filename[:-8] + '.valid.data'
#    
#    #dataset_dir = sys.argv[2]
#    #data_name = sys.argv[4]
#    
#    
#
#
#    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
#    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
#    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
#    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
#    
#    #n_variables = train_dataset.shape[1]
#    
#    
#    print("Learning Chow-Liu Trees on original data ......")
#    clt = CLT()
#    clt.learnStructure(train_dataset)
#    
#    
#    print('Train set LL scores')
#    print(clt.computeLL(train_dataset) / train_dataset.shape[0], "Chow-Liu")
#    print('Valid set LL scores')
#    print(clt.computeLL(valid_dataset) / valid_dataset.shape[0], "Chow-Liu")
#    print('Test set LL scores')
#    print(clt.computeLL(test_dataset) / test_dataset.shape[0], "Chow-Liu")
#    
#    
#    
#    """
#    cnet
#    """
#    #cnets = []
#    print("Learning Cutset Networks only Training data.....")
#    #max_depth = min(train_dataset.shape[1], 20) +1
#    #max_depth += 1
#    train_ll = np.zeros(max_depth)
#    valid_ll = np.zeros(max_depth)
#    test_ll = np.zeros(max_depth)
#    
#    train_gw = np.zeros(max_depth)
#    valid_gw = np.zeros(max_depth)
#    test_gw = np.zeros(max_depth)
#    
#    for i in range(min_depth, max_depth+1):
#    #for i in range(5, 6):
#        cnet = CNET(depth=i)
#        cnet.learnStructure(train_dataset)
#        #cnet.update_exact(train_dataset, np.random.random_sample((train_dataset.shape[0],)), structure_update_flag = True)
#        #cnet.update_exact(train_dataset, np.random.random_sample((train_dataset.shape[0],)), structure_update_flag = False)
#        #cnet.update_exact(train_dataset, np.ones(train_dataset.shape[0]), structure_update_flag = False)
#        #cnet.learnStructure(train_dataset)
##        temp_dataset = train_dataset[0:10]
##        print temp_dataset
##        
##        corr_ll = cnet.computeLL(temp_dataset) / temp_dataset.shape[0]
##        curr_ll = np.sum(cnet.getWeights(temp_dataset)) / temp_dataset.shape[0]
##        print 'correct_ll: ', corr_ll
##        print 'current_ll: ', curr_ll
##        ssss
#        
#        start = time.time()
#        train_ll[i-1] = cnet.computeLL(train_dataset) / train_dataset.shape[0]
#        valid_ll[i-1] = cnet.computeLL(valid_dataset) / valid_dataset.shape[0]
#        test_ll[i-1] = cnet.computeLL(test_dataset) / test_dataset.shape[0]
#        print 'running time for orignal: ', time.time()-start
#        
#        start2 = time.time()
#        train_gw[i-1] =  np.sum(cnet.getWeights(train_dataset)) / train_dataset.shape[0]
#        valid_gw[i-1] =  np.sum(cnet.getWeights(valid_dataset)) / valid_dataset.shape[0]
#        test_gw[i-1]  =  np.sum(cnet.getWeights(test_dataset))  / test_dataset.shape[0]
#        print 'running time for new: ', time.time()-start2
#        
#    print("done")
#    
#    print('Train set cnet LL scores')
#    for l in xrange(max_depth):
#        print (train_ll[l], l+1)
#    print()
#    
#    print('Valid set cnet LL scores')
#    for l in xrange(max_depth):
#        print (valid_ll[l], l+1)
#    print()   
#    
#    print('test set cnet LL scores')
#    for l in xrange(max_depth):
#        print (test_ll[l], l+1)
#        
#        
#    print ('--------------get weights-------------')
#    print('Train set cnet LL scores')
#    for l in xrange(max_depth):
#        print (train_gw[l], l+1)
#    print()
#    
#    print('Valid set cnet LL scores')
#    for l in xrange(max_depth):
#        print (valid_gw[l], l+1)
#    print()   
#    
#    print('test set cnet LL scores')
#    for l in xrange(max_depth):
#        print (test_gw[l], l+1)
#
##    save_dict= {}
##    save_dict['cnet'] = test_ll
##    np.save('../plots/' + data_name, save_dict)
    
    
def main_cutset():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    depth = int(sys.argv[6])

    
    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    

    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    
    #n_variables = train_dataset.shape[1]
    
    
#    print("Learning Chow-Liu Trees on original data ......")
#    clt = CLT()
#    clt.learnStructure(train_dataset)
#    
#    
#    print('Train set LL scores')
#    print(clt.computeLL(train_dataset) / train_dataset.shape[0], "Chow-Liu")
#    print('Valid set LL scores')
#    print(clt.computeLL(valid_dataset) / valid_dataset.shape[0], "Chow-Liu")
#    print('Test set LL scores')
#    print(clt.computeLL(test_dataset) / test_dataset.shape[0], "Chow-Liu")
    
    
    
    """
    cnet
    """
    #cnets = []
    print("Learning Cutset Networks only Training data.....")


    cnet = CNET(depth=depth)
    cnet.learnStructure(train_dataset)
    
#    train_ll = cnet.computeLL(train_dataset) / train_dataset.shape[0]
#    valid_ll = cnet.computeLL(valid_dataset) / valid_dataset.shape[0]
#    test_ll = cnet.computeLL(test_dataset) / test_dataset.shape[0]

    train_ll =  np.sum(cnet.getWeights(train_dataset)) / train_dataset.shape[0]
    valid_ll =  np.sum(cnet.getWeights(valid_dataset)) / valid_dataset.shape[0]
    test_ll  =  np.sum(cnet.getWeights(test_dataset))  / test_dataset.shape[0]

    print train_ll
    print valid_ll
    print test_ll
    


def main_cutset_mult():
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    max_depth = int(sys.argv[6])
    
    #dataset_dir = '/Users/shashajin/Desktop/TIM/dataset/'
    #data_name = 'nltcs'
    
    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    #train_filename = sys.argv[1]
    #train_filename = '/Users/shashajin/Desktop/TIM/dataset/nltcs.ts.data'
    #test_filename = train_filename[:-8] + '.test.data'
    #valid_filename = train_filename[:-8] + '.valid.data'
    
    #dataset_dir = sys.argv[2]
    #data_name = sys.argv[4]
    
    


    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    #train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    
    #n_variables = train_dataset.shape[1]
    
    

    
    """
    cnet
    """
    #cnets = []
    print("Learning Cutset Networks only Training data.....")
    #max_depth = min(train_dataset.shape[1], 20) +1
    train_ll = np.zeros(max_depth)
    valid_ll = np.zeros(max_depth)
    test_ll = np.zeros(max_depth)
    #cnet_list =[]
    best_valid = -np.inf
    best_module = None
    for i in range(1, max_depth+1):
    #for i in range(5, 6):
        cnet = CNET(depth=i)
        cnet.learnStructure(train_dataset)
        train_ll[i-1] = np.sum(cnet.getWeights(train_dataset)) / train_dataset.shape[0]
        valid_ll[i-1] = np.sum(cnet.getWeights(valid_dataset)) / valid_dataset.shape[0]
        test_ll[i-1] = np.sum(cnet.getWeights(test_dataset))  / test_dataset.shape[0]
        
        if best_valid < valid_ll[i-1]:
            best_valid = valid_ll[i-1]
            best_module = copy.deepcopy(cnet)
            
        #cnet_list.append(cnet)
        
       

    print("done")
    
    print('Train set cnet LL scores')
    for l in xrange(max_depth):
        print (train_ll[l], l+1)
    print()
    
    print('Valid set cnet LL scores')
    for l in xrange(max_depth):
        print (valid_ll[l], l+1)
    print()   
    
    print('test set cnet LL scores')
    for l in xrange(max_depth):
        print (test_ll[l], l+1)
        
    best_ind = np.argmax(valid_ll)
    #best_module = cnet_list[best_ind]
    
    print ('ll score for best', best_ind )
    
    print( np.sum(best_module.getWeights(train_dataset)) / train_dataset.shape[0])
    print( np.sum(best_module.getWeights(valid_dataset)) / valid_dataset.shape[0])
    print( np.sum(best_module.getWeights(test_dataset)) / test_dataset.shape[0])
    
    main_dict = {}
    utilM.save_cutset(main_dict, best_module.tree, np.arange(train_dataset.shape[1]), ccpt_flag = True)
    np.savez_compressed('../best_module_data/' + data_name, module = main_dict)

    

            
if __name__=="__main__":
    #main_cutset()
    #main_clt()
    start = time.time()
    #main_cutset()
    main_cutset_mult()
    print ('Total running time: ', time.time() - start)
            
            
            
            
            
            
            