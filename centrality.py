import networkx as nx
import json
import os
from networkx.readwrite import json_graph
from load_dataset import get_dataset_sample_ids

def get_degree_centrality(dataset_dir, news_source):
    #print("working in degree centrality caluation function ")
    
    labels = ["fake","real"]
    fake_real_centrality = []
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        for sample_id in get_dataset_sample_ids(news_source, news_label, "/content/FakeNewsPropagation/data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.degree_centrality(new_graph)
                
                #print("degree centrality calculated for current graph ")
                
                for x in deg_centrality:
                  fake_real_centrality.append(deg_centrality[x])
                  break
    return fake_real_centrality

def get_closness_centrality(dataset_dir, news_source):
    labels = ["fake","real"]
    fake_real_centrality = []
    
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        
        for sample_id in get_dataset_sample_ids(news_source, news_label, "/content/FakeNewsPropagation/data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.closeness_centrality(new_graph)
                for x in deg_centrality:
                  fake_real_centrality.append(deg_centrality[x])
                  break
    return fake_real_centrality

def get_betweenness_centrality(dataset_dir, news_source):
    labels = ["fake","real"]
    fake_real_centrality = []
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        
        for sample_id in get_dataset_sample_ids(news_source, news_label, "/content/FakeNewsPropagation/data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.betweenness_centrality(new_graph)
                for x in deg_centrality:
                  fake_real_centrality.append(deg_centrality[x])
                  break
                    
                    
    return fake_real_centrality


def get_pagerank(dataset_dir, news_source):
    labels = ["fake","real"]
    
    fake_real_centrality = []
    
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        
        for sample_id in get_dataset_sample_ids(news_source, news_label, "/content/FakeNewsPropagation/data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.pagerank(new_graph,alpha = 0.8)
                
                for x in deg_centrality:
                  fake_real_centrality.append(deg_centrality[x])
                  break
    return fake_real_centrality






