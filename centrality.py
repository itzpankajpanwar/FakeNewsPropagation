import networkx as nx
from load_dataset import get_dataset_sample_ids

def get_degree_centrality(data_dir, news_source):
    labels = ["fake","real"]
    fake_real_centrality = []
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        for sample_id in get_dataset_sample_ids(news_source, news_label, "data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.deg_centrality(new_graph)
                fake_real_centrality.append()
    return fake_real_centrality

def get_closness_centrality(data_dir, news_source):
    labels = ["fake","real"]
    fake_real_centrality = []
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        for sample_id in get_dataset_sample_ids(news_source, news_label, "data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.closeness_centrality(new_graph)
                fake_real_centrality.append()
    return fake_real_centrality

def get_betweenness_centrality(data_dir, news_source):
    labels = ["fake","real"]
    fake_real_centrality = []
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        for sample_id in get_dataset_sample_ids(news_source, news_label, "data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                new_graph = json_graph.tree_graph(json.load(file))
                deg_centrality = nx.betweenness_centrality(new_graph)
                fake_real_centrality.append()
    return fake_real_centrality



