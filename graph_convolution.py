import networkx as nx
from load_dataset import get_dataset_sample_ids

def get_one_big_graph(data_dir, news_source):
    itr = 0
    labels = ["fake","real"]
    fake_real_centrality = []
    for  news_label in labels:
        news_dataset_dir = "{}/{}_{}".format(dataset_dir, news_source, news_label)
        for sample_id in get_dataset_sample_ids(news_source, news_label, "data/sample_ids"):
            with open("{}/{}.json".format(news_dataset_dir, sample_id)) as file:
                if itr = 0:
                    f = json_graph.tree_graph(json.load(file))
                    ct=1
                else:
                    tt = json_graph.tree_graph(json.load(file))
                    f = nx.compos(f , tt)
    return f

                
      
