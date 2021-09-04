import networkx as nx
import pandas as pd
import numpy as np
import random
import pprint
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')


def _get_random_walk_from_node(g: nx.Graph, node: nx.Graph.nodes, path_length: int) -> []:
    random_walk = [node]
    for i in range(path_length - 1):
        temp = list(g.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
    return random_walk


def get_random_walks_from_all_nodes(g: nx.Graph, path_length: int) -> [[]]:
    all_nodes = list(g.nodes())
    random_walks = []
    for n in all_nodes:
        random_walks.append(_get_random_walk_from_node(g=g, node=n, path_length=path_length))
    return random_walks


def train_skip_gram_model(random_walks: [], progress_per: int):
    model = Word2Vec(window=4, sg=1, hs=0, negative=10, alpha=0.01, min_alpha=0.001, seed=14)
    model.build_vocab(corpus_iterable=random_walks, progress_per=progress_per)
    model.train(corpus_iterable=random_walks, total_examples=model.corpus_count, epochs=20, report_delay=1)
    return model


def run():
    df = pd.read_csv("./data/seealsology-data.tsv", sep="\t")
    g = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())
    pprint.pprint("Number of Nodes: {}".format(len(g)))
    random_walks = get_random_walks_from_all_nodes(g=g, path_length=5)
    model = train_skip_gram_model(random_walks=random_walks, progress_per=2)
    pprint.pprint('Most similar page to {} is {}'.format('google assistant', model.wv.similar_by_word('google assistant')))

if __name__ == "__main__":
    run()
