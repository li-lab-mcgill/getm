import networkx as nx
from node2vec import Node2Vec
import numpy as np
import argparse

args = argparse.ArgumentParser(description='Node2Vec')
args.add_argument('--graph_file', type=str, help='directory to load .txt file containing graph information')
args.add_argument('--save_file', type=str, help='directory to save embeddings in txt files')
args.add_argument('--node_name', type=str, help='The type of nodes, such as medication, etc.')
args.add_argument('--walk_length', type=int, default=30)
args.add_argument('--num_walks', type=int, default=10)
args.add_argument('--workers', type=int, default=4)
args.add_argument('--window', type=int, default=10)
args.add_argument('--min_count', type=int, default=4)
args.add_argument('--batch_words', type=int, default=4)
args.add_argument('--dimensions', type=int, default=128)

def get_node_embed(graph, save_file, dimensions, walk_length, num_walks, workers,\
                 window, min_count, batch_words, node_name):
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, \
                        num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    model.wv.save_word2vec_format(f"{save_file}/{node_name}.txt")

def get_emb_arr(save_file):
    file = open(f"{save_file}","r+")
    s = file.readline()
    n, dim = list(map(int, s.split()))
    embs = {}
    for _ in range(1, n+1):
        s = file.readline()
        nums = list(map(float, s.split()))
        node_id = nums[0]
        emb = np.array(nums[1:])
        embs[node_id] = emb
    file.close()

    return embs

def run_node2vec(args):
    G = nx.read_edgelist(args.graph_file, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G = G.to_undirected()
    get_node_embed(G, args.save_file, args.dimensions, args.walk_length, args.num_walks, args.workers,\
                 args.window, args.min_count, args.batch_words, args.node_name)
    node_embs = get_emb_arr(args.save_file)
    np.save(f"{save_file}/{node_name}.npy", node_embs)





