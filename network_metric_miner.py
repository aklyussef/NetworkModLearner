import sys

import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt


from network_plugin import read_file,get_edge_tuples
from IOHelper import IOHelper

#Returns the average of dictionary values
def get_avg(dict):
    return sum(dict.values())/len(dict)

def get_avg_degree(G):
    #TODO: add support for directed/undirected
    deg = 0
    nnodes = nx.number_of_nodes(G)
    if G.is_directed():
        deg = sum(d for n, d in G.in_degree()) / float(nnodes)
        deg = sum(d for n, d in G.out_degree()) / float(nnodes)
    else:
        deg = sum(dict(G.degree()).values())/float(nnodes)
    return deg

def compute_metrics(G):
    metrics = {}
    metrics['edge_node_r']     = nx.number_of_edges(G)/nx.number_of_nodes(G)
    metrics['avg_clustering']           = nx.average_clustering(G)
    metrics['avg_mdc']                  = get_avg(nx.average_degree_connectivity(G))
    metrics['avg_degree']               = get_avg_degree(G)
    metrics['transitivity']             = nx.transitivity(G)
    metrics['density']                  = nx.density(G)
    return metrics

# compute modularity
def compute_greedy_modularity_community_metrics(G):
    #starts with every node being a community and combines nodes to maximize modularity, stops when modularity no longer increases
    modularity = 0
    greedy_community = list(community.greedy_modularity_communities(G))
    print('There were {} communities found.'.format(len(greedy_community)))
    modularity = community.modularity(G,greedy_community)
    print('Graph modularity based on Clauset-Newman-Moore greedy modularity maximization clustering is: {}'.format(modularity))
    return modularity

# the more I apply next the more new communities appear
def compute_girvan_newman_community_metrics(G):
    #Depends on removing links between links of high vertex betweeness and identifying clusters
    modularity = 0
    communities = community.girvan_newman(G)
    top_level_communities = next(communities)
    next_level_communities = next(communities)
    modularity += community.modularity(G,next_level_communities)
    print('Graph modularity based on Girvan Newmann clustering is {}'.format(modularity))
    return modularity

def print_metrics(metrics):
    for (key,value) in metrics.items():
        print('{} \t {}'.format(key,value))

def print_network_characteristics(G):
    stars = '*'*10
    print(stars + '\tNetwork Characteristics\t' + stars)
    print(nx.info(G))
    print('network density {}'.format(nx.density(G)))
    print(stars + '\t'+ stars + '\t'+ stars)

filepath = r"/Users/aklyussef/Google Drive/School/Grad/Courses/Semester2/Optimization/Project/collab_network/data/jazz.net"

if(len(sys.argv) != 2):
    print('USAGE: {} PATH_TO_NETWORK_FILES'.format(sys.argv[0]))
    exit(1)

network_dir = sys.argv[1]

io_h = IOHelper(network_dir)
io_h.writeOutputHeader('filename,edges_node_r,avg_clustering,avg_mdc,avg_degree,transitivity,density,modularity')
for file in io_h.get_files_in_dir(network_dir):
    print('reading file {}'.format(file))
#   G = nx.read_weighted_edgelist(file)
    f = read_file(file)
    el = get_edge_tuples(f)
    G = nx.Graph()
    G.add_edges_from(el)
    print_network_characteristics(G)
    m = compute_metrics(G)
    print_metrics(m)
    #mod_gn          = compute_girvan_newman_community_metrics(G)
    mod_greedy      = compute_greedy_modularity_community_metrics(G)
    line = [file, str(m['edge_node_r']), str(m['avg_clustering']), str(m['avg_mdc']), str(m['avg_degree']),
                         str(m['transitivity']), str(m['density']),str(mod_gn)]
    io_h.write_out_line(','.join(line))
print('script finished!')
