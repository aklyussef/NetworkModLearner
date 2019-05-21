import sys

import networkx as nx
from networkx.algorithms import community
import numpy as np
import matplotlib.pyplot as plt
import community as louvain_community


from network_plugin import read_file,get_edge_tuples
from IOHelper import IOHelper

#Returns the average of dictionary values
def get_avg(dict):
    return sum(dict.values())/len(dict)

def get_weighted_avg(dict):
    return sum(i[0]*i[1] for i in zip(dict.keys(),dict.values()))/len(dict)

def get_avg_degree(G):
    deg = 0
    nnodes = nx.number_of_nodes(G)
    if G.is_directed():
        deg = sum(d for n, d in G.in_degree()) / float(nnodes)
        deg = sum(d for n, d in G.out_degree()) / float(nnodes)
    else:
        deg = sum(dict(G.degree()).values())/float(nnodes)
    return deg

#TODO: Figure out how to compute a measure for average(avg degree conn)
def compute_metrics(G):
    metrics = {}
    metrics['e_n_r']                    = nx.number_of_edges(G)/nx.number_of_nodes(G) #Edge to node ratio
    metrics['av_clu']                   = nx.average_clustering(G)  #avg clustering
    metrics['av_mdc']                   = get_weighted_avg(nx.average_degree_connectivity(G))
    metrics['av_deg']                   = get_avg_degree(G) #avg_degree
    metrics['tran']                     = nx.transitivity(G) # transitivity
    metrics['den']                      = nx.density(G)
    metrics['c_cen']                    = get_avg(nx.closeness_centrality(G))
    metrics['b_cen']                    = get_avg(nx.betweenness_centrality(G))
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

def compute_louvain_community_metrics(G):
    #starts with every node being a community and combines nodes to maximize modularity, stops when modularity no longer increases
    modularity = 0
    part = louvain_community.best_partition(G) 
    print('There were {} communities found.'.format(len(set(part.values()))))
    modularity = louvain_community.modularity(part,G)
    print('Graph modularity based on the louvain method is: {}'.format(modularity))
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

# function to return label based on threshold 1 if mod > threshold else 0
def get_class(modularity,threshold):
    label = '1' if modularity >= threshold else '0'
    return label

def generate_networks(io_h):
    # Dictionary for storing networks with names for dataset
    list_of_networks    = {}
    n_nodes             = 1000
    n_edges             = 3
    connection_probs    = np.arange(0.3,1,0.1)
    multipliers         = np.arange(50,100,5)

    processed_networks  = io_h.get_ignore_list()

    print('Generating Famous social networks')
    # Append famous social networks to list
    n_name = 'kartae_club'
    if n_name not in processed_networks:
        list_of_networks[n_name]  = nx.karate_club_graph()
    n_name = 'davis_southern_women_graph'
    if n_name not in processed_networks:
        list_of_networks[n_name]  = nx.davis_southern_women_graph()
    n_name = 'florentine_families_graph'
    if n_name not in processed_networks:
        list_of_networks[n_name]   = nx.florentine_families_graph()

    print('Generating Barabasi network')
    # Append Barabasi Graph
    n_name = 'barabasi_albert_graph'
    if n_name not in processed_networks:
        list_of_networks[n_name]  = nx.barabasi_albert_graph(n_nodes,n_edges)

    print('Generating Random Networks')
    # Generate random networks with connection probabilities
    for con_prob in connection_probs:
        n_name = '_'.join(['erdos_renyi_graph', '{:.1f}'.format(con_prob)])
        if n_name not in processed_networks:
            print('Generating {}'.format(n_name))
            list_of_networks[n_name]       = nx.erdos_renyi_graph(n_nodes,con_prob)
        #gnm_random_graph requires number of edges so we will use prob * multiplier * number_of_nodes
        for multiplier in multipliers:
            #computed for each iteration
            m_factor =  multiplier*con_prob*n_nodes
            n_name = 'x'.join(['gnm_random_graph','{:.1f}'.format(m_factor)])
            if n_name not in processed_networks:
                print('Generating {}'.format(n_name))
                n_edges = int(m_factor)
                list_of_networks[n_name]        = nx.gnm_random_graph(n_nodes,n_edges)
    print('Finished Generating Networks')
    return list_of_networks

def print_metrics(metrics):
    for (key,value) in metrics.items():
        print('{} \t {}'.format(key,value))

def print_network_characteristics(G):
    stars = '*'*10
    print(stars + '\tNetwork Characteristics\t' + stars)
    print(nx.info(G))
    print('network density {}'.format(nx.density(G)))
    print(stars + '\t'+ stars + '\t'+ stars)

def process_network_files(networkdir,io_handler):
    for file in networkdir:
        print('reading file {}'.format(file))
        f = read_file(file)
        el = get_edge_tuples(f)
        G = nx.Graph()
        G.add_edges_from(el)
        mine_graph(file,G,io_handler)

def process_generated_networks(network_d,io_handler):
    for name,graph in network_d.items():
        mine_graph(name,graph,io_handler)

def mine_graph(name,G,io_handler):
    print('mining graph {}'.format(name))
    print_network_characteristics(G)
    if nx.number_of_nodes(G) == 0 or nx.number_of_edges(G) == 0:
        return
    m = compute_metrics(G)
    print_metrics(m)
    #mod = compute_greedy_modularity_community_metrics(G)
    mod = compute_louvain_community_metrics(G)
    #mod_class = get_class(mod_greedy, mod_threshold)
    line = [name, str(m['e_n_r']), str(m['av_clu']), str(m['av_mdc']), str(m['av_deg']),
            str(m['tran']), str(m['den']), str(m['c_cen']), str(m['b_cen']), str(mod)]
    io_handler.write_out_line(','.join(line))

filepath = r"/Users/aklyussef/Google Drive/School/Grad/Courses/Semester2/Optimization/Project/collab_network/data/jazz.net"

if(len(sys.argv) != 2):
    print('USAGE: {} PATH_TO_NETWORK_FILES'.format(sys.argv[0]))
    exit(1)

network_dir = sys.argv[1]
#Modularity threshold for class cutoff can be extended to be used inside for loop for different classifications
mod_threshold = 0.6

io_h = IOHelper(network_dir)
io_h.writeOutputHeader('filename,edges_node_r,avg_clustering,avg_mdc,avg_degree,transitivity,density,c_centrality,b_centrality,modularity')
networks = generate_networks(io_h)
process_generated_networks(networks,io_h)
process_network_files(io_h.get_files_in_dir(network_dir),io_h)

print('script finished!')
