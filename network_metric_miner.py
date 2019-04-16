import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt

#Returns the average of dictionary values
def get_avg(dict):
    return sum(dict.values())/len(dict)

def compute_metrics(G):
    metrics = {}
    metrics['edges_to_nodes_ratio']     = nx.number_of_edges(G)/nx.number_of_nodes(G)
    metrics['avg_clustering']           = nx.average_clustering(G)
    metrics['avg_degree']               = get_avg(nx.average_degree_connectivity(G))
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

# the more I apply next the more new communities appear
def compute_girvan_newman_community_metrics(G):
    #Depends on removing links between links of high vertex betweeness and identifying clusters
    modularity = 0
    communities = community.girvan_newman(G)
    top_level_communities = next(communities)
    modularity += community.modularity(G,top_level_communities)
    print('Graph modularity based on Girvan Newmann clustering is {}'.format(modularity))

def print_metrics(metrics):
    for (key,value) in metrics.items():
        print('{} \t {}'.format(key,value))

def print_network_characteristics(G):
    stars = '*'*10
    print(stars + '\tNetwork Characteristics\t' + stars)
    print(nx.info(G))
    print('Number of Nodes: {}'.format(nx.number_of_nodes(G)))
    print('Number of Edges: {}'.format(nx.number_of_edges(G)))
    print('network density {}'.format(nx.density(G)))

filepath = r"/Users/aklyussef/Google Drive/School/Grad/Courses/Semester2/Optimization/Project/collab_network/data/communication_network/email-Eu-core-temporal-Dept1.txt"
#filepath = r"/Users/aklyussef/Google Drive/School/Grad/Courses/Semester2/Optimization/Project/collab_network/data/university_astrophysics/ca-AstroPh.txt"
# filepath = r"/Users/aklyussef/Google Drive/School/Grad/Courses/Semester2/Optimization/Project/collab_network/data/CollegeMsg.txt.gz"
G = nx.read_weighted_edgelist(filepath)

print_network_characteristics(G)
print_metrics(compute_metrics(G))
compute_girvan_newman_community_metrics(G)
compute_greedy_modularity_community_metrics(G)
