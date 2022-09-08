# Import necessary libraries
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import numpy as np
import gensim
import random


# Create the NT2VEC class
class NT2VEC:
    prob_key = 'probabilities'
    probabilities_key = prob_key
    attr_prob_key = 'attr'
    first_travel_key = 'first_travel_key'
    probabilities_key = 'probabilities'
    neighbors_key = 'neighbors'
    weight_key = 'weight'
    num_walks_key = 'num_walks'
    walk_length_key = 'walk_length'
    p_key = 'p'
    q_key = 'q'

    def __init__(self, graph, attr, labels=None, dim=200, knn=10, workers=12, num_walks=100, walk_length=50,
                 sampling_strategy=None, weight_key='weight', sg=1, p=0.4, q=0.3, t=0.3):
        self.graph = graph  # networkX graph
        self.attr = attr  # node attributes
        self.dim = dim  # length of the output vectors
        self.knn = knn  # number of neighbors to use in node_attr
        self.p = p  # node2vec return parameter
        self.q = q  # node2vec inout parameter
        self.t = t  # parameter that controls where to sample walks (0 is fully network, 1 is no network)
        self.num_walks = num_walks  # number of walks sample
        self.walk_length = walk_length  # length of samples
        self.walks = list()
        self.weight_key = weight_key
        self.d_graph = defaultdict(dict)
        self.d_attr = defaultdict(dict)
        self.sg = sg
        self.labels = labels

        self.workers = workers

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

    def precompute_nearest_neighbors(self):

        # Create Nearest Neighbors model
        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='auto', metric='cosine').fit(self.attr)

        # Get distances and nearest neighbors
        distances, indices = nbrs.kneighbors()

        similarities = 1 - distances

        # Normalize similarities into probability
        for i in range(len(similarities)):
            sum_ = sum(similarities[i])
            if sum_ != 0:
                similarities[i] = similarities[i] / (sum(similarities[i]))

        return similarities, indices

    def precompute_attr_probabilities(self):

        d_attr = self.d_attr

        similarities, nn_indices = self.precompute_nearest_neighbors()

        for s in range(len(nn_indices)):

            if self.prob_key not in d_attr[s]:
                d_attr[s][self.prob_key] = dict()

            for i in range(len(nn_indices[int(s)])):
                node = nn_indices[int(s)][i]
                d_attr[s][self.prob_key][node] = similarities[int(s)][i]

    # Node2Vec NODES
    def precompute_network_probabilities(self):

        d_graph = self.d_graph
        first_travel_done = set()

        nodes_generator = self.graph.nodes()

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.probabilities_key not in d_graph[source]:
                d_graph[source][self.probabilities_key] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.probabilities_key not in d_graph[current_node]:
                    d_graph[current_node][self.probabilities_key] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.p_key, self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.q_key, self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]: # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.probabilities_key][source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.first_travel_key] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.neighbors_key] = d_neighbors

    def generate_single_network_walk(self, source):

        walk = [source]
        d_graph = self.d_graph
        while len(walk) < self.walk_length:
            walk_options = d_graph[walk[-1]].get(self.neighbors_key, None)

            # Skip dead end nodes
            if not walk_options:
                break

            if len(walk) == 1:  # For the first step
                probabilities = d_graph[walk[-1]][self.first_travel_key]
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
            else:
                probabilities = d_graph[walk[-1]][self.probabilities_key][walk[-2]]
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

            walk.append(walk_to)
        walk = list(map(str, walk))

        return walk

    def generate_single_attr_walk(self, source):
        d_attr = self.d_attr
        walk = [source]

        while len(walk) < self.walk_length:
            destinations = list(d_attr[walk[-1]][self.prob_key].keys())  # possible destinations
            probabilities = list(d_attr[walk[-1]][self.prob_key].values())  # list of probabilities
            if not np.any(probabilities):
                break
            walk_to = np.random.choice(destinations, size=1, p=probabilities)[0]  # make a choice
            walk.append(walk_to)
        walk = list(map(str, walk))  # make sure walk contains only strings

        return walk

    def generate_walks(self):
        for n_walk in range(self.num_walks):
            nodes = list(self.d_attr.keys())
            random.shuffle(nodes)
            for source in nodes:
                choice = np.random.sample(size=1)[0]
                if choice > self.t:
                    current_walk = self.generate_single_network_walk(str(source))
                else:
                    current_walk = self.generate_single_attr_walk(source)
                self.walks.append(current_walk)

        return self.walks

    def fit(self, **skip_gram_params):

        print('Pre-computing probabilities...')
        self.precompute_attr_probabilities()
        self.precompute_network_probabilities()
        print('Generating walks...')
        self.generate_walks()

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'vector_size' not in skip_gram_params:
            skip_gram_params['vector_size'] = self.dim

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = self.sg  # 1 - use skip-gram; otherwise, use CBOW

        model = gensim.models.Word2Vec(self.walks, **skip_gram_params)
        if self.labels is None:  # do not need to label output, just return
            return model.wv
        else:  # create dictionary with source labels before outputting
            output = dict()
            for node in model.wv.key_to_index:
                output[self.labels[int(node)].strip()] = model.wv[node]
            return output





