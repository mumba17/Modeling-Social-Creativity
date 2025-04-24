import logging
import networkx as nx
import numpy as np
import os

from timing_utils import time_it

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NetworkTracker:
    """
    Tracks interactions between agents using adjacency matrices and a NetworkX graph.
    Records:
      - communication_matrix : total messages sent from i->j
      - acceptance_matrix : accepted communications from i->j
      - rejection_matrix : rejected communications from i->j
    """

    def __init__(self, num_agents, log_dir):
        """
        Parameters
        ----------
        num_agents : int
        log_dir : str
            Directory for saving network snapshots
        """
        self.num_agents = num_agents
        self.log_dir = log_dir
        self.network_dir = f"{log_dir}/network"
        os.makedirs(self.network_dir, exist_ok=True)

        self.communication_matrix = np.zeros((num_agents, num_agents))
        self.acceptance_matrix = np.zeros((num_agents, num_agents))
        self.rejection_matrix = np.zeros((num_agents, num_agents))

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_agents))

    @time_it
    def record_interaction(self, sender_id, receiver_id, accepted, timestamp):
        """
        Record an interaction event.

        Parameters
        ----------
        sender_id : int
        receiver_id : int
        accepted : bool
        timestamp : int
        """
        self.communication_matrix[sender_id][receiver_id] += 1
        if accepted:
            self.acceptance_matrix[sender_id][receiver_id] += 1
            if self.graph.has_edge(sender_id, receiver_id):
                self.graph[sender_id][receiver_id]['weight'] += 1
            else:
                self.graph.add_edge(sender_id, receiver_id, weight=1)
        else:
            self.rejection_matrix[sender_id][receiver_id] += 1

    @time_it
    def get_agent_stats(self, agent_id):
        """
        Get communication stats for a specific agent.

        Returns
        -------
        dict
        """
        sent = np.sum(self.communication_matrix[agent_id])
        received = np.sum(self.communication_matrix[:, agent_id])
        acc_sent = np.sum(self.acceptance_matrix[agent_id])
        acc_received = np.sum(self.acceptance_matrix[:, agent_id])
        return {
            'total_sent': sent,
            'total_received': received,
            'acceptances_sent': acc_sent,
            'acceptances_received': acc_received,
            'acceptance_rate_sent': acc_sent / (sent + 1e-10),
            'acceptance_rate_received': acc_received / (received + 1e-10)
        }

    @time_it
    def save_snapshot(self, step):
        """
        Save network matrices and a .gexf graph snapshot.
        """
        np.save(f"{self.network_dir}/comm_matrix_{step}.npy", self.communication_matrix)
        np.save(f"{self.network_dir}/accept_matrix_{step}.npy", self.acceptance_matrix)
        np.save(f"{self.network_dir}/reject_matrix_{step}.npy", self.rejection_matrix)
        nx.write_gexf(self.graph, f"{self.network_dir}/network_{step}.gexf")

    @time_it
    def get_network_metrics(self):
        """
        Compute basic network metrics (density, clustering, path length, etc.).

        Returns
        -------
        dict
        """
        if self.graph.number_of_edges() == 0:
            return {
                'density': 0,
                'average_clustering': 0,
                'average_path_length': float('inf'),
                'average_degree': 0,
                'total_edges': 0,
                'connected_components': self.num_agents,
                'acceptance_rate': 0
            }
        metrics = {
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'average_path_length': (
                nx.average_shortest_path_length(self.graph)
                if nx.is_connected(self.graph) else float('inf')
            ),
            'average_degree': np.mean([d for _, d in self.graph.degree()]),
            'total_edges': self.graph.number_of_edges(),
            'connected_components': nx.number_connected_components(self.graph),
            'acceptance_rate': np.sum(self.acceptance_matrix) / (np.sum(self.communication_matrix) + 1e-10)
        }

        centrality = nx.degree_centrality(self.graph)
        top_agents = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (agent_id, cent) in enumerate(top_agents):
            metrics[f'top_agent_{i}_id'] = agent_id
            metrics[f'top_agent_{i}_centrality'] = cent
        return metrics
