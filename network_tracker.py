import networkx as nx
import numpy as np
import os
from timing_utils import time_it

class NetworkTracker:
    def __init__(self, num_agents, log_dir):
        """Initialize network tracking structures"""
        self.num_agents = num_agents
        self.log_dir = log_dir
        
        # Create network directory
        self.network_dir = f"{log_dir}/network"
        os.makedirs(self.network_dir, exist_ok=True)
        
        # Initialize matrices
        self.communication_matrix = np.zeros((num_agents, num_agents))  # Total communications
        self.acceptance_matrix = np.zeros((num_agents, num_agents))     # Accepted communications
        self.rejection_matrix = np.zeros((num_agents, num_agents))      # Rejected communications
        
        # Initialize NetworkX graph for data collection only
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_agents))
        
        
    @time_it
    def record_interaction(self, sender_id, receiver_id, accepted, timestamp):
        """Record an interaction between agents"""
        # Update communication counts
        self.communication_matrix[sender_id][receiver_id] += 1
        
        if accepted:
            self.acceptance_matrix[sender_id][receiver_id] += 1
            # Update graph edge weight (undirected)
            if self.graph.has_edge(sender_id, receiver_id):
                self.graph[sender_id][receiver_id]['weight'] += 1
            else:
                self.graph.add_edge(sender_id, receiver_id, weight=1)
        else:
            self.rejection_matrix[sender_id][receiver_id] += 1
    
    @time_it
    def get_agent_stats(self, agent_id):
        """Get communication statistics for a specific agent"""
        return {
            'total_sent': np.sum(self.communication_matrix[agent_id]),
            'total_received': np.sum(self.communication_matrix[:, agent_id]),
            'acceptances_sent': np.sum(self.acceptance_matrix[agent_id]),
            'acceptances_received': np.sum(self.acceptance_matrix[:, agent_id]),
            'acceptance_rate_sent': np.sum(self.acceptance_matrix[agent_id]) / 
                                  (np.sum(self.communication_matrix[agent_id]) + 1e-10),
            'acceptance_rate_received': np.sum(self.acceptance_matrix[:, agent_id]) / 
                                      (np.sum(self.communication_matrix[:, agent_id]) + 1e-10)
        }
        
    @time_it 
    def save_snapshot(self, step):
        """Save network state to files"""
        # Save matrices
        np.save(f"{self.network_dir}/comm_matrix_{step}.npy", self.communication_matrix)
        np.save(f"{self.network_dir}/accept_matrix_{step}.npy", self.acceptance_matrix)
        np.save(f"{self.network_dir}/reject_matrix_{step}.npy", self.rejection_matrix)
        
        # Save graph structure
        nx.write_gexf(self.graph, f"{self.network_dir}/network_{step}.gexf")
     
    @time_it       
    def get_network_metrics(self):
        """Calculate basic network metrics without visualization"""
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
            'average_path_length': nx.average_shortest_path_length(self.graph) 
                                 if nx.is_connected(self.graph) else float('inf'),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'total_edges': self.graph.number_of_edges(),
            'connected_components': nx.number_connected_components(self.graph),
            'acceptance_rate': np.sum(self.acceptance_matrix) / 
                             (np.sum(self.communication_matrix) + 1e-10)
        }
        
        # Get degree centrality for top agents (data only)
        centrality = nx.degree_centrality(self.graph)
        top_agents = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (agent_id, cent) in enumerate(top_agents):
            metrics[f'top_agent_{i}_id'] = agent_id
            metrics[f'top_agent_{i}_centrality'] = cent
            
        return metrics