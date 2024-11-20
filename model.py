import mesa
import numpy as np
from knn import kNN
from features import FeatureExtractor
import torchvision.transforms as transforms
import genart as genart
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import os
import io
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

log_dir = f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v1"

class Agent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.k = 10
        self.knn = kNN(k=self.k, agent_id=unique_id)
        
        # Normally distributed novelty preference
        self.preferred_novelty = np.random.normal(0.5, 0.155)
        
        # Wundt curve parameters
        self.reward_threshold = self.preferred_novelty - 0.2
        self.punish_threshold = self.preferred_novelty + 0.2
        self.sigmoid_steepness = 10.0
        self.punishment_weight = 1.2
        
        # Generation parameters
        self.gen_depth = np.random.randint(4, 8)
        self.current_expression = None
        self.artifact_memory = []
        
        # Communication
        self.inbox = []
            
    def sigmoid(self, x, threshold):
        """Helper function to compute sigmoid curve"""
        return 1 / (1 + np.exp(-self.sigmoid_steepness * (x - threshold)))
        
    def hedonic_evaluation(self, input_value):
        """
        Compute hedonic value based on novelty using Wundt curve.
        Returns value between -1 and 1
        """
        # Compute novelty if features provided
        if torch.is_tensor(input_value):
            distances = self.knn.aggregate_distances(method='mean')
            novelty = np.mean(list(distances.values())) if distances else 0.0
        else:
            novelty = float(input_value)  # Ensure float type
            
        # Normalize novelty to 0-1 range if needed
        if novelty > 1.0:
            novelty = 1.0 / (1.0 + np.log(novelty))
            
        # Add checks for NaN/Inf
        if np.isnan(novelty) or np.isinf(novelty):
            print(f"Warning: Invalid novelty value: {novelty}")
            return 0.0
            
        # Compute reward and punishment components
        try:
            reward = self.sigmoid(novelty, self.reward_threshold)
            punishment = self.sigmoid(novelty, self.punish_threshold)
            
            # Combine using the Wundt curve formula
            hedonic_value = reward - self.punishment_weight * punishment
            
            # Ensure output is valid
            if np.isnan(hedonic_value) or np.isinf(hedonic_value):
                print(f"Warning: Invalid hedonic value calculated. R:{reward}, P:{punishment}")
                return 0.0
                
            return np.clip(hedonic_value, -1, 1)
            
        except Exception as e:
            print(f"Error in hedonic evaluation: {e}")
            return 0.0

    def generate_artifact(self):
        # Generate artifact
        if not self.current_expression or random.random() < 0.1:
            self.current_expression = genart.ExpressionNode.create_random(depth=8)
        else:
            if self.artifact_memory and random.random() < 0.5:
                parent = random.choice(self.artifact_memory)
                if isinstance(parent, dict):
                    parent = parent['expression']
                self.current_expression = self.current_expression.breed(parent)
            else:
                self.current_expression.mutate(rate=0.1)

        # Generate image and extract features
        try:
            image = self.model.image_generator.generate(self.current_expression)
            features = self.model.feature_extractor.extract_features(image)
            
            # Validate features
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"Warning: Invalid features generated for agent {self.unique_id}")
                features = torch.zeros_like(features)
                
            self.generated_image = image
            self.generated_expression = self.current_expression.to_string()
                
            return {
                'image': image,
                'features': features,
                'expression': self.current_expression
            }
            
        except Exception as e:
            print(f"Error in generate_artifact: {e}")
            # Return safe default
            return {
                'image': None,
                'features': torch.zeros((1, 64), device=self.model.feature_extractor.device),
                'expression': self.current_expression
            }
        
    def log_metrics(self, artifact_data=None):
        """Enhanced logging with dedicated agent cards and image saving"""
        step = self.model.schedule.time
        writer = self.model.agent_writers[self.unique_id]
        
        # Agent parameters
        writer.add_scalar('parameters/preferred_novelty', self.preferred_novelty, step)
        writer.add_scalar('parameters/reward_threshold', self.reward_threshold, step)
        writer.add_scalar('parameters/punish_threshold', self.punish_threshold, step)
        
        # Memory metrics
        writer.add_scalar('memory/size', len(self.artifact_memory), step)
        writer.add_scalar('memory/inbox_size', len(self.inbox), step)
        
        if artifact_data:
            # Log generated artifact
            image = artifact_data['image']
            expression = artifact_data['expression'].to_string()
            
            # Save image
            if image:
                # Convert PIL image to tensor
                image_tensor = transforms.ToTensor()(image)
                writer.add_image('generated/current_image', image_tensor, step)
            
            # Log expression
            writer.add_text('generated/expression', expression, step)
            
            # Log novelty/interest metrics
            distances = self.knn.aggregate_distances(method='mean')
            if distances:
                avg_distance = np.mean(list(distances.values()))
                writer.add_scalar('evaluation/avg_distance', avg_distance, step)
                
            if 'novelty' in artifact_data:
                writer.add_scalar('evaluation/novelty', artifact_data['novelty'], step)
            if 'interest' in artifact_data:
                writer.add_scalar('evaluation/interest', artifact_data['interest'], step)
        
        # Communication metrics
        in_edges = self.model.communication_graph.in_degree(self.unique_id)
        out_edges = self.model.communication_graph.out_degree(self.unique_id)
        writer.add_scalar('communication/in_degree', in_edges, step)
        writer.add_scalar('communication/out_degree', out_edges, step)
        
        # Add custom figure showing agent's state
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.title(f"Agent {self.unique_id} Wundt Curve")
        x = np.linspace(0, 1, 100)
        y = [self.hedonic_evaluation(xi) for xi in x]
        plt.plot(x, y)
        plt.grid(True)
        
        # Convert figure to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        writer.add_image('state/wundt_curve', transforms.ToTensor()(img), step)
        plt.close(fig)

    def step(self):
        """One step of agent behavior encompassing:
        - Artifact generation and self-evaluation 
        - Sharing if sufficiently novel
        - Processing received artifacts"""

        # Generate and evaluate artifact
        artifact = self.generate_artifact()
        
        # Calculate self-evaluated novelty and interest
        if artifact['features'] is not None:
            # Add to personal experience/memory
            self.knn.add_feature_vectors(artifact['features'])
            
            # Get novelty distance using kNN
            distances = self.knn.aggregate_distances(method='mean')
            novelty = np.mean(list(distances.values())) if distances else 0.0
            
            # Calculate interest using Wundt curve (hedonic function)
            interest = self.hedonic_evaluation(novelty)
            
            artifact['novelty'] = novelty
            artifact['interest'] = interest
            
            # If interesting enough, share with random other agent
            if interest > self.model.communication_threshold:
                # Package artifact with metadata
                message = {
                    'artifact': artifact,
                    'sender_id': self.unique_id,
                    'novelty': novelty,
                    'interest': interest,
                    'timestamp': self.model.schedule.time
                }
                
                # Select random recipient that isn't self
                other_agents = [a for a in self.model.schedule.agents 
                            if a.unique_id != self.unique_id]
                if other_agents:
                    recipient = self.random.choice(other_agents)
                    recipient.inbox.append(message)
                    
                    # Log communication for analysis
                    self.model.log_communication(self.unique_id, recipient.unique_id)

        # Process inbox
        for message in self.inbox:
            received_artifact = message['artifact']
            sender_id = message['sender_id']
            
            # Evaluate received artifact
            if received_artifact['features'] is not None:
                # Calculate novelty and interest
                self.knn.add_feature_vectors(received_artifact['features'])
                distances = self.knn.aggregate_distances(method='mean')
                novelty = np.mean(list(distances.values())) if distances else 0.0
                interest = self.hedonic_evaluation(novelty)
                
                # If interesting enough, consider for domain
                if interest > self.model.domain_threshold:
                    domain_entry = {
                        'artifact': received_artifact,
                        'creator_id': sender_id,
                        'evaluator_id': self.unique_id,
                        'novelty': novelty,
                        'interest': interest,
                        'timestamp': self.model.schedule.time
                    }
                    self.model.add_to_domain(domain_entry)

        # Clear inbox
        self.inbox = []
        self.log_metrics(artifact)

class Model(mesa.Model):
    def __init__(self, number_agents=100):
        super().__init__()
        self.num_agents = number_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.feature_extractor = FeatureExtractor(output_dims=64)
        self.image_generator = genart.ImageGenerator(48, 48)

        # Thresholds
        self.communication_threshold = 0.3  # When to share
        self.domain_threshold = 0.5  # When to add to domain

        # Domain repository
        self.domain = []
        self.max_domain_size = 10000000
        
        # Network analysis
        self.communication_graph = nx.DiGraph()
        self.communication_graph.add_nodes_from(range(number_agents))
        
        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_groups = {
            'agents': {},
            'domain': {},
            'network': {},
            'system': {}
        }
        
        # Initialize agents
        self._initialize_agents()
        self.running = True
        
        # Create dedicated logging directories
        self.log_base = log_dir
        self.image_dir = f"{log_dir}/images"
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Create agent-specific writers
        self.agent_writers = {}
        for i in range(number_agents):
            agent_dir = f"{log_dir}/agent_{i}"
            os.makedirs(agent_dir, exist_ok=True)
            self.agent_writers[i] = SummaryWriter(log_dir=agent_dir)

    def add_to_domain(self, entry):
        """Enhanced domain addition with proper logging"""
        if len(self.domain) >= self.max_domain_size:
            self.domain.pop(0)
        
        artifact = entry['artifact']
        
        # Save the image
        image_path = f"{self.image_dir}/domain_{len(self.domain)}.png"
        artifact['image'].save(image_path)
        
        # Save the complete entry with metadata
        domain_entry = {
            'image_path': image_path,
            'expression': artifact['expression'].to_string(),
            'creator_id': entry['creator_id'],
            'evaluator_id': entry['evaluator_id'],
            'novelty': entry['novelty'],
            'interest': entry['interest'],
            'timestamp': entry['timestamp']
        }
        
        self.domain.append(domain_entry)
        
        # Log domain metrics
        self.writer.add_scalar('domain/total_size', len(self.domain), self.schedule.time)
        self.writer.add_scalar('domain/last_novelty', entry['novelty'], self.schedule.time)
        self.writer.add_scalar('domain/last_interest', entry['interest'], self.schedule.time)
        

    def log_communication(self, sender_id, recipient_id):
        """Track communication patterns between agents"""
        if not self.communication_graph.has_edge(sender_id, recipient_id):
            self.communication_graph.add_edge(sender_id, recipient_id, weight=0)
        self.communication_graph[sender_id][recipient_id]['weight'] += 1
        
    def _initialize_agents(self):
        """Create agents with normally distributed novelty preferences"""
        for i in range(self.num_agents):
            agent = Agent(i, self)
            self.schedule.add(agent)
    
    def log_system_metrics(self):
        """Log system-wide metrics"""
        step = self.schedule.time
        
        # Domain metrics
        self.writer.add_scalar("domain/size", len(self.domain), step)
        if self.domain:
            recent_interests = [entry.get('interest_score', 0) for entry in self.domain[-100:]]
            self.writer.add_scalar("domain/recent_avg_interest", np.mean(recent_interests), step)
        
        # Network metrics
        graph = self.communication_graph
        if len(graph.nodes) > 0:
            # Connectivity metrics
            density = nx.density(graph)
            self.writer.add_scalar("network/density", density, step)
            
            # Clique analysis
            cliques = list(nx.find_cliques(graph.to_undirected()))
            max_clique_size = len(max(cliques, key=len)) if cliques else 0
            self.writer.add_scalar("network/max_clique_size", max_clique_size, step)
            
            # Centrality metrics
            in_degree_centrality = nx.in_degree_centrality(graph)
            self.writer.add_scalar("network/avg_in_degree_centrality", 
                                np.mean(list(in_degree_centrality.values())), step)
            
            if graph.number_of_edges() > 0:
                communities = nx.community.greedy_modularity_communities(graph.to_undirected())
                self.writer.add_scalar("network/num_communities", len(communities), step)
            else:
                # When no edges exist, each node is its own community
                self.writer.add_scalar("network/num_communities", len(graph.nodes), step)
    
    def step(self):
        """Advance the model by one step with logging"""
        self.schedule.step()
        self.log_system_metrics()
        
        # Log network graph periodically (every 10 steps)
        if self.schedule.time % 10 == 0:
            fig = plt.figure()
            nx.draw(self.communication_graph, with_labels=True, 
                    node_color='lightblue', node_size=500)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            self.writer.add_image('network/graph', transforms.ToTensor()(img), self.schedule.time)
            plt.close(fig)
            
def run_simulation(num_agents=25, steps=20):
    """Run simulation with proper cleanup"""
    print(f"\nStarting simulation with {num_agents} agents for {steps} steps")
    model = Model(num_agents)
    
    with tqdm(total=steps, desc="Simulation Progress") as pbar:
        for step in range(steps):
            print(f"\nStep {step+1}/{steps}")
            model.step()
            pbar.update(1)
    
    # Proper cleanup of all writers
    model.writer.close()
    for writer in model.agent_writers.values():
        writer.close()
    
    print("\nSimulation completed!")
        
if __name__ == "__main__":
    run_simulation()