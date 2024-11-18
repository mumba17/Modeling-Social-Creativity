import mesa
import numpy as np
from knn import kNN
from features import FeatureExtractor
import genart as genart
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

log_dir = f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v1"

class Agent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.k = 10
        self.knn = kNN(k=self.k, agent_id=unique_id)
        self.preferred_novelty = np.random.normal(0.5, 0.1)
        
        # Parameters for reward/punishment sigmoids
        self.reward_threshold = self.preferred_novelty - 0.2  # n1 in the paper
        self.punish_threshold = self.preferred_novelty + 0.2  # n2 in the paper
        self.sigmoid_steepness = 10.0
        self.punishment_weight = 1.2  # Alpha in the paper
        
        self.gen_depth = np.random.randint(4, 8)
        self.current_expression = None
        self.artifact_memory = [] # Store previous successful expressions
        
        # Connection parameters
        self.decay_rate = 0.95  # Rate at which connections decay
        self.min_connection = 0.1  # Threshold below which connections break
        self.max_connection = 1.0  # Maximum connection strength
        self.reinforcement_rate = 0.4  # How quickly connections strengthen/weaken
        self.exploration_prob = 0.2  # Probability to form new random connection
        
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

    def get_connected_agents(self):
        """Get list of agents this agent is connected to (non-zero weights)"""
        connections = self.model.connections[self.unique_id]
        return np.where(abs(connections) > self.min_connection)[0]
    
    def form_new_connection(self):
        """Form new random connection with unconnected agent"""
        # Get currently unconnected agents
        connections = self.model.connections[self.unique_id]
        unconnected = np.where(abs(connections) <= self.min_connection)[0]
        unconnected = unconnected[unconnected != self.unique_id]  # Remove self
        
        if len(unconnected) > 0:
            # Choose random unconnected agent
            new_connection = np.random.choice(unconnected)
            # Initialize with small random weight
            self.model.connections[self.unique_id, new_connection] = np.random.uniform(-0.3, 0.3)
    
    def update_connection(self, other_agent_id, interest_value):
        """Update connection weight based on interest in received artifact"""
        current_weight = self.model.connections[self.unique_id, other_agent_id]
        
        # Update weight based on interest
        delta = self.reinforcement_rate * interest_value
        new_weight = current_weight + delta
        
        # Clip to valid range
        new_weight = np.clip(new_weight, -self.max_connection, self.max_connection)
        
        # Update connection weight
        self.model.connections[self.unique_id, other_agent_id] = new_weight
        
    def evaluate_artifact(self, artifact):
        """Evaluate artifact novelty and calculate interest"""
        try:
            features = artifact['features']
            
            # Add features to kNN memory
            self.knn.add_feature_vectors(features)
            
            # Get novelty score from kNN distances
            distances = self.knn.aggregate_distances(method='mean')
            novelty = np.mean(list(distances.values())) if distances else 0.0
            
            # Calculate interest using hedonic function
            interest = self.hedonic_evaluation(novelty)
            
            # Only remove features if exceeding capacity
            if self.knn.feature_vectors.shape[0] > 100:  # Set reasonable capacity
                self.knn.remove_feature_vectors([0])  # Remove oldest
                
            return interest
                
        except Exception as e:
            print(f"Error evaluating artifact: {e}")
            return 0.0
        

    def step(self):
        # Generate and evaluate own artifact 
        self_artifact = self.generate_artifact()
        interest = self.evaluate_artifact(self_artifact)
        
        # Store successful artifacts with complete info
        if interest > 0.3:  # High personal interest threshold
            if len(self.artifact_memory) >= 10:
                self.artifact_memory.pop(0)
            self.artifact_memory.append({
                'expression': self_artifact['expression'],
                'features': self_artifact['features'], 
                'interest': interest,  # Store the interest value
                'timestamp': self.model.schedule.time
            })
        
        # Submit to domain if interesting enough
        if interest > self.model.domain_threshold:  # Lower domain threshold
            self.model.add_to_domain(self_artifact)
        
        # Decay existing connections
        connections = self.model.connections[self.unique_id]
        connections *= self.decay_rate
        
        # Remove weak connections
        connections[abs(connections) < self.min_connection] = 0
        
        # Possibly form new random connection
        if random.random() < self.exploration_prob:
            self.form_new_connection()
        
        # Get currently connected agents
        connected_agents = self.get_connected_agents()
        
        if len(connected_agents) > 0:
            # Select agent to request from based on connection weights
            weights = self.model.connections[self.unique_id, connected_agents]
            probs = abs(weights) / abs(weights).sum()
            selected_agent = np.random.choice(connected_agents, p=probs)
            
            # Request artifact from selected agent
            selected_agent_object = self.model.schedule.agents[selected_agent]
            received_artifact = selected_agent_object.generate_artifact()
            
            # Evaluate received artifact using kNN-based novelty detection
            interest = self.evaluate_artifact(received_artifact)
            
            # Update connection based on interest
            self.update_connection(selected_agent, interest)
            
            # If interesting enough, add to memory and possibly adopt
            if interest > 0.5:  # Higher threshold for others' work
                # Store successful expression in memory
                if len(self.artifact_memory) >= 10:
                    self.artifact_memory.pop(0)  # Remove oldest
                self.artifact_memory.append(received_artifact['expression'])
                
                # Potentially adopt and modify the expression
                if random.random() < interest:  # More likely to adopt if more interesting
                    self.current_expression = received_artifact['expression']._copy()
                    self.current_expression.mutate(rate=0.1)
                
                # Consider adding to domain if highly interesting
                if interest > self.model.domain_threshold:
                    self.model.add_to_domain(received_artifact)
        self.model.writer.add_image('Connections', self.model.connections, self.model.schedule.time, dataformats='HW')
            
        
class Model(mesa.Model):
    def __init__(self, number_agents=100):
        super().__init__()
        self.num_agents = number_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.feature_extractor = FeatureExtractor(output_dims=64)
        self.image_generator = genart.ImageGenerator(48, 48)
        
        # Connection parameters
        self.connections = np.zeros((number_agents, number_agents))
        self.connection_density = 0.25
        
        # Domain repository
        self.domain = []
        self.domain_threshold = 0.02
        self.max_domain_size = 1000
        
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Data collection for analysis
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "domain_size": lambda m: len(m.domain),
                "avg_connections": lambda m: np.mean(np.count_nonzero(m.connections, axis=1)),
                "connection_density": lambda m: np.count_nonzero(m.connections) / (m.num_agents * m.num_agents),
                "avg_interest": self.get_average_interest,
                "num_cliques": self.count_cliques
            },
            agent_reporters={
                "num_connections": lambda a: len(a.get_connected_agents()),
                "memory_size": lambda a: len(a.artifact_memory),
                "avg_connection_strength": lambda a: np.mean(np.abs(self.connections[a.unique_id]))
            }
        )
        
        # Initialize agents
        self._initialize_agents()
        # Initialize random sparse connections
        self._initialize_connections()
        
        self.running = True
        
    def _initialize_agents(self):
        """Create agents with normally distributed novelty preferences"""
        for i in range(self.num_agents):
            agent = Agent(i, self)
            self.schedule.add(agent)
            
    def _initialize_connections(self):
        """Initialize sparse random connections between agents"""
        for i in range(self.num_agents):
            # Select random subset of other agents to connect with
            num_connections = int(self.connection_density * self.num_agents)
            possible_connections = list(range(self.num_agents))
            possible_connections.remove(i)  # Remove self from possible connections
            
            initial_connections = np.random.choice(
                possible_connections,
                size=num_connections,
                replace=False
            )
            
            # Initialize connections with small random weights
            self.connections[i, initial_connections] = np.random.uniform(-0.3, 0.3, size=num_connections)
            
    def add_to_domain(self, artifact):
        """Add artifact to domain repository if space available"""
        if len(self.domain) >= self.max_domain_size:
            self.domain.pop(0)  # Remove oldest artifact
        
        # Store artifact with metadata
        artifact_entry = {
            'artifact': artifact,
            'timestamp': self.schedule.time,
            'creator_id': artifact.get('creator_id'),
            'interest_score': artifact.get('interest_score')
        }
        self.domain.append(artifact_entry)
        self.writer.add_scalar('Domain Size', len(self.domain), self.schedule.time)
        
    def get_average_interest(self):
        """Calculate average interest across all agents' recent interactions"""
        if not self.schedule.agents:
            return 0
        
        total_interest = 0
        count = 0
        for agent in self.schedule.agents:
            if agent.artifact_memory:
                for artifact in agent.artifact_memory:
                    # Get features from stored expression
                    image = self.image_generator.generate(artifact)
                    features = self.feature_extractor.extract_features(image)
                    total_interest += agent.hedonic_evaluation(features)
                    count += 1
                    
        avg_interest = total_interest / count if count > 0 else 0
        self.writer.add_scalar('Average Interest', avg_interest, self.schedule.time)
                    
        return avg_interest
        
    def count_cliques(self):
        """Identify cliques in the connection network"""
        # Convert connection matrix to binary adjacency matrix
        adjacency = np.where(abs(self.connections) > 0.3, 1, 0)
        
        # Count groups of mutually connected agents
        cliques = 0
        visited = set()
        
        def find_clique(agent_id, current_clique):
            for neighbor in np.where(adjacency[agent_id] == 1)[0]:
                if neighbor not in visited:
                    # Check if neighbor is connected to all clique members
                    if all(adjacency[neighbor][member] == 1 for member in current_clique):
                        visited.add(neighbor)
                        find_clique(neighbor, current_clique | {neighbor})
            
            if len(current_clique) >= 3:  # Consider groups of 3+ as cliques
                nonlocal cliques
                cliques += 1
        
        for i in range(self.num_agents):
            if i not in visited:
                visited.add(i)
                find_clique(i, {i})
                
        return cliques
        
    def get_network_stats(self):
        """Calculate various network statistics"""
        stats = {
            'avg_connections': np.mean(np.count_nonzero(self.connections, axis=1)),
            'connection_density': np.count_nonzero(self.connections) / (self.num_agents * self.num_agents),
            'num_cliques': self.count_cliques(),
            'avg_weight': np.mean(np.abs(self.connections[self.connections != 0]))
        }
        
        for stat, value in stats.items():
            self.writer.add_scalar(f'Network/{stat}', value, self.schedule.time)
        return stats    
    
    def step(self):
        """Advance the model by one step"""
        self.schedule.step()
        
        # Collect and log metrics each step
        metrics = self.collect_metrics()
        self.log_metrics(metrics)
        self.datacollector.collect(self)

    def collect_metrics(self):
        """Collect all relevant metrics for current step"""
        metrics = {
            'network': self.get_network_stats(),
            'domain': {
                'size': len(self.domain)
            },
            'agents': self.get_agent_metrics(),
            'interest': self.get_average_interest()
        }
        
        print(f"\nStep {self.schedule.time} Metrics:")
        print(f"Average Interest: {metrics['interest']:.3f}")
        print(f"Domain Size: {metrics['domain']['size']}")
        print(f"Network Stats:")
        for key, val in metrics['network'].items():
            print(f"  {key}: {val:.3f}")
        print(f"Number of Cliques: {metrics['network']['num_cliques']}")
        
        return metrics

    def get_agent_metrics(self):
        """Collect metrics for all agents"""
        metrics = {
            'avg_connections': [],
            'avg_memory_size': [],
            'interest_values': []
        }
        
        for agent in self.schedule.agents:
            metrics['avg_connections'].append(len(agent.get_connected_agents()))
            metrics['avg_memory_size'].append(len(agent.artifact_memory))
            
            # Calculate average interest for agent's memory
            if agent.artifact_memory:
                interests = []
                for artifact in agent.artifact_memory[-5:]:  # Look at last 5 artifacts
                    if isinstance(artifact, dict) and 'interest' in artifact:
                        interests.append(artifact['interest'])
                if interests:
                    metrics['interest_values'].append(np.mean(interests))
        
        return {
            'avg_connections': np.mean(metrics['avg_connections']),
            'avg_memory_size': np.mean(metrics['avg_memory_size']),
            'avg_agent_interest': np.mean(metrics['interest_values']) if metrics['interest_values'] else 0
        }

    def log_metrics(self, metrics):
        """Log all metrics to tensorboard"""
        # Log network metrics
        for key, value in metrics['network'].items():
            self.writer.add_scalar(f'Network/{key}', value, self.schedule.time)
        
        # Log domain metrics
        self.writer.add_scalar('Domain/size', metrics['domain']['size'], self.schedule.time)
        
        # Log agent metrics
        for key, value in metrics['agents'].items():
            self.writer.add_scalar(f'Agents/{key}', value, self.schedule.time)
        
        # Log overall interest
        self.writer.add_scalar('Interest/average', metrics['interest'], self.schedule.time)

    def get_average_interest(self):
        """Calculate average interest across all agents' recent interactions"""
        if not self.schedule.agents:
            return 0
        
        interest_values = []
        for agent in self.schedule.agents:
            if agent.artifact_memory:
                # Get most recent artifacts
                recent_artifacts = agent.artifact_memory[-5:]  # Look at last 5 artifacts
                for artifact in recent_artifacts:
                    if isinstance(artifact, dict) and 'interest' in artifact:
                        interest_values.append(artifact['interest'])
        
        avg_interest = np.mean(interest_values) if interest_values else 0
        return avg_interest
        
def run_simulation(num_agents=10, steps=50):
    """Run a complete simulation and return results"""
    print(f"\nStarting simulation with {num_agents} agents for {steps} steps")
    model = Model(num_agents)
    
    with tqdm(total=steps, desc="Simulation Progress") as pbar:
        for step in range(steps):
            print(f"\nStep {step+1}/{steps}")
            model.step()
            pbar.update(1)
    
    model.writer.close()
    
    print("\nSimulation completed!")
    print("Final metrics:")
    final_metrics = model.collect_metrics()
    
    return {
        'model': model,
        'model_data': model.datacollector.get_model_vars_dataframe(),
        'agent_data': model.datacollector.get_agent_vars_dataframe()
    }
        
if __name__ == "__main__":
    run_simulation()