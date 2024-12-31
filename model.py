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
        self.knn = kNN(agent_id=unique_id)
        
        # Normally distributed novelty preference
        self.preferred_novelty = np.random.normal(0.5, 0.155)
        self.preferred_novelty = np.clip(self.preferred_novelty, 0.1, 0.9)
        
        self.average_interest_self = []
        self.average_interest_other = []
        
        # Generation parameters
        self.gen_depth = np.random.randint(3, 9)
        self.current_expression = None
        self.artifact_memory = []
        
        self.reward_threshold = max(0.1, self.preferred_novelty - 0.2)
        self.punish_threshold = min(0.9, self.preferred_novelty + 0.2)
        self.sigmoid_steepness = 10.0
        self.punishment_weight = 1.2
        self.alpha = 0.5  # Interest decay rate
        self.average_interest = 0.0
        
        self.boredom_threshold = 0.2
        
        self.init_plot = True
        
        # Communication
        self.inbox = []
        
    def sigmoid(self, x, threshold):
        """Helper function for Wundt curve computation"""
        return 1 / (1 + np.exp(-self.sigmoid_steepness * (x - threshold)))

    def hedonic_evaluation(self, novelty):
        """
        Compute hedonic value from novelty using Wundt curve with proper scaling
        Returns value between -1 and 1
        """
        # Make sure novelty is between 0 and 1
        novelty = np.clip(novelty, 0, 1)
        
        # Calculate reward and punishment components
        reward = self.sigmoid(novelty, self.reward_threshold)
        punishment = self.sigmoid(novelty, self.punish_threshold)
        
        # The Wundt curve is the difference between reward and punishment
        # Scale it properly to ensure full range usage
        hedonic_value = (reward - self.punishment_weight * punishment)
        
        # Scale to [-1, 1] range
        max_possible = 1 - 0  # Max reward - min punishment
        min_possible = 0 - self.punishment_weight  # Min reward - max punishment * weight
        hedonic_value = 2 * (hedonic_value - min_possible) / (max_possible - min_possible) - 1
        
        # Update running average with proper scaling
        self.average_interest = self.alpha * self.average_interest + (1 - self.alpha) * hedonic_value
        
        return hedonic_value

    def generate_artifact(self):
        # Generate artifact
        if not self.current_expression:
            self.current_expression = genart.ExpressionNode.create_random(depth=self.gen_depth)
        else:
            if self.artifact_memory:
                parent = random.choice(self.artifact_memory)
                if isinstance(parent, dict):
                    parent = parent['expression']
                self.current_expression = self.current_expression.breed(parent)

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
            print(f"Error in generate_artifact: {e}, Agent {self.unique_id}, Expression: {self.current_expression}, Depth: {self.gen_depth}")
            # Return safe default
            return {
                'image': None,
                'features': torch.zeros((1, 32), device=self.model.feature_extractor.device),
                'expression': self.current_expression
            }
        
    def log_metrics(self, artifact_data=None):
        """Enhanced logging with dedicated agent cards and image saving"""
        step = self.model.schedule.time
        writer = self.model.agent_writers[self.unique_id]
        
        if artifact_data is None:
            return
        
        if step % 20 == 0:
            
            # Memory metrics
            writer.add_scalar('memory/size', len(self.artifact_memory), step)
            
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

        
        if not self.init_plot:
            self.init_plot = True
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
            
            # Agent parameters
            writer.add_scalar('parameters/preferred_novelty', self.preferred_novelty, step)
            writer.add_scalar('parameters/reward_threshold', self.reward_threshold, step)
            writer.add_scalar('parameters/punish_threshold', self.punish_threshold, step)

    def step(self):
        """One step of agent behavior with updated boredom threshold"""
        # Generate and evaluate artifact
        artifact = self.generate_artifact()
        
        # Calculate self-evaluated novelty and interest
        if artifact['features'] is not None:
            # Add to personal experience/memory
            features = artifact['features']
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            elif len(features.shape) == 3:
                features = features.squeeze(1)
                
            self.knn.add_feature_vectors(features)
            self.artifact_memory.append(artifact)
            
            # Get novelty using batch method for consistency
            try:
                novelty_scores = self.knn.batch_get_novelty(features)
                novelty = novelty_scores[0].item()
                
                # Calculate interest using Wundt curve
                interest = self.hedonic_evaluation(novelty)
                self.average_interest_self.append(interest)
                
                # Log metrics
                artifact['novelty'] = novelty
                artifact['interest'] = interest
                
                # If interesting enough, share with random other agents
                if interest > self.model.self_threshold:
                    message = {
                        'artifact': artifact,
                        'sender_id': self.unique_id,
                        'timestamp': self.model.schedule.time
                    }
                    
                    other_agents = [a for a in self.model.schedule.agents 
                                if a.unique_id != self.unique_id]
                    if other_agents:
                        for _ in range(8):  # Share with 8 random agents
                            recipient = self.random.choice(other_agents)
                            recipient.inbox.append(message)
            
            except Exception as e:
                print(f"Error in agent step: {e}")
                print(f"Features shape: {features.shape}")
                interest = 0.0
                
            # Update accumulated interest with more detailed logging
            old_interest = self.average_interest
            self.average_interest = self.alpha * self.average_interest + (1 - self.alpha) * interest
            
            # Log interest changes in agent's writer
            writer = self.model.agent_writers[self.unique_id]
            writer.add_scalar('interest/current', interest, self.model.schedule.time)
            writer.add_scalar('interest/average', self.average_interest, self.model.schedule.time)
            writer.add_scalar('interest/delta', self.average_interest - old_interest, self.model.schedule.time)
            
            # Check for boredom using model's dynamic threshold
            if self.average_interest < self.model.boredom_threshold:
                # Retrieve artifact from domain
                domain_artifact = self.model.get_random_domain_artifact()
                if domain_artifact:
                    domain_features = domain_artifact['features']
                    if len(domain_features.shape) == 1:
                        domain_features = domain_features.unsqueeze(0)
                    elif len(domain_features.shape) == 3:
                        domain_features = domain_features.squeeze(1)
                        
                    domain_novelty_scores = self.knn.batch_get_novelty(domain_features)
                    domain_novelty = domain_novelty_scores[0].item()
                    domain_interest = self.hedonic_evaluation(domain_novelty)
                    
                    # Log domain interaction
                    writer.add_scalar('domain/interaction_novelty', domain_novelty, self.model.schedule.time)
                    writer.add_scalar('domain/interaction_interest', domain_interest, self.model.schedule.time)
                    
                    # If interesting enough, adopt it
                    if domain_interest > interest:
                        self.current_expression = domain_artifact['expression']
                        self.knn.add_feature_vectors(domain_features)
                        writer.add_scalar('domain/adoption', 1.0, self.model.schedule.time)
                    else:
                        writer.add_scalar('domain/adoption', 0.0, self.model.schedule.time)

        self.log_metrics(artifact)

class Model(mesa.Model):
    def __init__(self, number_agents=100):
        super().__init__()
        self.num_agents = number_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.feature_extractor = FeatureExtractor(output_dims=32)
        self.image_generator = genart.ImageGenerator(32, 32)

        # Thresholds
        self.self_threshold = None  # When to share
        self.domain_threshold = None  # When to add to domain
        
        self.interest_threshold_self_list = []
        self.interest_threshold_other_list = []
        
        self.communication_matrix = np.zeros((number_agents, number_agents))

        # Domain repository
        self.domain = []
        self.max_domain_size = 10000000
        
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
            
    def calculate_novelty_threshold(self):
        """Calculate thresholds based on interest distributions from all agents in current step"""
        window_size = 100  # Window size for rolling calculations
        
        # Get average interest values from this step from all agents
        step_interests_self = [np.mean(agent.average_interest_self[-window_size:]) 
                            for agent in self.schedule.agents 
                            if agent.average_interest_self]
        
        step_interests_other = [np.mean(agent.average_interest_other[-window_size:]) 
                            for agent in self.schedule.agents
                            if agent.average_interest_other]
        
        # Get current average interests for boredom calculation
        current_interests = [agent.average_interest 
                            for agent in self.schedule.agents]

        # Calculate percentile-based thresholds if we have values
        if step_interests_self:
            self.self_threshold = np.percentile(step_interests_self, 80)  # 80th percentile for sharing
        else:
            self.self_threshold = 0.1

        if step_interests_other:
            self.domain_threshold = np.percentile(step_interests_other, 80)  # 80th percentile for domain
        else:
            self.domain_threshold = 0.1
            
        if current_interests:
            self.boredom_threshold = np.percentile(current_interests, 10)  # 10th percentile for boredom
        else:
            self.boredom_threshold = 0.2

        # Log all thresholds
        self.writer.add_scalar('thresholds/communication', self.self_threshold, self.schedule.time)
        self.writer.add_scalar('thresholds/domain', self.domain_threshold, self.schedule.time)
        self.writer.add_scalar('thresholds/boredom', self.boredom_threshold, self.schedule.time)
        
        # Log distribution statistics
        if current_interests:
            self.writer.add_scalar('interest/mean', np.mean(current_interests), self.schedule.time)
            self.writer.add_scalar('interest/median', np.median(current_interests), self.schedule.time)
            self.writer.add_scalar('interest/std', np.std(current_interests), self.schedule.time)
            
            # Calculate percentage of bored agents
            bored_agents = sum(1 for i in current_interests if i < self.boredom_threshold)
            bored_percentage = (bored_agents / len(current_interests)) * 100
            self.writer.add_scalar('agents/bored_percentage', bored_percentage, self.schedule.time)

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
        self.writer.add_scalar('domain/last_interest', entry['interest'], self.schedule.time)
        self.writer.add_scalar('domain/last_novelty', entry['novelty'], self.schedule.time)
        
    def get_random_domain_artifact(self):
        """Return random artifact from domain if available"""
        if self.domain:
            domain_entry = random.choice(self.domain)
            # Need to reconstruct artifact structure
            return {
                'features': self.feature_extractor.extract_features(Image.open(domain_entry['image_path'])),
                'expression': genart.ExpressionNode.from_string(domain_entry['expression']),
                'image': Image.open(domain_entry['image_path'])
            }
        return None
    
    def process_inboxes_parallel(self):
        """Process all agent inboxes in parallel using CUDA streams"""
        if not torch.cuda.is_available():
            # Fallback to regular processing if CUDA not available
            self.process_inboxes_batch()
            return

        # Group messages by recipient
        agent_messages = {}
        for agent in self.schedule.agents:
            if agent.inbox:
                agent_messages[agent.unique_id] = {
                    'agent': agent,
                    'messages': agent.inbox
                }
        
        if not agent_messages:
            return

        # Create streams for each agent
        streams = {agent_id: torch.cuda.Stream() for agent_id in agent_messages.keys()}
        
        # Process each agent's messages in parallel streams
        results = {}  # Store computation results
        
        for agent_id, data in agent_messages.items():
            agent = data['agent']
            messages = data['messages']
            stream = streams[agent_id]
            
            with torch.cuda.stream(stream):
                try:
                    # Collect features
                    features_list = []
                    valid_messages = []
                    
                    for msg in messages:
                        if msg['artifact']['features'] is not None:
                            features = msg['artifact']['features']
                            features_list.append(features)
                            valid_messages.append(msg)
                    
                    if not features_list:
                        continue
                    
                    # Stack features and move to GPU in this stream
                    batch_features = torch.stack(features_list).cuda(non_blocking=True)
                    
                    # Get novelty scores
                    novelty_scores = agent.knn.batch_get_novelty_stream(batch_features, stream)
                    
                    # Store results for later processing
                    results[agent_id] = {
                        'agent': agent,
                        'messages': valid_messages,
                        'novelty_scores': novelty_scores
                    }
                    
                except Exception as e:
                    print(f"Error in stream processing for agent {agent_id}: {e}")
        
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Process results after all computations are done
        for agent_id, result in results.items():
            agent = result['agent']
            valid_messages = result['messages']
            novelty_scores = result['novelty_scores']
            
            try:
                # Process each message with its novelty score
                for msg, novelty in zip(valid_messages, novelty_scores):
                    interest = agent.hedonic_evaluation(novelty.item())
                    
                    if interest > self.domain_threshold:
                        features_to_add = msg['artifact']['features']
                        agent.knn.add_feature_vectors(features_to_add)
                        domain_entry = {
                            'artifact': msg['artifact'],
                            'creator_id': msg['sender_id'],
                            'evaluator_id': agent_id,
                            'novelty': novelty.item(),
                            'interest': interest,
                            'timestamp': self.schedule.time
                        }
                        self.add_to_domain(domain_entry)
                    
                    agent.average_interest_other.append(interest)
                    
            except Exception as e:
                print(f"Error processing results for agent {agent_id}: {e}")
            
            # Clear processed messages
            agent.inbox = []
        
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
    
    def step(self):
        # First process all inboxes in batch
        self.process_inboxes_parallel()
        self.calculate_novelty_threshold()
        self.schedule.step()
            
def run_simulation(num_agents=100, steps=10000):
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