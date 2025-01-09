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
from wundtcurve import WundtCurve
from timing_utils import time_it, TimingStats
from image_saver import ImageSaver

log_dir = f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v1"

class Agent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.knn = kNN(agent_id=unique_id)
        
        self.average_interest_self = []
        self.average_interest_other = []
        
        self.generated_image = None
        
        self.generated_expression = None
        
        
        # Generation parameters
        self.gen_depth = np.random.randint(4, 10)
        self.current_expression = None
        self.artifact_memory = []
        
        # Initialize with normally distributed novelty preference
        self.preferred_novelty = np.random.normal(0.5, 0.155)
        self.preferred_novelty = np.clip(self.preferred_novelty, 0, 1)
        
        # Set means around preferred novelty
        reward_mean = max(0.1, self.preferred_novelty - 0.2)
        punish_mean = min(0.9, self.preferred_novelty + 0.2)
        
        self.wundt = WundtCurve(
            reward_mean=reward_mean,  # Reward mean
            reward_std=0.15,          # Reward std dev
            punish_mean=punish_mean,  # Punishment mean
            punish_std=0.15,          # Punishment std dev
            alpha=1.2                 # Punishment weight
        )
        
        self.alpha = 0.35
        self.average_interest = 0.0
        
        self.boredom_threshold = 0.2
        
        self.init_plot = False
        
        # Communication
        self.inbox = []
        
    @time_it
    def hedonic_evaluation(self, novelty):
        """
        Compute hedonic value from novelty using paper's Wundt curve
        Returns value between -1 and 1
        """
        try:
            # Ensure novelty is a float and properly bounded
            novelty = float(novelty)
            novelty = max(0.0, min(1.0, novelty))
            
            # Get hedonic value from Wundt curve
            hedonic_value = self.wundt.hedonic_value(novelty)
            
            # Ensure we have a valid numeric value
            if hedonic_value is None or not np.isfinite(hedonic_value):
                print(f"Warning: Invalid hedonic value for agent {self.unique_id}, novelty: {novelty}")
                hedonic_value = 0.0
                
            # Update running average with type checking
            if not hasattr(self, 'average_interest'):
                self.average_interest = 0.0
                
            self.average_interest = float(self.alpha * self.average_interest + (1 - self.alpha) * hedonic_value)
            
            return hedonic_value
            
        except Exception as e:
            print(f"Error in hedonic evaluation for agent {self.unique_id}: {e}")
            return 0.0  # Safe fallback

    @time_it
    def log_metrics(self, artifact_data=None):
        """Enhanced logging with dedicated agent cards and image saving"""
        step = self.model.schedule.time
        writer = self.model.agent_writers[self.unique_id]
        
        if artifact_data is None:
            return
            
        if step % 100 == 0 or step == 1 or step == 2 or step == 3:  
            # Handle PIL Image directly - artifact_data is the image itself
            if isinstance(artifact_data, Image.Image):
                image = artifact_data
                # Convert PIL image to tensor
                image_tensor = transforms.ToTensor()(image)
                writer.add_image('generated/current_image', image_tensor, step)
                
            # Handle expression if we have it
            if hasattr(self, 'generated_expression'):
                writer.add_text('generated/expression', self.generated_expression, step)
                
            # Metrics logging
            if hasattr(self, 'average_interest_self') and self.average_interest_self:
                writer.add_scalar('evaluation/interest_self', 
                                float(np.mean(self.average_interest_self[-100:])), step)
                
            if hasattr(self, 'average_interest_other') and self.average_interest_other:
                writer.add_scalar('evaluation/interest_other',
                                float(np.mean(self.average_interest_other[-100:])), step)

        if not self.init_plot:
            self.init_plot = True
            try:
                # Add custom figure showing agent's state
                fig = plt.figure(figsize=(10, 6))
                plt.subplot(121)
                plt.title(f"Agent {self.unique_id} Wundt Curve")
                x = np.linspace(0, 1, 100)
                y = [self.hedonic_evaluation(float(xi)) for xi in x]
                plt.plot(x, y)
                plt.grid(True)
                
                # Convert figure to tensor safely
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                writer.add_image('state/wundt_curve', transforms.ToTensor()(img), step)
                plt.close(fig)
                buf.close()
                
                # Agent parameters
                writer.add_scalar('parameters/preferred_novelty', float(self.preferred_novelty), step)
            except Exception as e:
                print(f"Error in plot generation for agent {self.unique_id}: {e}")
            
    @time_it
    def step(self):
        """One step of agent behavior focused on boredom and logging"""
        # Update accumulated interest with more detailed logging
        writer = self.model.agent_writers[self.unique_id]
        writer.add_scalar('interest/average', self.average_interest, self.model.schedule.time)
        writer.add_scalar('agents/last_k', self.knn.k, self.model.schedule.time)
        
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
                domain_novelty = self.model.normalise_novelty(domain_novelty)
                domain_interest = self.hedonic_evaluation(domain_novelty)
                
                # Log domain interaction
                writer.add_scalar('domain_interaction/novelty_from_domain', domain_novelty, self.model.schedule.time)
                writer.add_scalar('domain_interaction/interest_from_domain', domain_interest, self.model.schedule.time)
                
                # If interesting enough, adopt it
                if domain_interest > self.average_interest:
                    self.current_expression = domain_artifact['expression']
                    self.knn.add_feature_vectors(domain_features, self.model.schedule.time)
                    self.average_interest = domain_novelty
                    writer.add_scalar('domain_interaction/cumulative_adoptions', 
                            self.model.agent_adoption_counts[self.unique_id], 
                            self.model.schedule.time)
                    self.model.agent_adoption_counts[self.unique_id] += 1
        
        # Log final metrics
        self.log_metrics(self.generated_image if hasattr(self, 'generated_image') else None)


class Model(mesa.Model):
    def __init__(self, number_agents=100):
        super().__init__()
        self.num_agents = number_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.feature_extractor = FeatureExtractor(output_dims=64)
        self.image_generator = genart.ImageGenerator(32, 32)
        
        self.agent_adoption_counts = np.zeros(number_agents)
        
        self.amount_shares = 10
        
        self.bottom_1_percentile_novelty = 0
        self.top_99_percentile_novelty = 1

        # Thresholds
        self.self_threshold = None  # When to share
        self.domain_threshold = None  # When to add to domain

        self.novelty_values = []
        self.novelty_values_normalized = []
        
        self.interest_threshold_self_list = []
        self.interest_threshold_other_list = []
        
        self.communication_matrix = np.zeros((number_agents, number_agents))
        self.interaction_timestamps = np.zeros((number_agents, number_agents))  # Track when interactions occur
        self.successful_interactions = np.zeros((number_agents, number_agents))

        # Domain repository
        self.domain = []
        self.max_domain_size = 10000000
        
        self.image_saver = ImageSaver()
        
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
            
        # Connection tracking matrices
        self.connection_matrix = np.zeros((number_agents, number_agents))
        self.cumulative_matrix = np.zeros((number_agents, number_agents))
        
        # Agent contribution tracking 
        self.domain_contributions = np.zeros(number_agents)
        self.agent_contribution_history = {i: [] for i in range(number_agents)}
        self.agent_success_rates = np.zeros(number_agents)
        
    def update_connection_strength(self, sender_id, receiver_id, success=True, timestamp=None):
        """Update connection strength with proper temporal tracking"""
        if timestamp is None:
            timestamp = self.schedule.time
            
        # Update raw interaction count
        self.connection_matrix[sender_id][receiver_id] += 1
        
        if success:
            # Track successful interactions
            self.successful_interactions[sender_id][receiver_id] += 1
            # Update timestamp of last successful interaction
            self.interaction_timestamps[sender_id][receiver_id] = timestamp
            
            # Log success rate
            total_attempts = self.connection_matrix[sender_id][receiver_id]
            success_rate = self.successful_interactions[sender_id][receiver_id] / total_attempts
            self.writer.add_scalar(f'connections/success_rate_{sender_id}_{receiver_id}', 
                             success_rate, timestamp)
    @time_it   
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

    @time_it
    def add_to_domain(self, entry):
        """Enhanced domain addition with async I/O"""
        if len(self.domain) >= self.max_domain_size:
            self.domain.pop(0)
        
        artifact = entry['artifact']
        creator_id = entry['creator_id']
        self.domain_contributions[creator_id] += 1
        self.agent_contribution_history[creator_id].append(self.schedule.time)
        
        self.writer.add_scalar(f'agents/contributions/agent_{creator_id}', 
                            self.domain_contributions[creator_id],
                            self.schedule.time)
        
        # Generate image filename
        image_filename = f"domain_id-{len(self.domain)}_creator-{entry['creator_id']}_evaluator-{entry['evaluator_id']}_nov-{entry['novelty']}_int-{entry['interest']}.png"
        image_filename = image_filename.replace(" ", "_").replace("/", "-")
        image_path = f"{self.image_dir}/{image_filename}"
        
        # Queue image save instead of saving immediately
        self.image_saver.queue_image_save(artifact['image'], image_path)
        
        # If queue is getting full, process some saves
        if len(self.image_saver.image_save_queue) >= self.image_saver.max_queue_size:
            self.image_saver.process_save_queue()
        
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
        
    @time_it
    def get_random_domain_artifact(self):
        """Return random artifact from domain if available"""
        if not self.domain:
            return None
            
        # Try up to 5 times to get a saved image
        for _ in range(5):
            domain_entry = random.choice(self.domain)
            image_path = domain_entry['image_path']
            
            # Check if image is ready
            if self.image_saver.is_image_ready(image_path):
                try:
                    return {
                        'features': self.feature_extractor.extract_features(Image.open(image_path)),
                        'expression': genart.ExpressionNode.from_string(domain_entry['expression']),
                        'image': Image.open(image_path)
                    }
                except Exception as e:
                    print(f"Error loading artifact image {image_path}: {e}")
                    continue
        
        # If we couldn't find a ready image after 5 tries, return None
        return None
    
    @time_it
    def process_inboxes_parallel(self):
        """Optimized parallel inbox processing using batch operations and memory pinning"""
        if not torch.cuda.is_available():
            self.process_inboxes_batch()
            return

        # Pre-allocate lists for batch processing
        all_features = []
        all_messages = []
        agent_indices = []
        
        # Collect all messages in a single pass
        for agent_idx, agent in enumerate(self.schedule.agents):
            if not agent.inbox:
                continue
                
            for msg in agent.inbox:
                if msg['artifact']['features'] is not None:
                    all_features.append(msg['artifact']['features'])
                    all_messages.append((agent_idx, msg))
                    agent_indices.append(agent_idx)
        
        if not all_features:
            return

        try:
            # Convert to contiguous tensor and pin memory
            batch_features = torch.stack(all_features).contiguous()
            if batch_features.device.type == 'cpu':
                batch_features = batch_features.pin_memory()
                
            # Create a single CUDA stream for batch processing
            stream = torch.cuda.Stream()
            
            with torch.cuda.stream(stream):
                # Move entire batch to GPU at once
                batch_features = batch_features.cuda(non_blocking=True)
                
                # Process all novelty scores in one batch
                novelty_scores = {}
                for agent_idx in set(agent_indices):
                    agent = self.schedule.agents[agent_idx]
                    mask = torch.tensor([i == agent_idx for i in agent_indices], 
                                    device=batch_features.device)
                    agent_features = batch_features[mask]
                    
                    if len(agent_features) > 0:
                        scores = agent.knn.batch_get_novelty_stream(agent_features, stream)
                        novelty_scores[agent_idx] = scores
                
            # Synchronize after all GPU operations
            stream.synchronize()

            # Process results in batches
            for agent_idx, agent in enumerate(self.schedule.agents):
                if agent_idx not in novelty_scores:
                    agent.inbox = []  # Clear processed inbox
                    continue
                    
                scores = novelty_scores[agent_idx]
                relevant_messages = [msg for idx, msg in all_messages if idx == agent_idx]
                
                # Pre-allocate features for addition
                features_to_add = []
                domain_entries = []
                
                for msg, novelty in zip(relevant_messages, scores):
                    novelty_score = self.normalise_novelty(novelty.item())
                    interest = agent.hedonic_evaluation(novelty_score)
                    
                    if interest > self.domain_threshold:
                        features_to_add.append(msg['artifact']['features'])
                        domain_entries.append({
                            'artifact': msg['artifact'],
                            'creator_id': msg['sender_id'],
                            'evaluator_id': agent_idx,
                            'novelty': novelty.item(),
                            'interest': interest,
                            'timestamp': self.schedule.time
                        })
                    
                    agent.average_interest_other.append(interest)
                
                # Batch add features
                if features_to_add:
                    features_tensor = torch.stack(features_to_add)
                    agent.knn.add_feature_vectors(features_tensor, self.schedule.time)
                    
                    # Batch add to domain
                    for entry in domain_entries:
                        self.add_to_domain(entry)
                
                agent.inbox = []  # Clear processed inbox

        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Clear all inboxes to prevent message accumulation on error
            for agent in self.schedule.agents:
                agent.inbox = []
    
    @time_it
    def process_generations_parallel(self):
        """Batch process art generation using CUDA streams with proper type handling"""
        if not torch.cuda.is_available():
            return
                
        # Create lists for batch processing
        expressions = []
        agents = []
        
        # Debug logging
        #print(f"Initial device check - CUDA available: {torch.cuda.is_available()}")
        
        # Collect all expressions that need to be generated
        for agent in self.schedule.agents:
            if not agent.current_expression:
                agent.current_expression = genart.ExpressionNode.create_random(depth=agent.gen_depth)
            else:
                if agent.artifact_memory:
                    parent = random.choice(agent.artifact_memory)
                    if isinstance(parent, dict):
                        parent = parent['expression']
                    agent.current_expression = agent.current_expression.breed(parent)
            
            expressions.append(agent.current_expression)
            agents.append(agent)
        
        try:

            # Create multiple CUDA streams for pipeline parallelism
            num_streams = min(4, torch.cuda.device_count())
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            #print(f"Created {num_streams} CUDA streams")
            
            # Split into batches
            batch_size = max(1, len(expressions) // num_streams)
            batches = [expressions[i:i + batch_size] for i in range(0, len(expressions), batch_size)]
            agent_batches = [agents[i:i + batch_size] for i in range(0, len(agents), batch_size)]
            
            results = []
            
            # Process each batch in parallel streams
            for stream_idx, (expr_batch, agent_batch) in enumerate(zip(batches, agent_batches)):
                #print(f"Processing batch {stream_idx + 1}/{len(batches)}")
                with torch.cuda.stream(streams[stream_idx]):
                    # Step 1: Generate coordinates batch
                    coords_batch = self.image_generator.coords.data.expand(len(expr_batch), -1, -1, -1)
                    coords_batch = coords_batch.to(dtype=torch.float32, device='cuda')
                    
                    # Debug coords batch
                    #print(f"Coords batch - shape: {coords_batch.shape}, dtype: {coords_batch.dtype}")
                    
                    # Step 2: Evaluate expressions in batch
                    evaluated_batch = []
                    for expr, coords in zip(expr_batch, coords_batch):
                        quat_coords = genart.QuaternionTensor(coords)
                        result = expr.evaluate(quat_coords)
                        evaluated_batch.append(result.data)
                    
                    evaluated_tensor = torch.stack(evaluated_batch).to(dtype=torch.float32)
                    #print(f"Evaluated tensor - shape: {evaluated_tensor.shape}, dtype: {evaluated_tensor.dtype}")
                    
                    # Step 3: Convert to RGB in batch with explicit type casting
                    rgb_batch = []
                    for quat_data in evaluated_tensor:
                        quat = genart.QuaternionTensor(quat_data)
                        rgb = quat.to_rgb()  # This returns uint8 tensor
                        rgb_batch.append(rgb)
                    
                    rgb_tensor = torch.stack(rgb_batch)
                    #print(f"RGB tensor - shape: {rgb_tensor.shape}, dtype: {rgb_tensor.dtype}")
                    
                    # Step 4: Convert to float32 for feature extraction
                    rgb_float = rgb_tensor.to(dtype=torch.float32) / 255.0  # Normalize to [0,1]
                    rgb_float = rgb_float.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    
                    # Step 5: Extract features
                    features_batch = self.feature_extractor(rgb_float)
                    #print(f"Features batch - shape: {features_batch.shape}, dtype: {features_batch.dtype}")
                    
                    # Step 6: Create PIL images (CPU operation, do last)
                    images = [Image.fromarray(rgb.cpu().numpy()) for rgb in rgb_tensor]
                    
                    # Store results
                    batch_results = []
                    for expr, img, feat in zip(expr_batch, images, features_batch):
                        batch_results.append({
                            'image': img,
                            'features': feat.detach(),  # Detach from computation graph
                            'expression': expr
                        })
                    results.extend(batch_results)
            
            # Synchronize all streams
            torch.cuda.synchronize()
            #print("All CUDA streams synchronized")
            
            # Assign results back to agents
            for agent, result in zip(agents, results):
                agent.generated_image = result['image']
                agent.generated_expression = result['expression'].to_string()
                
                # Add to personal experience/memory
                features = result['features']
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)
                elif len(features.shape) == 3:
                    features = features.squeeze(1)
                    
                agent.knn.add_feature_vectors(features, self.schedule.time)
                agent.artifact_memory.append(result)
                
                # Calculate novelty and interest
                novelty_scores = agent.knn.batch_get_novelty(features)
                novelty = novelty_scores[0].item()
                novelty = self.normalise_novelty(novelty)
                interest = agent.hedonic_evaluation(novelty)
                
                result['novelty'] = novelty
                result['interest'] = interest
                agent.average_interest_self.append(interest)
                
                # Handle sharing if interesting enough
                if interest > self.self_threshold:
                    message = {
                        'artifact': result,
                        'sender_id': agent.unique_id,
                        'timestamp': self.schedule.time
                    }
                    
                    other_agents = [a for a in self.schedule.agents 
                                if a.unique_id != agent.unique_id]
                    if other_agents:
                        for _ in range(self.amount_shares):
                            recipient = random.choice(other_agents)
                            recipient.inbox.append(message)
                        
        except Exception as e:
            print(f"Error in parallel generation: {e}")
            torch.cuda.empty_cache()  # Clear GPU memory
            # Print CUDA memory stats
            #if torch.cuda.is_available():
                #print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                #print(f"CUDA memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            # Provide fallback generation if needed
            for agent in agents:
                if not hasattr(agent, 'generated_image'):
                    fallback = agent.generate_artifact()
                    agent.generated_image = fallback['image']
                    agent.generated_expression = fallback['expression'].to_string()
            
    @time_it
    def update_novelty_bounds(self):
        """Update the percentile bounds using a rolling window of novelty values"""
        self.bottom_1_percentile_novelty = np.percentile(self.novelty_values, 1)
        self.top_99_percentile_novelty = np.percentile(self.novelty_values, 99)

    @time_it
    def normalise_novelty(self, novelty):
        """
        Normalize novelty and maintain a rolling window of recent values
        """
        MAX_HISTORY = 10000  # Keep last 10k values
        
        # Add the new novelty value and trim if needed
        self.novelty_values.append(novelty)
        if len(self.novelty_values) > MAX_HISTORY:
            self.novelty_values = self.novelty_values[-MAX_HISTORY:]

        # Normalize using current bounds
        normalized_novelty = (novelty - self.bottom_1_percentile_novelty) / (self.top_99_percentile_novelty - self.bottom_1_percentile_novelty + 1e-8)
        normalized_novelty = np.clip(normalized_novelty, 0, 1)
        
        return normalized_novelty

    @time_it    
    def _initialize_agents(self):
        """Create agents with normally distributed novelty preferences"""
        for i in range(self.num_agents):
            agent = Agent(i, self)
            self.schedule.add(agent)
    @time_it
    def log_system_metrics(self):
        """Enhanced system-wide metrics logging"""
        step = self.schedule.time
        
        # Domain metrics
        self.writer.add_scalar("domain/size", len(self.domain), step)
        
        # Enhanced top contributors logging
        if step % 25 == 0:  # Update every 10 steps
            top_contributors = np.argsort(self.domain_contributions)[-8:][::-1]
            for rank, agent_id in enumerate(top_contributors):
                contrib_count = len(self.agent_contribution_history[agent_id])
                self.writer.add_text(f'top_contributors/rank_{rank}', 
                                f'Agent {agent_id} (Total contributions: {contrib_count})', 
                                step)
                self.writer.add_scalar(f'top_contributors/agent_{agent_id}/contribution_count', 
                                    contrib_count,
                                    step)
                self.writer.add_scalar(f'top_contributors/agent_{agent_id}/current_rank', 
                                    rank,
                                    step)
    @time_it
    def step(self):
        # First process all inboxes in batch
        self.image_saver.process_save_queue()
        self.process_inboxes_parallel()
        self.process_generations_parallel()
        self.calculate_novelty_threshold()
        
        if self.schedule.time % 5 == 0 or self.schedule.time == 1 or self.schedule.time == 2:
            self.update_novelty_bounds
        
        self.novelty_calculated = False
        
        if self.schedule.time % 1 == 0:
            # Log overall network density
            connections = np.count_nonzero(self.connection_matrix)
            total_possible = self.num_agents * (self.num_agents - 1)
            network_density = connections / total_possible
            self.writer.add_scalar('network/density', network_density, self.schedule.time)
            
            # Log top contributing agents
            top_contributors = np.argsort(self.domain_contributions)[-5:][::-1]
            for rank, agent_id in enumerate(top_contributors):
                self.writer.add_scalar(f'top_contributors/rank_{rank}', 
                                    agent_id,
                                    self.schedule.time)
                self.writer.add_scalar(f'top_contributors/score_{rank}', 
                                    self.domain_contributions[agent_id],
                                    self.schedule.time)
            
            # Log overall connection strengths
            self.writer.add_scalar('network/total_connections', 
                                np.sum(self.connection_matrix),
                                self.schedule.time)
            self.writer.add_scalar('network/average_strength',
                                np.mean(self.connection_matrix[self.connection_matrix > 0]),
                                self.schedule.time)

        self.schedule.step()
            
def run_simulation(num_agents=1000, steps=5000):
    """Run simulation with proper cleanup"""
    print(f"\nStarting simulation with {num_agents} agents for {steps} steps")
    model = Model(num_agents)
    
    with tqdm(total=steps, desc="Simulation Progress") as pbar:
        for step in range(steps):
            print(f"\nStep {step+1}/{steps}")
            TimingStats().reset_step()
            model.step()
            TimingStats().print_step_report()
            pbar.update(1)
    # Proper cleanup of all writers
    model.writer.close()
    for writer in model.agent_writers.values():
        writer.close()
    
    print("\nSimulation completed!")
        
if __name__ == "__main__":
    run_simulation()