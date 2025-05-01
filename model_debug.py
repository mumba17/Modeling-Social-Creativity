import logging
import os
import io
import csv
import datetime
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import torchvision.transforms as transforms
import mesa
import sys

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from knn import kNN
from features import FeatureExtractor
import genart
from wundtcurve import WundtCurve
from timing_utils import time_it, TimingStats
from image_saver import ImageSaver
from network_tracker import NetworkTracker
from logging_utils import setup_logger, log_event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Agent(mesa.Agent):
    """
    A Mesa agent representing a creative individual. The Agent can:
      - Generate artifacts using 'genart' code
      - Evaluate artifacts via a Wundt curve
      - Track novelty with a kNN instance
      - Communicate artifacts to other agents
      - Possibly adopt artifacts from the domain
    """

    def __init__(self, unique_id, model):
        """
        Initialize the agent with random parameters, WundtCurve for hedonic evaluations,
        kNN for novelty tracking, etc.

        Parameters
        ----------
        unique_id : int
            Unique identifier for this agent
        model : Model
            Reference to the Mesa Model instance
        """
        print(f"Initializing Agent {unique_id}...")
        super().__init__(unique_id, model)
        self.unique_id = unique_id

        # kNN novelty tracker
        self.knn = kNN(agent_id=unique_id)

        # Arrays for storing recent interest
        self.average_interest_self = []
        self.average_interest_other = []

        # Placeholders for the last generated artifact
        self.generated_image = None
        self.generated_expression = None

        # Generation parameters
        self.gen_depth = np.random.randint(config.INIT_GEN_DEPTH_MIN, config.INIT_GEN_DEPTH_MAX)
        self.current_expression = None
        self.artifact_memory = []

        # Initialize novelty preference
        self.preferred_novelty = np.random.normal(0.5, 0.155)
        self.preferred_novelty = np.clip(self.preferred_novelty, 0, 1)

        # Wundt curve parameters revolve around the preferred novelty
        reward_mean = max(0.1, self.preferred_novelty - 0.2)
        punish_mean = min(0.9, self.preferred_novelty + 0.2)
        self.wundt = WundtCurve(
            reward_mean=reward_mean,
            reward_std=config.WUNDT_REWARD_STD,
            punish_mean=punish_mean,
            punish_std=config.WUNDT_PUNISH_STD,
            alpha=config.WUNDT_ALPHA
        )

        # Decay factor for interest
        self.alpha = config.ALPHA
        self.average_interest = 0.0

        # Boredom threshold (dynamic per step in Model)
        self.boredom_threshold = config.BOREDOM_THRESHOLD

        # For controlling initial plot logging
        self.init_plot = False

        # Communication inbox
        self.inbox = []
        print(f"Agent {unique_id} initialized successfully")

    @time_it
    def hedonic_evaluation(self, novelty):
        """
        Compute hedonic value from novelty using the Wundt curve.
        Returns a value between -1 and 1.

        Parameters
        ----------
        novelty : float
            Novelty score in [0, 1]

        Returns
        -------
        float
            Hedonic (pleasantness) evaluation in [-1, 1].
        """
        try:
            novelty = float(novelty)
            novelty = max(0.0, min(1.0, novelty))

            # Get hedonic value from Wundt curve
            hedonic_value = self.wundt.hedonic_value(novelty)
            if hedonic_value is None or not np.isfinite(hedonic_value):
                logger.warning(f"Invalid hedonic value for agent {self.unique_id}, novelty: {novelty}")
                hedonic_value = 0.0

            if not hasattr(self, 'average_interest'):
                self.average_interest = 0.0

            # Update running average
            self.average_interest = float(
                self.alpha * self.average_interest + (1 - self.alpha) * hedonic_value
            )
            return hedonic_value

        except Exception as e:
            logger.error(f"Error in hedonic evaluation for agent {self.unique_id}: {e}")
            return 0.0  # Safe fallback

    @time_it
    def step(self):
        """
        Single step of agent behavior. Logs interest, checks for boredom,
        possibly adopts from domain. Agents also handle final metrics logging
        at the end.
        """
        # Log agent's current interest using the new log_event
        log_event(
            step=self.model.schedule.time,
            event_type='agent_step_summary',
            agent_id=self.unique_id,
            details={
                'average_interest': self.average_interest,
                'interaction_count': self.model.communication_matrix[self.unique_id].sum(),
                'domain_contributions': self.model.domain_contributions[self.unique_id]
            }
        )

        # If average_interest is below boredom threshold, try adopting from domain
        if self.average_interest < self.model.boredom_threshold:
            domain_artifact = self.model.get_random_domain_artifact()
            if domain_artifact:
                domain_features = domain_artifact['features']

                # Ensure shape correctness
                if len(domain_features.shape) == 1:
                    domain_features = domain_features.unsqueeze(0)
                elif len(domain_features.shape) == 3:
                    domain_features = domain_features.squeeze(1)

                domain_novelty_scores = self.knn.batch_get_novelty(domain_features)
                domain_novelty = domain_novelty_scores[0].item()
                domain_novelty_raw = domain_novelty # Keep raw value
                domain_novelty = self.model.normalise_novelty(domain_novelty) # Use normalized novelty
                domain_interest = self.hedonic_evaluation(domain_novelty)

                # Log domain interaction evaluation using the new log_event
                log_event(
                    step=self.model.schedule.time,
                    event_type='domain_interaction_evaluation',
                    agent_id=self.unique_id,
                    details={
                        'source_artifact_expression': domain_artifact['expression'].to_string(), # Log expression
                        'source_artifact_image_path': domain_artifact.get('image_path'), # Log image path
                        'raw_novelty': domain_novelty_scores[0].item(), # Log raw novelty too
                        'normalized_novelty': domain_novelty,
                        'calculated_interest': domain_interest,
                        'average_interest_before': self.average_interest, # Log context
                        'boredom_threshold': self.model.boredom_threshold, # Log threshold
                        'adopted': domain_interest > self.average_interest # Log decision
                    }
                )

                # Possibly adopt if more interesting than current average
                if domain_interest > self.average_interest:
                    # Log the adoption event itself
                    log_event(
                        step=self.model.schedule.time,
                        event_type='domain_adoption',
                        agent_id=self.unique_id,
                        details={
                            'adopted_expression': domain_artifact['expression'].to_string(),
                            'adopted_image_path': domain_artifact.get('image_path'), # Log image path
                            'adopted_raw_novelty': domain_novelty_scores[0].item(),
                            'adopted_normalized_novelty': domain_novelty,
                            'adopted_interest': domain_interest,
                            'previous_average_interest': self.average_interest
                        }
                    )
                    self.current_expression = domain_artifact['expression']
                    self.knn.add_feature_vectors(domain_features, self.model.schedule.time)
                    self.average_interest = domain_interest # Update average interest with the adopted interest
                    self.model.agent_adoption_counts[self.unique_id] += 1


class Model(mesa.Model):
    """
    Main Mesa model for simulating computational social creativity using multiple agents.
    Agents generate artifacts, share them, evaluate novelty and interest,
    and store accepted artifacts in a domain.
    """

    def __init__(self,
                 number_agents=config.NUMBER_AGENTS,
                 output_dims=config.OUTPUT_DIMS,
                 log_dir=None):
        """
        Initialize the model with specified parameters.

        Parameters
        ----------
        number_agents : int
            Number of agents to create
        output_dims : int
            Output feature dimension for ResNet-based feature extraction
        log_dir : str, optional
            Logging directory for TensorBoard and CSV outputs.
        """
        print("Starting Model initialization...")
        super().__init__()
        self.num_agents = number_agents
        self.output_dims = output_dims
        self.schedule = mesa.time.RandomActivation(self)

        # Logging directories
        self.log_dir = log_dir or f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v2"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Created log directory: {self.log_dir}")

        # Initialize system-wide TensorBoard writer
        print("Initializing TensorBoard writer...")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print("TensorBoard writer initialized")

        # Set up logging
        print("Setting up JSON logger...")
        try:
            self.json_logger = setup_logger(self.log_dir)
            print("JSON logger initialized successfully")
        except Exception as e:
            print(f"ERROR in JSON logger setup: {e}")
            logger.warning(f"Error initializing JSON logger: {e}. Falling back to simple logging.")
            self.json_logger = logging.getLogger('simulation')
            handler = logging.FileHandler(os.path.join(self.log_dir, "simulation.log"))
            self.json_logger.addHandler(handler)

        # Initialize network tracker before heavy components
        print("Initializing network tracker...")
        self.network_tracker = NetworkTracker(number_agents, self.log_dir)
        print("Network tracker initialized")

        # Initialize image saver with proper directory structure  
        print("Initializing image saver...")      
        try:
            self.image_saver = ImageSaver(self.log_dir)
            self.image_dir = os.path.join(self.log_dir, "images") 
            print(f"Image saver initialized successfully with directory: {self.image_dir}")
        except Exception as e:
            print(f"ERROR in image saver initialization: {e}")
            logger.warning(f"Error initializing structured image saver: {e}")
            # Fall back to simple directory structure
            self.image_saver = ImageSaver()
            self.image_dir = os.path.join(self.log_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)

        # For logging agent adoptions
        self.agent_adoption_counts = np.zeros(number_agents)

        # Communication parameters
        self.amount_shares = 10  # Can be changed dynamically if needed

        # Novelty normalization bounds
        self.bottom_1_percentile_novelty = 0
        self.top_99_percentile_novelty = 1

        # Initialize thresholds with default values
        self.self_threshold = 0.1  # Default value
        self.domain_threshold = 0.1  # Default value 
        self.boredom_threshold = config.BOREDOM_THRESHOLD

        # Novelty distributions
        self.novelty_values = []
        self.novelty_values_normalized = []

        # Domain and interest tracking
        self.domain = []
        self.max_domain_size = config.MAX_DOMAIN_SIZE
        self.domain_contributions = np.zeros(number_agents)
        self.agent_contribution_history = {i: [] for i in range(number_agents)}
        self.agent_success_rates = np.zeros(number_agents)

        # Communication matrix
        self.communication_matrix = np.zeros((number_agents, number_agents))
        self.interaction_timestamps = np.zeros((number_agents, number_agents))
        self.successful_interactions = np.zeros((number_agents, number_agents))

        # Feature extraction, image generation - these are heavy operations
        print("Initializing feature extractor (this might take a while)...")
        sys.stdout.flush()  # Ensure print is displayed immediately
        self.feature_extractor = FeatureExtractor(output_dims=self.output_dims)
        print("Feature extractor initialized successfully")
        
        print("Initializing image generator...")
        sys.stdout.flush()  # Ensure print is displayed immediately
        self.image_generator = genart.ImageGenerator(32, 32)
        print("Image generator initialized successfully")

        # Create agents
        print(f"Creating {number_agents} agents...")
        self._initialize_agents()
        print("All agents initialized successfully")

        # For Mesa run loop
        self.running = True
        print("Model initialization complete!")

    def _initialize_agents(self):
        """
        Create and add agents to the model schedule.
        """
        for i in range(self.num_agents):
            if i % 10 == 0:  # Print status every 10 agents
                print(f"Initializing agent {i}/{self.num_agents}...")
                sys.stdout.flush()
            agent = Agent(i, self)
            self.schedule.add(agent)

    @time_it
    def update_connection_strength(self, sender_id, receiver_id, success=True, timestamp=None):
        """
        Update network tracker with a new interaction event.

        Parameters
        ----------
        sender_id : int
            ID of the sending agent
        receiver_id : int
            ID of the receiving agent
        success : bool
            Whether the artifact was accepted
        timestamp : int, optional
            Simulation step/time of this interaction
        """
        if timestamp is None:
            timestamp = self.schedule.time

        self.network_tracker.record_interaction(
            sender_id=sender_id,
            receiver_id=receiver_id,
            accepted=success,
            timestamp=timestamp
        )

    @time_it
    def calculate_novelty_threshold(self):
        """
        Calculate thresholds for:
          - Self threshold (when to share)
          - Domain threshold (when to add to domain)
          - Boredom threshold (when an agent seeks new artifacts)

        Based on agent interest distributions in the current step.
        """
        window_size = config.WINDOW_SIZE

        step_interests_self = [
            np.mean(agent.average_interest_self[-window_size:])
            for agent in self.schedule.agents if agent.average_interest_self
        ]
        step_interests_other = [
            np.mean(agent.average_interest_other[-window_size:])
            for agent in self.schedule.agents if agent.average_interest_other
        ]
        current_interests = [agent.average_interest for agent in self.schedule.agents]

        if step_interests_self:
            self.self_threshold = np.percentile(step_interests_self, 80)
        else:
            self.self_threshold = 0.1

        if step_interests_other:
            self.domain_threshold = np.percentile(step_interests_other, 80)
        else:
            self.domain_threshold = 0.1

        if current_interests:
            self.boredom_threshold = np.percentile(current_interests, 10)
        else:
            self.boredom_threshold = config.BOREDOM_THRESHOLD

        # Log system-wide threshold stats in TensorBoard
        step = self.schedule.time
        self.writer.add_scalar('thresholds/communication', self.self_threshold, step)
        self.writer.add_scalar('thresholds/domain', self.domain_threshold, step)
        self.writer.add_scalar('thresholds/boredom', self.boredom_threshold, step)

        if current_interests:
            self.writer.add_scalar('interest/mean', np.mean(current_interests), step)
            self.writer.add_scalar('interest/median', np.median(current_interests), step)
            self.writer.add_scalar('interest/std', np.std(current_interests), step)

            bored_agents = sum(1 for i in current_interests if i < self.boredom_threshold)
            bored_percentage = (bored_agents / len(current_interests)) * 100
            self.writer.add_scalar('agents/bored_percentage', bored_percentage, step)

    @time_it
    def add_to_domain(self, domain_entry):
        """
        Add a new artifact to the domain repository.
        """
        print(f"Adding artifact to domain (current size: {len(self.domain)})")
        if len(self.domain) >= self.max_domain_size:
            self.domain.pop(0)

        artifact = domain_entry['artifact']
        creator_id = domain_entry['creator_id']
        self.domain_contributions[creator_id] += 1
        
        # Save image with semantic metadata
        metadata = {
            'creator': domain_entry['creator_id'],
            'evaluator': domain_entry['evaluator_id'],
            'nov': f"{domain_entry['novelty']:.3f}",
            'int': f"{domain_entry['interest']:.3f}",
            'step': domain_entry['timestamp']
        }
        
        print(f"Queueing domain image save with metadata: {metadata}")
        # Queue the image save with metadata
        image_path = None
        if 'image' in artifact:
            image_path = self.image_saver.queue_domain_image_save(
                artifact['image'],
                metadata=metadata
            )
            print(f"Domain image queued with path: {image_path}")
        else:
            print("No image in artifact to save")
            
        # Store the path in the domain entry and remove the image object to save memory
        if image_path:
            artifact['image_path'] = image_path
            if 'image' in artifact:
                del artifact['image']
        
        self.domain.append(domain_entry)
        
        # Log domain update in TensorBoard
        self.writer.add_scalar('domain/total_size', len(self.domain), self.schedule.time)
        self.writer.add_scalar('domain/last_interest', domain_entry['interest'], self.schedule.time)
        self.writer.add_scalar('domain/last_novelty', domain_entry['novelty'], self.schedule.time)
        print(f"Domain updated, new size: {len(self.domain)}")

    @time_it
    def get_random_domain_artifact(self):
        """
        Return a random artifact from the domain if available.

        Returns
        -------
        dict or None
            Dictionary with 'features', 'expression', 'image' if found; else None.
        """
        if not self.domain:
            return None

        # Try up to 5 times to find an artifact with an available image
        for _ in range(5):
            domain_entry = random.choice(self.domain)
            
            # Check if image path is available
            image_path = domain_entry.get('image_path') or domain_entry.get('artifact', {}).get('image_path')
            if image_path:
                try:
                    # Try to load the image
                    print(f"Loading domain artifact image from: {image_path}")
                    full_path = os.path.join(self.log_dir, image_path)
                    if not os.path.exists(full_path):
                        print(f"WARNING: Image file not found at: {full_path}")
                        continue
                        
                    image = Image.open(full_path)
                    
                    # Get expression
                    expression_str = domain_entry.get('expression') or domain_entry.get('artifact', {}).get('expression')
                    if isinstance(expression_str, str):
                        expression = genart.ExpressionNode.from_string(expression_str)
                    else:
                        expression = expression_str
                    
                    # Get features - may need to extract if not available
                    features = domain_entry.get('features') or domain_entry.get('artifact', {}).get('features')
                    if features is None:
                        print("Extracting features for domain artifact")
                        features = self.feature_extractor.extract_features(image)
                    
                    # Return artifact with loaded image
                    return {
                        'features': features,
                        'expression': expression,
                        'image': image,
                        'image_path': image_path
                    }
                except Exception as e:
                    print(f"Error loading artifact image {image_path}: {e}")
                    continue
        
        print("Could not find a valid artifact with available image")
        return None

    @time_it
    def process_inboxes_parallel(self):
        """
        Batch process agent inboxes in parallel on GPU (if available).
        Each message includes an artifact with features. The receiving
        agent computes novelty and decides whether to accept it.
        """
        print(f"Processing inboxes at step {self.schedule.time}")
        if not torch.cuda.is_available():
            # Fallback to simple batch if no CUDA
            print("No CUDA available, using batch processing")
            self.process_inboxes_batch()
            return

        all_features = []
        all_messages = []
        agent_indices = []

        # Collect all messages - use efficient preprocessing
        message_count = 0
        for agent_idx, agent in enumerate(self.schedule.agents):
            message_count += len(agent.inbox)
            
        # Pre-allocate arrays for better memory efficiency
        all_features = []
        all_messages = []
        agent_indices = []
        
        for agent_idx, agent in enumerate(self.schedule.agents):
            if not agent.inbox:
                continue
            for msg in agent.inbox:
                if 'artifact' in msg and 'features' in msg['artifact'] and msg['artifact']['features'] is not None:
                    all_features.append(msg['artifact']['features'])
                    all_messages.append((agent_idx, msg))
                    agent_indices.append(agent_idx)

        if not all_features:
            print("No messages with features found")
            return

        print(f"Processing {len(all_features)} messages")
        try:
            # Process messages in smaller batches to avoid OOM issues
            BATCH_SIZE = 256
            novelty_scores = {}
            
            for batch_start in range(0, len(all_features), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(all_features))
                
                # Get batch data
                batch_features = all_features[batch_start:batch_end]
                batch_indices = agent_indices[batch_start:batch_end]
                batch_messages = all_messages[batch_start:batch_end]
                
                # Stack features for batch processing
                batch_features_tensor = torch.stack(batch_features).contiguous()
                if batch_features_tensor.device.type == 'cpu':
                    batch_features_tensor = batch_features_tensor.pin_memory()
                
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    batch_features_tensor = batch_features_tensor.cuda(non_blocking=True)
                    
                    # Group by agent for efficient batch processing
                    unique_agents = set(batch_indices)
                    for agent_idx in unique_agents:
                        agent = self.schedule.agents[agent_idx]
                        mask = torch.tensor([i == agent_idx for i in batch_indices],
                                            device=batch_features_tensor.device)
                        agent_features = batch_features_tensor[mask]
                        
                        if len(agent_features) > 0:
                            # Use the more optimized batch_get_novelty_stream
                            scores = agent.knn.batch_get_novelty_stream(agent_features, stream)
                            
                            # Store or extend scores
                            if agent_idx not in novelty_scores:
                                novelty_scores[agent_idx] = scores
                            else:
                                novelty_scores[agent_idx] = torch.cat([novelty_scores[agent_idx], scores])
                
                # Sync before processing next batch
                stream.synchronize()
                
                # Free up memory after each batch
                del batch_features_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Process acceptance decisions agent by agent
            for agent_idx, agent in enumerate(self.schedule.agents):
                if agent_idx not in novelty_scores:
                    agent.inbox = []
                    continue
                    
                agent_messages = [msg for i, (idx, msg) in enumerate(all_messages) if idx == agent_idx]
                scores = novelty_scores[agent_idx]
                
                if len(scores) != len(agent_messages):
                    logger.warning(f"Score count mismatch for agent {agent_idx}: {len(scores)} scores vs {len(agent_messages)} messages")
                    continue
                
                # Process all messages for this agent at once
                for msg, novelty in zip(agent_messages, scores):
                    raw_novelty_score = novelty.item() # Keep raw score
                    normalized_novelty_score = self.normalise_novelty(raw_novelty_score)
                    interest = agent.hedonic_evaluation(normalized_novelty_score)
                    accepted = interest > self.domain_threshold

                    # Log the message evaluation event
                    log_event(
                        step=self.schedule.time,
                        event_type='message_evaluation',
                        agent_id=agent_idx,
                        details={
                            'sender_id': msg['sender_id'],
                            'artifact_expression': msg['artifact']['expression'].to_string(),
                            'artifact_image_path': msg['artifact'].get('image_path', None), # Get path if available
                            'raw_novelty': raw_novelty_score,
                            'normalized_novelty': normalized_novelty_score,
                            'calculated_interest': interest,
                            'domain_threshold': self.domain_threshold,
                            'accepted': accepted
                        }
                    )

                    # Update network
                    self.update_connection_strength(
                        sender_id=msg['sender_id'],
                        receiver_id=agent_idx,
                        success=accepted,
                        timestamp=self.schedule.time
                    )

                    if accepted:
                        features = msg['artifact']['features']
                        # Ensure shape correctness
                        if len(features.shape) == 1:
                            features = features.unsqueeze(0)
                        elif len(features.shape) == 3:
                            features = features.squeeze(1)

                        agent.knn.add_feature_vectors(features, self.schedule.time)
                        self.add_to_domain(domain_entry={
                            'artifact': msg['artifact'].copy(),
                            'creator_id': msg['sender_id'],
                            'evaluator_id': agent_idx,
                            'novelty': raw_novelty_score,
                            'interest': interest,
                            'timestamp': self.schedule.time
                        })

                    agent.average_interest_other.append(interest)
                    
                # Clear inbox after processing
                agent.inbox = []
                
            # Clear memory
            del novelty_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Clear inboxes on error to prevent message accumulation
            for agent in self.schedule.agents:
                agent.inbox = []

    def process_inboxes_batch(self):
        """
        Fallback method if no CUDA is available: process each agent's inbox
        in a batched manner on CPU.
        """
        for agent in self.schedule.agents:
            if not agent.inbox:
                continue
            features_list = []
            messages = []

            for msg in agent.inbox:
                if 'artifact' in msg and 'features' in msg['artifact'] and msg['artifact']['features'] is not None:
                    features_list.append(msg['artifact']['features'])
                    messages.append(msg)

            if not features_list:
                agent.inbox = []
                continue

            # Combine into one batch
            batch_features = torch.stack(features_list)
            novelty_scores = agent.knn.batch_get_novelty(batch_features)
            
            for msg, novelty in zip(messages, novelty_scores):
                raw_novelty_score = novelty.item() # Keep raw score
                normalized_novelty_score = self.normalise_novelty(raw_novelty_score)
                interest = agent.hedonic_evaluation(normalized_novelty_score)
                accepted = interest > self.domain_threshold

                # Log the message evaluation event
                log_event(
                    step=self.schedule.time,
                    event_type='message_evaluation',
                    agent_id=agent.unique_id,
                    details={
                        'sender_id': msg['sender_id'],
                        'artifact_expression': msg['artifact']['expression'].to_string(),
                        'artifact_image_path': msg['artifact'].get('image_path', None), # Get path if available
                        'raw_novelty': raw_novelty_score,
                        'normalized_novelty': normalized_novelty_score,
                        'calculated_interest': interest,
                        'domain_threshold': self.domain_threshold,
                        'accepted': accepted
                    }
                )

                # Update network
                self.update_connection_strength(
                    sender_id=msg['sender_id'],
                    receiver_id=agent.unique_id,
                    success=accepted,
                    timestamp=self.schedule.time
                )

                if accepted:
                    feats = msg['artifact']['features']
                    if len(feats.shape) == 1:
                        feats = feats.unsqueeze(0)
                    elif len(feats.shape) == 3:
                        feats = feats.squeeze(1)

                    agent.knn.add_feature_vectors(feats, self.schedule.time)
                    self.add_to_domain(domain_entry={
                        'artifact': msg['artifact'].copy(),
                        'creator_id': msg['sender_id'],
                        'evaluator_id': agent.unique_id,
                        'novelty': raw_novelty_score,
                        'interest': interest,
                        'timestamp': self.schedule.time
                    })

                agent.average_interest_other.append(interest)
            agent.inbox = []

    @time_it
    def process_generations_parallel(self):
        """
        Parallel generation of expressions, images, and features.
        """
        print(f"Starting generations for step {self.schedule.time}")
        # Generate expression for each agent
        for agent in self.schedule.agents:
            if not agent.current_expression:
                # Create a random expression if needed
                print(f"Creating new random expression for agent {agent.unique_id}")
                agent.current_expression = genart.ExpressionNode.create_random(depth=agent.gen_depth)
            else:
                # Decide whether to mutate or breed based on interest
                if agent.average_interest < self.boredom_threshold and agent.artifact_memory and random.random() < 0.3:
                    # Breed with a memory expression
                    parent1 = agent.current_expression
                    parent2_data = random.choice(agent.artifact_memory)
                    if isinstance(parent2_data, dict) and 'expression_str' in parent2_data:
                        try:
                            parent2 = genart.ExpressionNode.from_string(parent2_data['expression_str'])
                            print(f"Breeding expression for agent {agent.unique_id}")
                            agent.current_expression = parent1.breed(parent2, agent_id=agent.unique_id, step=self.schedule.time)
                        except Exception as e:
                            print(f"Error breeding for agent {agent.unique_id}: {e}")
                            logger.warning(f"Error breeding for agent {agent.unique_id}: {e}")
                            # Fall back to mutation
                            agent.current_expression = parent1.mutate(agent_id=agent.unique_id, step=self.schedule.time)
                    else:
                        # Fall back to mutation
                        print(f"Mutating expression for agent {agent.unique_id}")
                        agent.current_expression = agent.current_expression.mutate(agent_id=agent.unique_id, step=self.schedule.time)
                else:
                    # Mutate the existing expression
                    print(f"Mutating expression for agent {agent.unique_id}")
                    agent.current_expression = agent.current_expression.mutate(agent_id=agent.unique_id, step=self.schedule.time)

        # Process in batches for better GPU utilization
        batch_size = 32
        print(f"Processing agent generation in batches of {batch_size}")
        for batch_start in range(0, len(self.schedule.agents), batch_size):
            batch_end = min(batch_start + batch_size, len(self.schedule.agents))
            batch_agents = self.schedule.agents[batch_start:batch_end]
            print(f"Processing batch {batch_start}-{batch_end}")
            
            # Generate images and extract features for the batch
            for agent in batch_agents:
                try:
                    print(f"Generating image for agent {agent.unique_id}")
                    # Generate image from expression
                    expr = agent.current_expression
                    img = self.image_generator.generate(expr)
                    agent.generated_image = img
                    agent.generated_expression = expr.to_string()
                    
                    # Save image with agent context
                    print(f"Saving image for agent {agent.unique_id}")
                    metadata = {
                        'step': self.schedule.time,
                        'type': 'generation'
                    }
                    image_path = self.image_saver.queue_agent_image_save(
                        img,
                        agent.unique_id,
                        metadata=metadata
                    )
                    print(f"Image path for agent {agent.unique_id}: {image_path}")
                    
                    # Extract features
                    print(f"Extracting features for agent {agent.unique_id}")
                    features = self.feature_extractor.extract_features(img)
                    
                    # Get raw novelty
                    print(f"Calculating novelty for agent {agent.unique_id}")
                    if hasattr(agent.knn, 'get_index_size') and agent.knn.get_index_size() > 0:
                        novelty_raw = agent.knn.get_novelty(features)
                    else:
                        novelty_raw = torch.tensor(1.0)
                        
                    # Normalize novelty
                    normalized_novelty = self.normalise_novelty(novelty_raw.item())
                    
                    # Calculate interest
                    print(f"Evaluating interest for agent {agent.unique_id}")
                    interest = agent.hedonic_evaluation(normalized_novelty)
                    
                    # Update KNN
                    print(f"Updating KNN for agent {agent.unique_id}")
                    agent.knn.add_feature_vectors(features, self.schedule.time)
                    
                    # Log generation
                    print(f"Logging generation for agent {agent.unique_id}")
                    log_event(
                        step=self.schedule.time,
                        event_type='artifact_generated',
                        agent_id=agent.unique_id,
                        details={
                            'expression': expr.to_string(),
                            'image_path': image_path,
                            'raw_novelty': novelty_raw.item(),
                            'normalized_novelty': normalized_novelty,
                            'calculated_interest': interest,
                            'will_be_shared': interest > self.self_threshold
                        }
                    )
                    
                    # Save to agent memory
                    print(f"Updating memory for agent {agent.unique_id}")
                    agent.artifact_memory.append({
                        'expression_str': expr.to_string(),
                        'features': features.cpu(),
                        'image_path': image_path,
                        'step_generated': self.schedule.time
                    })
                    
                    # Keep track of interest
                    agent.average_interest_self.append(interest)
                    
                    # Possibly share with other agents
                    if interest > self.self_threshold:
                        print(f"Agent {agent.unique_id} sharing artifact")
                        # Log the sharing event
                        log_event(
                            step=self.schedule.time,
                            event_type='message_sent',
                            agent_id=agent.unique_id,
                            details={
                                'artifact_expression': expr.to_string(),
                                'artifact_image_path': image_path,
                                'artifact_interest': interest,
                                'num_recipients': self.amount_shares
                            }
                        )
                        
                        # Prepare message
                        message = {
                            'artifact': {
                                'expression': expr,
                                'features': features,
                                'image': img,
                                'image_path': image_path
                            },
                            'sender_id': agent.unique_id,
                            'timestamp': self.schedule.time
                        }
                        
                        # Select recipients and send message
                        recipients = self._select_recipients(agent.unique_id)
                        for recipient_id in recipients:
                            recipient = self.schedule.agents[recipient_id]
                            recipient.inbox.append(message)
                    
                except Exception as e:
                    print(f"ERROR in generation for agent {agent.unique_id}: {e}")
                    logger.error(f"Error in generation for agent {agent.unique_id}: {e}")
                    
            # Process image save queue to avoid memory buildup
            print("Processing image save queue")
            self.image_saver.process_save_queue()
            
        # Release memory
        print("Clearing image memory")
        for agent in self.schedule.agents:
            agent.generated_image = None
    
    def _select_recipients(self, sender_id):
        """Helper method to select recipients for a message."""
        # Get the IDs of all agents except the sender
        possible_recipients = [a.unique_id for a in self.schedule.agents if a.unique_id != sender_id]
        
        # Sample random recipients if we have more agents than needed
        if len(possible_recipients) > self.amount_shares:
            recipients = random.sample(possible_recipients, self.amount_shares)
        else:
            recipients = possible_recipients
            
        return recipients

    @time_it
    def update_novelty_bounds(self):
        """
        Update novelty percentile bounds using a rolling window of recent values.
        """
        if len(self.novelty_values) < 2:
            return

        self.bottom_1_percentile_novelty = np.percentile(self.novelty_values, 1)
        self.top_99_percentile_novelty = np.percentile(self.novelty_values, 99)

    @time_it
    def normalise_novelty(self, novelty):
        """
        Normalize novelty to [0,1] range using the current percentile bounds.

        Parameters
        ----------
        novelty : float

        Returns
        -------
        float
            Normalized novelty score in [0, 1].
        """
        self.novelty_values.append(novelty)
        if len(self.novelty_values) > config.MAX_HISTORY:
            self.novelty_values = self.novelty_values[-config.MAX_HISTORY:]

        if len(self.novelty_values) < 2:
            # Not enough data to scale properly, just clamp
            return np.clip(novelty, 0, 1)

        normalized_novelty = (
            (novelty - self.bottom_1_percentile_novelty) /
            (self.top_99_percentile_novelty - self.bottom_1_percentile_novelty + 1e-8)
        )
        normalized_novelty = np.clip(normalized_novelty, 0, 1)
        return normalized_novelty

    @time_it
    def log_system_metrics(self):
        """
        Log system-wide domain and network metrics to TensorBoard.
        Also logs top contributor data every so often.
        """
        step = self.schedule.time
        self.writer.add_scalar("domain/size", len(self.domain), step)

        # Log top contributors
        if step % 25 == 0:  # occasionally
            top_contributors = np.argsort(self.domain_contributions)[-8:][::-1]
            for rank, agent_id in enumerate(top_contributors):
                contrib_count = len(self.agent_contribution_history[agent_id])
                self.writer.add_text(f'top_contributors/rank_{rank}',
                                     f'Agent {agent_id} (Total: {contrib_count})',
                                     step)
                self.writer.add_scalar(f'top_contributors/agent_{agent_id}/contribution_count',
                                       contrib_count, step)
                self.writer.add_scalar(f'top_contributors/agent_{agent_id}/current_rank',
                                       rank, step)

    @time_it
    def step(self):
        """
        Execute a single step of the simulation:
          1) Save pending images.
          2) Process all inboxes (communication).
          3) Generate new artifacts in parallel (batch).
          4) Update novelty thresholds if needed.
          5) Log network metrics.
          6) Mesa schedule step.
        """
        print(f"\n\n=== Starting step {self.schedule.time} ===")
        
        print("Processing image save queue")
        self.image_saver.process_save_queue()
        
        print("Processing inboxes")
        self.process_inboxes_parallel()
        
        print("Generating new artifacts")
        self.process_generations_parallel()
        
        print("Calculating novelty thresholds")
        self.calculate_novelty_threshold()

        # Update novelty bounds occasionally
        if self.schedule.time % 5 == 0 or self.schedule.time in [1, 2]:
            print("Updating novelty bounds")
            self.update_novelty_bounds()

        print("Logging system metrics")
        self.log_system_metrics()
        
        print("Running Mesa schedule step")
        self.schedule.step()
        
        print(f"=== Completed step {self.schedule.time-1} ===\n")

    def run_model(self, steps=config.EXPERIMENT_STEPS):
        """
        Convenience method to run the simulation for a given number of steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation
        """
        print(f"Starting simulation run for {steps} steps")
        for step_i in range(steps):
            TimingStats().reset_step()
            self.step()
            TimingStats().print_step_report()

        # Cleanup
        print("Simulation complete, performing cleanup...")
        self.writer.close()
        self.image_saver.stop() # Use stop instead of just process_save_queue
        # Close the logger handlers properly
        for handler in self.json_logger.handlers[:]:
            handler.close()
            self.json_logger.removeHandler(handler)
        print("Cleanup complete")


@time_it
def run_sim_from_config():
    """
    Run a single simulation using parameters from config.py with a tqdm progress bar.
    Logs everything in a timestamped directory. Instantiates a Model with:
      - number_agents = config.NUMBER_AGENTS
      - output_dims = config.OUTPUT_DIMS
      - runs for config.EXPERIMENT_STEPS steps
    """
    import datetime
    import os
    from tqdm import tqdm

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join("logs", f"single_sim_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Initializing model with log directory: {log_dir}")
    model = Model(
        number_agents=config.NUMBER_AGENTS,
        output_dims=config.OUTPUT_DIMS,
        log_dir=log_dir
    )

    try:
        print(f"Starting simulation for {config.EXPERIMENT_STEPS} steps")
        for _ in tqdm(range(config.EXPERIMENT_STEPS), desc="Simulation Progress"):
            TimingStats().reset_step()
            model.step()
            TimingStats().print_step_report()
    finally:
        # Ensure cleanup happens even on error
        print("Simulation finished or interrupted, cleaning up...")
        if hasattr(model, 'writer'):
            model.writer.close()
        
        # Process any remaining images in the queue
        if hasattr(model, 'image_saver'):
            try:
                # Call the stop method which ensures all images are saved
                model.image_saver.stop()
            except Exception as e:
                logger.error(f"Error stopping image_saver: {e}")
        
        # Close logger handlers properly
        if hasattr(model, 'json_logger'):
            for handler in model.json_logger.handlers[:]:
                try:
                    handler.close()
                    model.json_logger.removeHandler(handler)
                except Exception as e:
                    logger.error(f"Error closing logger handler: {e}")
                    
        print(f"Simulation completed. Results saved in: {log_dir}")

if __name__ == "__main__":
    print("Starting simulation script...")
    run_sim_from_config()