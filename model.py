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
        # Log agent's current interest in the shared CSV
        self.model.log_agent_step(
            agent_id=self.unique_id,
            novelty=None,
            hedonic=None,
            average_interest=self.average_interest,
            interaction_count=self.model.communication_matrix[self.unique_id].sum(),
            domain_contributions=self.model.domain_contributions[self.unique_id]
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
                domain_novelty = self.model.normalise_novelty(domain_novelty)
                domain_interest = self.hedonic_evaluation(domain_novelty)

                # Log domain interaction
                self.model.log_agent_step(
                    agent_id=self.unique_id,
                    novelty=domain_novelty,
                    hedonic=domain_interest,
                    average_interest=self.average_interest,
                    interaction_count=self.model.communication_matrix[self.unique_id].sum(),
                    domain_contributions=self.model.domain_contributions[self.unique_id],
                    note="domain_interaction"
                )

                # Possibly adopt if more interesting than current average
                if domain_interest > self.average_interest:
                    self.current_expression = domain_artifact['expression']
                    self.knn.add_feature_vectors(domain_features, self.model.schedule.time)
                    self.average_interest = domain_novelty
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
        super().__init__()
        self.num_agents = number_agents
        self.output_dims = output_dims
        self.schedule = mesa.time.RandomActivation(self)

        # Feature extraction, image generation
        self.feature_extractor = FeatureExtractor(output_dims=self.output_dims)
        self.image_generator = genart.ImageGenerator(32, 32)

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

        # Logging directories
        self.log_dir = log_dir or f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_v2"
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize system-wide TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # CSV logger for agent metrics
        self.agent_log_path = os.path.join(self.log_dir, config.CSV_LOGGER_FILENAME)
        self.agent_log_file = open(self.agent_log_path, mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(
            self.agent_log_file,
            fieldnames=[
                "step", "agent_id", "novelty", "hedonic", "average_interest",
                "interaction_count", "domain_contributions", "note"
            ]
        )
        self.csv_writer.writeheader()

        # Network tracking
        self.network_tracker = NetworkTracker(number_agents, self.log_dir)

        # Image saver
        self.image_saver = ImageSaver(max_queue_size=config.MAX_IMAGE_QUEUE_SIZE)
        self.image_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Create agents
        self._initialize_agents()

        # For Mesa run loop
        self.running = True

    def _initialize_agents(self):
        """
        Create and add agents to the model schedule.
        """
        for i in range(self.num_agents):
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
    def add_to_domain(self, entry):
        """
        Add a new artifact to the domain repository with async image saving and
        logs domain metrics.

        Parameters
        ----------
        entry : dict
            Dictionary containing keys:
            'artifact', 'creator_id', 'evaluator_id', 'novelty', 'interest', 'timestamp'
        """
        if len(self.domain) >= self.max_domain_size:
            self.domain.pop(0)

        artifact = entry['artifact']
        creator_id = entry['creator_id']
        self.domain_contributions[creator_id] += 1
        self.agent_contribution_history[creator_id].append(self.schedule.time)

        self.writer.add_scalar(f'domain/total_contributions_agent_{creator_id}',
                               self.domain_contributions[creator_id],
                               self.schedule.time)

        # Image filename and path
        image_filename = (
            f"domain_id-{len(self.domain)}"
            f"_creator-{entry['creator_id']}"
            f"_evaluator-{entry['evaluator_id']}"
            f"_nov-{entry['novelty']}"
            f"_int-{entry['interest']}.png"
        )
        image_filename = image_filename.replace(" ", "_").replace("/", "-")
        image_path = os.path.join(self.image_dir, image_filename)

        # Queue async save
        self.image_saver.queue_image_save(artifact['image'], image_path)
        if len(self.image_saver.image_save_queue) >= self.image_saver.max_queue_size:
            self.image_saver.process_save_queue()

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

        # Log system-wide domain metrics
        self.writer.add_scalar('domain/total_size', len(self.domain), self.schedule.time)
        self.writer.add_scalar('domain/last_interest', entry['interest'], self.schedule.time)
        self.writer.add_scalar('domain/last_novelty', entry['novelty'], self.schedule.time)

    @time_it
    def get_random_domain_artifact(self):
        """
        Return a random artifact from the domain if available, ensuring
        the image file is saved and ready for reading.

        Returns
        -------
        dict or None
            Dictionary with 'features', 'expression', 'image' if found; else None.
        """
        if not self.domain:
            return None

        for _ in range(5):
            domain_entry = random.choice(self.domain)
            image_path = domain_entry['image_path']
            if self.image_saver.is_image_ready(image_path):
                try:
                    # Load artifact from disk
                    with Image.open(image_path) as img:
                        features = self.feature_extractor.extract_features(img)
                    expr = genart.ExpressionNode.from_string(domain_entry['expression'])
                    return {
                        'features': features,
                        'expression': expr,
                        'image': img
                    }
                except Exception as e:
                    logger.error(f"Error loading artifact image {image_path}: {e}")
                    continue
        return None

    @time_it
    def process_inboxes_parallel(self):
        """
        Batch process agent inboxes in parallel on GPU (if available).
        Each message includes an artifact with features. The receiving
        agent computes novelty and decides whether to accept it.
        """
        if not torch.cuda.is_available():
            # Fallback to simple batch if no CUDA
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
        all_features.reserve(message_count) if hasattr(all_features, 'reserve') else None
        
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
                    novelty_score = self.normalise_novelty(novelty.item())
                    interest = agent.hedonic_evaluation(novelty_score)

                    # Log the agent's acceptance decision
                    accepted = interest > self.domain_threshold
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
                        domain_entry = {
                            'artifact': msg['artifact'],
                            'creator_id': msg['sender_id'],
                            'evaluator_id': agent_idx,
                            'novelty': novelty.item(),
                            'interest': interest,
                            'timestamp': self.schedule.time
                        }
                        self.add_to_domain(domain_entry)

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
                if msg['artifact']['features'] is not None:
                    features_list.append(msg['artifact']['features'])
                    messages.append(msg)

            if not features_list:
                agent.inbox = []
                continue

            # Combine into one batch
            batch_features = torch.stack(features_list)
            novelty_scores = agent.knn.batch_get_novelty(batch_features)
            for msg, novelty in zip(messages, novelty_scores):
                novelty_score = self.normalise_novelty(novelty.item())
                interest = agent.hedonic_evaluation(novelty_score)

                # Record network
                accepted = interest > self.domain_threshold
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
                    domain_entry = {
                        'artifact': msg['artifact'],
                        'creator_id': msg['sender_id'],
                        'evaluator_id': agent.unique_id,
                        'novelty': novelty.item(),
                        'interest': interest,
                        'timestamp': self.schedule.time
                    }
                    self.add_to_domain(domain_entry)

                agent.average_interest_other.append(interest)
            agent.inbox = []

    @time_it
    def process_generations_parallel(self):
        """
        Batch process artifact generation in parallel, using CUDA streams if available.
        Each agent either creates a new random expression or mutates/breeds from existing memory.
        Forces result shapes to remain [H, W, 4] after evaluation to prevent dimension mismatch.
        """
        if not torch.cuda.is_available():
            return

        expressions = []
        agents = []

        # Collect expressions to be generated
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
            # Configure parallel generation - use batch_generate
            batch_size = config.OPTIMAL_BATCH_SIZE
            num_streams = config.NUM_STREAMS
            
            # Split work into stream-sized batches
            expr_batches = [expressions[i:i + batch_size] 
                           for i in range(0, len(expressions), batch_size)]
            agent_batches = [agents[i:i + batch_size] 
                            for i in range(0, len(agents), batch_size)]
            
            all_images = []
            all_features = []
            
            # Process each batch with appropriate CUDA stream
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            
            for batch_idx, (expr_batch, agent_batch) in enumerate(zip(expr_batches, agent_batches)):
                stream_idx = batch_idx % num_streams
                with torch.cuda.stream(streams[stream_idx]):
                    # Use batch generation instead of one-by-one
                    images_batch = self.image_generator.batch_generate(expr_batch)
                    all_images.extend(images_batch)
                    
                    # Convert images to tensor batch for feature extraction
                    tensor_batch = []
                    for img in images_batch:
                        img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
                        tensor_batch.append(img_tensor)
                    
                    tensor_batch = torch.stack(tensor_batch).to(self.feature_extractor.device)
                    features_batch = self.feature_extractor(tensor_batch)
                    all_features.extend([feat.detach() for feat in features_batch])
            
            # Synchronize all streams before continuing
            torch.cuda.synchronize()
            
            # Now assign results back to agents
            for agent, expr, img, features in zip(agents, expressions, all_images, all_features):
                agent.generated_image = img
                agent.generated_expression = expr.to_string()
                
                # kNN update
                features_tensor = features
                if len(features_tensor.shape) == 1:
                    features_tensor = features_tensor.unsqueeze(0)
                elif len(features_tensor.shape) == 3:
                    features_tensor = features_tensor.squeeze(1)
                
                agent.knn.add_feature_vectors(features_tensor, self.schedule.time)
                
                # Store artifact in agent memory
                result = {
                    'image': img,
                    'features': features_tensor,
                    'expression': expr
                }
                agent.artifact_memory.append(result)
                
                # Evaluate novelty
                novelty_scores = agent.knn.batch_get_novelty(features_tensor)
                novelty = novelty_scores[0].item()
                novelty = self.normalise_novelty(novelty)
                interest = agent.hedonic_evaluation(novelty)
                
                result['novelty'] = novelty
                result['interest'] = interest
                agent.average_interest_self.append(interest)
                
                # Possibly share
                if interest > self.self_threshold:
                    message = {
                        'artifact': result,
                        'sender_id': agent.unique_id,
                        'timestamp': self.schedule.time
                    }
                    other_agents = [a for a in self.schedule.agents if a.unique_id != agent.unique_id]
                    if other_agents:
                        for _ in range(self.amount_shares):
                            recipient = random.choice(other_agents)
                            recipient.inbox.append(message)

                # Log generation step to CSV
                self.log_agent_step(
                    agent_id=agent.unique_id,
                    novelty=novelty,
                    hedonic=interest,
                    average_interest=agent.average_interest,
                    interaction_count=self.communication_matrix[agent.unique_id].sum(),
                    domain_contributions=self.domain_contributions[agent.unique_id],
                    note="generation"
                )

        except Exception as e:
            logger.error(f"Error in parallel generation: {e}")
            torch.cuda.empty_cache()

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
        self.image_saver.process_save_queue()
        self.process_inboxes_parallel()
        self.process_generations_parallel()
        self.calculate_novelty_threshold()

        # Update novelty bounds occasionally
        if self.schedule.time % 5 == 0 or self.schedule.time in [1, 2]:
            self.update_novelty_bounds()

        self.novelty_calculated = False

        if self.schedule.time % 1 == 0:
            # Network metrics
            metrics = self.network_tracker.get_network_metrics()
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'network/{metric_name}', value, self.schedule.time)

            # Per-agent network stats
            for agent_id in range(self.num_agents):
                stats = self.network_tracker.get_agent_stats(agent_id)
                for stat_name, val in stats.items():
                    self.writer.add_scalar(f'agents/{agent_id}/network/{stat_name}',
                                           val, self.schedule.time)

            # Save network snapshot every 100 steps
            if self.schedule.time % 100 == 0:
                self.network_tracker.save_snapshot(self.schedule.time)

            # Optional debug printing
            if self.schedule.time % config.TENSORBOARD_UPDATE_STEPS == 0:
                comm_matrix = self.network_tracker.communication_matrix
                top_pairs = []
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        total_comms = comm_matrix[i, j] + comm_matrix[j, i]
                        if total_comms > 0:
                            top_pairs.append((i, j, total_comms))
                top_pairs.sort(key=lambda x: x[2], reverse=True)
                logger.info("\nTop 5 Communicating Pairs:")
                for i, j, comms in top_pairs[:5]:
                    accepts = (
                        self.network_tracker.acceptance_matrix[i, j] +
                        self.network_tracker.acceptance_matrix[j, i]
                    )
                    logger.info(f"Agents {i}-{j}: {comms} communications, {accepts} acceptances")

        self.log_system_metrics()
        self.schedule.step()

    def log_agent_step(self,
                       agent_id,
                       novelty,
                       hedonic,
                       average_interest,
                       interaction_count,
                       domain_contributions,
                       note=""):
        """
        Log agent-specific metrics to the CSV file. Minimizes overhead
        by not using TensorBoard for every agent at every step.

        Parameters
        ----------
        agent_id : int
            Unique ID of the agent
        novelty : float or None
            Novelty of the current artifact (if any)
        hedonic : float or None
            Hedonic value from Wundt curve
        average_interest : float
            Agent's running interest average
        interaction_count : int
            Number of communications sent by this agent
        domain_contributions : float
            Number of domain contributions by this agent
        note : str
            Optional note describing the event type (e.g., "generation")
        """
        step = self.schedule.time
        row = {
            "step": step,
            "agent_id": agent_id,
            "novelty": novelty if novelty is not None else "",
            "hedonic": hedonic if hedonic is not None else "",
            "average_interest": average_interest,
            "interaction_count": interaction_count,
            "domain_contributions": domain_contributions,
            "note": note
        }
        self.csv_writer.writerow(row)
        self.agent_log_file.flush()

    def run_model(self, steps=config.EXPERIMENT_STEPS):
        """
        Convenience method to run the simulation for a given number of steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation
        """
        for step_i in range(steps):
            TimingStats().reset_step()
            self.step()
            TimingStats().print_step_report()

        # Cleanup
        self.writer.close()
        self.agent_log_file.close()
        self.image_saver.process_save_queue()

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

    model = Model(
        number_agents=config.NUMBER_AGENTS,
        output_dims=config.OUTPUT_DIMS,
        log_dir=log_dir
    )

    for _ in tqdm(range(config.EXPERIMENT_STEPS), desc="Simulation Progress"):
        TimingStats().reset_step()
        model.step()
        TimingStats().print_step_report()

    model.writer.close()
    model.agent_log_file.close()
    model.image_saver.process_save_queue()

if __name__ == "__main__":
    run_sim_from_config()