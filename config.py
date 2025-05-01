"""
config.py

Stores global parameters and configuration values used across the simulation.
"""

# Simulation and Model Parameters
NUMBER_AGENTS = 10
OUTPUT_DIMS = 64

# Agent parameters
BOREDOM_THRESHOLD = 0.2
ALPHA = 0.35
INIT_GEN_DEPTH_MIN = 4
INIT_GEN_DEPTH_MAX = 8
AMOUNT_SHARES = 10

# Domain parameters
MAX_DOMAIN_SIZE = 10_000_000

# kNN parameters
KNN_DEFAULT_K = 3

# Wundt Curve parameters
WUNDT_REWARD_STD = 0.15
WUNDT_PUNISH_STD = 0.15
WUNDT_ALPHA = 1.2

# Misc thresholds and windows
WINDOW_SIZE = 100
MAX_HISTORY = 10000  # Rolling window for novelty
BATCH_SIZE_IMAGE_SAVES = 100
MAX_IMAGE_QUEUE_SIZE = 1000

# Parallel Generation Tuning
OPTIMAL_BATCH_SIZE = 128  # Reduced from 250 for better GPU memory usage
NUM_STREAMS = 4

# Performance Optimization Parameters
ENABLE_EXPRESSION_CACHING = True
MAX_FEATURE_CACHE_SIZE = 10000  # Increased cache size for better hit ratio
CLEAR_CACHE_INTERVAL = 250  # Less frequent cache clearing to retain useful entries
VECTORIZE_BATCH_OPS = True
BATCH_SIZE_GENERATION = 32  # Smaller batch size for more efficient memory usage
MAX_EXPRESSION_DEPTH = 6  # Limit expression depth for faster evaluation
PRECOMPUTE_COMMON_QUATERNIONS = True  # Enable precomputation for common quaternion operations
MIN_CACHE_HITS_TO_RETAIN = 3  # Min number of hits before clearing an entry

# kNN Optimization
ENABLE_KNN_APPROX = True  # Use approximate nearest neighbor to speed up kNN
KNN_UPDATE_FREQUENCY = 5  # Update k less frequently

# Logging and Monitoring
TENSORBOARD_UPDATE_STEPS = 50
CSV_LOGGER_FILENAME = "agent_metrics.csv" # Kept for reference, but JSONL is primary
EXPERIMENT_STEPS = 20

# --- NEW Logging Configuration ---
import logging
LOG_LEVEL = logging.INFO # Default logging level (e.g., logging.DEBUG, logging.INFO)
LOG_FILENAME = "simulation_events.jsonl" # Name of the log file
LOG_MAX_BYTES = 10 * 1024 * 1024 * 1000 * 1000 # Max size per log file (e.g., 10MB)
LOG_BACKUP_COUNT = 5 # Number of backup log files to keep
# --- END NEW Logging Configuration ---
