"""
config.py

Stores global parameters and configuration values used across the simulation.
"""

# Simulation and Model Parameters
NUMBER_AGENTS = 1000
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
CSV_LOGGER_FILENAME = "agent_metrics.csv"
EXPERIMENT_STEPS = 5000
