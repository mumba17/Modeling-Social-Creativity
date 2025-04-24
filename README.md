<!-- Optional: Add a logo or banner here -->
<!-- ![Project Banner](path/to/your/banner.png) -->

# Modeling Social Creativity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!-- Add other badges as needed, e.g., build status, code coverage -->

A computational framework for simulating social creativity in multi-agent systems, exploring how novelty, interest, and social interaction drive the evolution of creative artifacts.

## Project Overview

This project implements a computational model of social creativity, examining how creative artifacts evolve in a multi-agent system with social interactions. The simulation demonstrates how novelty, interest, and social dynamics influence the emergence of creative artifacts within a collective.

The model simulates a population of agents engaging in artifact creation, evaluation, and sharing. Each agent:
- Generates visual artifacts using quaternion-based generative art expressions
- Evaluates the novelty of artifacts using k-Nearest Neighbors (kNN)
- Determines interest in artifacts using a Wundt curve (an inverted U-shaped relationship between novelty and interest)
- Shares interesting artifacts with other agents
- Contributes to a collective domain of artifacts when they resonate with the receiving agents

Through these interactions, the simulation studies how:
- Social networks form between agents with compatible novelty preferences
- The domain of artifacts evolves over time in terms of novelty and diversity
- Group creativity emerges from individual interactions
- **Modular Design**: Components (generation, evaluation, etc.) are designed to be potentially interchangeable for flexible research experimentation.

## Key Features

- Multi-Agent Simulation using the Mesa framework.
- Quaternion-based Generative Art for artifact creation.
- ResNet-based Feature Extraction for artifact embedding.
- k-Nearest Neighbors (kNN) for Novelty Detection with adaptive k.
- Wundt Curve implementation for modeling Interest based on novelty.
- Dynamic Social Network formation and analysis.
- Configurable parameters for experimentation.
- Performance optimizations including GPU acceleration and caching.

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended for faster processing)
- Mesa (agent-based modeling framework)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/Modeling-Social-Creativity.git
cd Modeling-Social-Creativity
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run a single simulation with default parameters:

```bash
python model.py
```

The simulation will create a `logs` directory with:
- TensorBoard logs for visualizing metrics
- Generated images in the `images` subdirectory
- Network data as .gexf files for network analysis
- CSV logs of agent interactions and metrics

You can view TensorBoard metrics by running:
```bash
tensorboard --logdir=logs
```

## Configuration

Modify `config.py` to change simulation parameters:

```python
# Example: Change the number of agents and simulation steps
NUMBER_AGENTS = 500  # Default: 1000
EXPERIMENT_STEPS = 2000  # Default: 5000
```

Key parameters include:
- `NUMBER_AGENTS`: Size of the agent population
- `OUTPUT_DIMS`: Dimensionality of feature vectors
- `BOREDOM_THRESHOLD`: When agents seek new influences
- `ALPHA`: Learning rate for interest updates
- `INIT_GEN_DEPTH_MIN/MAX`: Complexity range for generated expressions
- `WUNDT_REWARD_STD`, `WUNDT_PUNISH_STD`, `WUNDT_ALPHA`: Shape of the interest curve

## Documentation

Detailed technical explanations of the components, model architecture, and theoretical background are intended for the `/docs` directory (coming soon).

## Core Components (Overview)

*(Detailed descriptions will be moved to `/docs`)*

### Generative Art (genart.py)

The artifact generation system uses quaternion mathematics to create abstract visual patterns:

- `ExpressionNode`: Represents a node in the expression tree for generating images
- `ImageGenerator`: Evaluates expressions over a coordinate grid to produce images
- `QuaternionTensor`: Custom PyTorch implementation of quaternion operations

Expressions can be mutated, crossed-over, and evolved over time, allowing for exploration of the creative space.

### Feature Extraction (features.py)

A ResNet-based feature extractor converts generated images into embeddings:

- Uses a pre-trained ResNet-18 model with custom adaptation layers
- Generates 64-dimensional feature vectors by default
- Includes caching mechanisms for optimization
- Normalizes features for consistent distance calculations

### Novelty Calculation (knn.py)

Implements k-Nearest Neighbors to calculate how novel an artifact is relative to previously seen artifacts:

- Dynamically updates k using the elbow method
- Supports GPU acceleration for large-scale calculations
- Includes batch processing optimizations for parallel evaluation
- Offers approximate nearest neighbor support for large datasets

### Interest Evaluation (wundtcurve.py)

Models the hedonic response to novelty using a Wundt curve:

- Represents the "inverted U" relationship between novelty and interest
- Uses parameterized Gaussian functions for reward and punishment components
- Allows personalized curves per agent with different peak interest points
- Normalizes output to a [-1, 1] range for consistent evaluation

### Network Analysis (network_tracker.py)

Tracks and analyzes the evolving social network between agents:

- Records communication events, acceptance/rejection rates
- Calculates network metrics like density, clustering, path length
- Exports network snapshots as .gexf files for analysis in tools like Gephi
- Identifies influential agents based on centrality measures

## Performance Optimizations

The simulation includes several optimizations for handling large agent populations:

- CUDA acceleration for batch processing when available
- Parallel image generation using multiple CUDA streams
- Asynchronous image saving to reduce I/O bottlenecks
- Feature caching to avoid redundant calculations
- Memory management to handle limited GPU resources

## Citation

If you use this code in your research, please cite the associated thesis:

```
@thesis{motz2025modeling,
  title={Modeling Social Creativity},
  author={Motz, Luuk},
  year={2025},
  school={Leiden University}
}
```
*Link to published paper/preprint will be added here when available.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.