import torch
import numpy as np
import random

class kNN:
    def __init__(self, agent_id=None):
        """
        Initialize the kNN instance with GPU support.
        Uses elbow method to automatically determine optimal k value.
        """
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_vectors = torch.tensor([], device=self.device)
        self.k = 3  # Default initial k value
        self.last_update_step = -1
        
        print(f"Initialized kNN for agent {agent_id}")
        
    def get_max_k(self, n_samples, feature_dim=64):
        """
        Calculate maximum reasonable k value based on dataset size and dimensionality.
        
        Args:
            n_samples: Number of data points
            feature_dim: Dimensionality of feature vectors
        
        Returns:
            int: Maximum reasonable k value
        """
        # Statistical rules of thumb:
        # 1. k ≈ sqrt(n) for basic estimation
        # 2. k should be less than n/10 to avoid oversmoothing
        # 3. k should account for dimensionality
        
        sqrt_n = int(torch.sqrt(torch.tensor(n_samples)).item())
        dim_factor = int(torch.log2(torch.tensor(feature_dim)).item())
        
        # Balance between sqrt(n) and dimensionality considerations
        max_k = min(
            sqrt_n * dim_factor,  # Scale with both size and dimensionality
            n_samples // 10,      # Upper bound to prevent oversmoothing
            500                   # Hard upper limit for computational efficiency
        )
        
        return max(max_k, 3)  # Ensure minimum k of 3

    def generate_k_values(self, max_k):
        """
        Generate a reasonable progression of k values to test.
        
        Args:
            max_k: Maximum k value to consider
        
        Returns:
            list: Sorted list of k values to test
        """
        k_values = []
        
        if max_k < 3:
            return [3]  # Ensure at least the minimum k
        
        # Start with dense sampling for small k
        k_values.extend(range(3, min(11, max_k + 1)))
        
        if max_k > 10:
            # Add values with increasing steps
            k = 12
            while k <= max_k:
                if k <= 30:
                    k_values.append(k)
                    k += 2  # Step by 2 up to 30
                elif k <= 100:
                    k_values.append(k)
                    k += 10  # Step by 10 up to 100
                else:
                    k_values.append(k)
                    k += 25  # Step by 25 beyond 100
                    
        return sorted(list(set(k_values)))  # Remove any duplicates

    def calculate_k_elbow(self):
        """
        Implement the elbow method to find optimal k value.
        Uses sampling for large datasets and proper k value progression.
        
        Returns:
            int: Optimal k value based on the elbow method
        """
        try:
            n_samples = self.feature_vectors.shape[0]

            if n_samples < 2:
                return 3  # Default minimum k if not enough vectors

            # Determine maximum k and generate test values
            max_k = self.get_max_k(n_samples, self.feature_vectors.shape[1])

            k_values = self.generate_k_values(max_k)
            #print(f"max_k: {max_k}")  # Debug statement
            #print(f"k_values: {k_values}")  # Debug statement

            if not k_values:  # Ensure k_values is not empty
                print("Warning: k_values is empty. Returning default k=3")
                return 3

            # Sample data if dataset is large
            MAX_SAMPLE_SIZE = 1000
            if n_samples > MAX_SAMPLE_SIZE:
                indices = torch.randperm(n_samples)[:MAX_SAMPLE_SIZE]
                vectors_sample = self.feature_vectors[indices]
            else:
                vectors_sample = self.feature_vectors

            # Initialize tensors for distortions
            distortions = torch.zeros(len(k_values), device=self.device)

            # Normalize sampled vectors once
            normalized_vectors = vectors_sample / (torch.norm(vectors_sample, dim=1, keepdim=True) + 1e-8)

            # Calculate all pairwise distances once
            similarities = torch.matmul(normalized_vectors, normalized_vectors.T)
            distances = 1 - similarities

            # Calculate distortion for each k
            for idx, k in enumerate(k_values):
                # Exclude self-similarity by setting diagonal to inf
                distances_no_self = distances.clone()
                distances_no_self.fill_diagonal_(float('inf'))

                # Get top k distances for each point
                top_k_distances, _ = torch.topk(distances_no_self, k, largest=False, dim=1)

                # Calculate average distortion for this k
                distortion = torch.mean(top_k_distances).item()
                distortions[idx] = distortion

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Calculate the elbow point using the angle method
            coords = torch.stack([torch.tensor(k_values, device=self.device), distortions], dim=1)

            # Normalize coordinates to [0,1] range for better angle calculation
            coords_normalized = (coords - coords.min(0)[0]) / (coords.max(0)[0] - coords.min(0)[0])

            # Calculate angles between consecutive segments
            vectors = coords_normalized[1:] - coords_normalized[:-1]
            angles = []

            for i in range(len(vectors) - 1):
                v1 = vectors[i]
                v2 = vectors[i + 1]
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
                angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
                angles.append(angle.item())

            # Find the point with maximum angle (elbow point)
            if not angles:  # Ensure angles is not empty
                #print("Warning: angles is empty. Returning default k=3")
                return 3

            optimal_idx = angles.index(max(angles)) + 1
            optimal_k = k_values[optimal_idx]

            return optimal_k

        except Exception as e:
            print(f"Error in calculate_k_elbow: {e}")
            return 3  # Return default k value on error

            
    def remove_feature_vectors(self, indices):
        """
        Remove feature vectors from the kNN instance based on their indices.
        :param indices: The indices of the feature vectors to be removed
        """
        try:
            mask = torch.ones(self.feature_vectors.shape[0], dtype=torch.bool, device=self.device)
            indices = torch.tensor(indices, device=self.device)
            mask[indices] = False
            self.feature_vectors = self.feature_vectors[mask]
            
            #print(f"Removed features. Memory size: {self.feature_vectors.shape[0]}")
        except RuntimeError as e:
            print(f"Error during removal: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise
                
    def find_nearest_neighbors(self, query_vector, batch_size=1000):
        """
        Find the indices of the k nearest neighbors for a given query vector.
        Implements batched processing for large feature sets.
        :param query_vector: The query vector for which to find the nearest neighbors
        :param batch_size: Size of batches for processing large feature sets
        :return: The indices of the k nearest neighbors
        """
        try:
            # Ensure query vector is on correct device
            query_vector = query_vector.to(self.device)
            
            if self.feature_vectors.shape[0] <= batch_size:
                # If feature set is small enough, process all at once
                distances = torch.cdist(query_vector.unsqueeze(0), self.feature_vectors).squeeze(0)
                _, indices = torch.topk(distances, min(self.k, len(distances)), largest=False)
                return indices.cpu()  # Return indices to CPU
            else:
                # For large feature sets, process in batches
                min_distances = torch.full((self.k,), float('inf'), device=self.device)
                min_indices = torch.zeros(self.k, dtype=torch.long, device=self.device)
                
                for i in range(0, self.feature_vectors.shape[0], batch_size):
                    batch = self.feature_vectors[i:i + batch_size]
                    batch_distances = torch.cdist(query_vector.unsqueeze(0), batch).squeeze(0)
                    
                    # Combine current and previous results
                    combined_distances = torch.cat([min_distances, batch_distances])
                    combined_indices = torch.cat([min_indices, 
                                               torch.arange(i, i + len(batch_distances), 
                                                          device=self.device)])
                    
                    # Update top-k
                    _, top_k_indices = torch.topk(combined_distances, self.k, largest=False)
                    min_distances = combined_distances[top_k_indices]
                    min_indices = combined_indices[top_k_indices]
                
                return min_indices.cpu()
                
        except RuntimeError as e:
            print(f"Error during neighbor search: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise
            
    def get_distances(self, batch_size=1000):
        """Calculate all pairwise distances in parallel using batched matrix operations"""
        try:
            if self.feature_vectors.shape[0] < 2:
                return {}
                
            distances_dict = {}
            num_vectors = self.feature_vectors.shape[0]

            # Normalize all vectors at once
            all_normalized = self.feature_vectors / (torch.norm(self.feature_vectors, dim=1, keepdim=True) + 1e-8)
            
            # Calculate full similarity matrix in batches
            for i in range(0, num_vectors, batch_size):
                batch_end = min(i + batch_size, num_vectors)
                
                # Calculate similarities between current batch and all vectors
                batch_similarities = torch.matmul(
                    all_normalized[i:batch_end], 
                    all_normalized.T  # Transpose for matrix multiplication
                )
                
                # Convert to distances
                batch_distances = 1 - batch_similarities
                
                # For each vector in batch, get k nearest neighbors
                # excluding self-similarity
                for j, idx in enumerate(range(i, batch_end)):
                    # Zero out self-similarity
                    distances = batch_distances[j].clone()
                    distances[idx] = float('inf')
                    
                    # Get top k
                    k = min(self.k, num_vectors - 1)
                    topk_distances, _ = torch.topk(distances, k, largest=False)
                    distances_dict[idx] = topk_distances
                    
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            return distances_dict
            
        except Exception as e:
            print(f"Error calculating distances: {e}")
            return {}

    def aggregate_distances(self, method='mean'):
        """Aggregate distances with validation"""
        try:
            if self.feature_vectors.shape[0] < 2:
                return {0: 0.0}  # Return 0 distance for single vector
                
            distances_dict = self.get_distances()
            if not distances_dict:
                return {0: 0.0}
                
            aggregated = {}
            
            for idx, distances in distances_dict.items():
                if distances.numel() == 0:
                    continue
                    
                if method == 'mean':
                    mean = torch.mean(distances).item()
                    std = torch.std(distances).item() + 1e-8  # Add small epsilon to avoid division by zero
                    agg_value = mean / std
                elif method == 'min':
                    agg_value = torch.min(distances).item()
                else:
                    agg_value = torch.mean(distances).item()
                    
                if not np.isnan(agg_value) and not np.isinf(agg_value):
                    aggregated[idx] = agg_value
                    
            if not aggregated:
                return {0: 0.0}
                
            return aggregated
            
        except Exception as e:
            print(f"Error in aggregate_distances: {e}")
            return {0: 0.0}
        
    def get_novelty(self, query_vector):
        """
        Calculate novelty score as average distance to k nearest neighbors.
        Returns a value between 0-1 where higher means more novel.
        """
        try:
            if self.feature_vectors.shape[0] < 2:
                return 0.0

            #print(self.k)
                
            # Ensure query vector is on device and has correct shape
            query_vector = query_vector.to(self.device)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.unsqueeze(0)  # Add batch dimension
                
            # Normalize query vector
            query_norm = query_vector / (torch.norm(query_vector, dim=1, keepdim=True) + 1e-8)
            
            # Find k nearest neighbors efficiently
            indices = self.find_nearest_neighbors(query_vector.squeeze(), self.k)
            if indices is None or len(indices) == 0:
                return 0.0
                
            # Get neighbors and normalize
            neighbors = self.feature_vectors[indices]
            if len(neighbors.shape) == 1:
                neighbors = neighbors.unsqueeze(0)
                
            neighbors_norm = neighbors / (torch.norm(neighbors, dim=1, keepdim=True) + 1e-8)
            
            # Calculate cosine similarities
            similarities = torch.matmul(query_norm, neighbors_norm.T)
            
            # Convert to distances (1 - similarity) and get mean
            distances = 1 - similarities
            novelty_score = torch.mean(distances).item()
            
            # Normalize to 0-1 range
            novelty_score = min(max(novelty_score, 0.0), 1.0)
            
            return novelty_score
            
        except Exception as e:
            print(f"Error calculating novelty: {e}")
            print(f"Query vector shape: {query_vector.shape}")
            print(f"Feature vectors shape: {self.feature_vectors.shape}")
            return 0.0
        
    def batch_get_novelty(self, query_vectors):
        """
        Calculate novelty scores for multiple query vectors at once.
        Args:
            query_vectors: torch.Tensor of shape (batch_size, feature_dim) or (batch_size, 1, feature_dim)
        Returns:
            torch.Tensor of shape (batch_size,) containing novelty scores
        """
        try:
            if self.feature_vectors.shape[0] < 2:
                return torch.zeros(query_vectors.shape[0], device=self.device)
            
            # Ensure query vectors are on device and have correct shape
            query_vectors = query_vectors.to(self.device)
            if len(query_vectors.shape) == 3:  # Shape: [batch_size, 1, feature_dim]
                query_vectors = query_vectors.squeeze(1)  # Convert to [batch_size, feature_dim]
            elif len(query_vectors.shape) == 1:  # Shape: [feature_dim]
                query_vectors = query_vectors.unsqueeze(0)  # Convert to [1, feature_dim]
            
            # Normalize all vectors at once (add small epsilon to avoid division by zero)
            query_norm = query_vectors / (torch.norm(query_vectors, dim=1, keepdim=True) + 1e-8)
            neighbors_norm = self.feature_vectors / (torch.norm(self.feature_vectors, dim=1, keepdim=True) + 1e-8)
            
            # Calculate all pairwise similarities at once
            similarities = torch.matmul(query_norm, neighbors_norm.T)  # shape: (batch_size, num_neighbors)
            
            # For each query, get k nearest neighbors
            _, top_k_indices = torch.topk(similarities, min(self.k, similarities.shape[1]), dim=1, largest=True)
            
            # Gather the k nearest neighbor similarities for each query
            top_k_similarities = torch.gather(similarities, 1, top_k_indices)
            
            # Convert similarities to distances and compute mean for each query
            distances = 1 - top_k_similarities
            novelty_scores = torch.mean(distances, dim=1)  # shape: [batch_size]
            
            # Normalize to 0-1 range
            novelty_scores = torch.clamp(novelty_scores, 0.0, 1.0)
            
            return novelty_scores
            
        except Exception as e:
            print(f"Error in batch_get_novelty: {e}")
            print(f"Query vectors shape: {query_vectors.shape}")
            print(f"Feature vectors shape: {self.feature_vectors.shape}")
            return torch.zeros(query_vectors.shape[0], device=self.device)
        
    def batch_get_novelty_stream(self, query_vectors, stream):
        """
        Stream-aware version of batch_get_novelty that works with CUDA streams
        Args:
            query_vectors: torch.Tensor of shape (batch_size, feature_dim) or (batch_size, 1, feature_dim)
            stream: torch.cuda.Stream instance
        Returns:
            torch.Tensor of shape (batch_size,) containing novelty scores
        """
        try:
            if self.feature_vectors.shape[0] < 2:
                return torch.zeros(query_vectors.shape[0], device=self.device)
            
            with torch.cuda.stream(stream):
                # Ensure query vectors are on device and have correct shape
                query_vectors = query_vectors.to(self.device, non_blocking=True)
                if len(query_vectors.shape) == 3:
                    query_vectors = query_vectors.squeeze(1)
                elif len(query_vectors.shape) == 1:
                    query_vectors = query_vectors.unsqueeze(0)
                
                # Normalize vectors (use separate streams for parallelization)
                query_norm = query_vectors / (torch.norm(query_vectors, dim=1, keepdim=True) + 1e-8)
                neighbors_norm = self.feature_vectors / (torch.norm(self.feature_vectors, dim=1, keepdim=True) + 1e-8)
                
                # Calculate similarities
                similarities = torch.matmul(query_norm, neighbors_norm.T)
                
                # Get k nearest neighbors
                _, top_k_indices = torch.topk(similarities, min(self.k, similarities.shape[1]), dim=1, largest=True)
                
                # Gather similarities and compute novelty
                top_k_similarities = torch.gather(similarities, 1, top_k_indices)
                distances = 1 - top_k_similarities
                novelty_scores = torch.mean(distances, dim=1)
                
                # Normalize scores
                novelty_scores = torch.clamp(novelty_scores, 0.0, 1.0)
                
                return novelty_scores
                
        except Exception as e:
            print(f"Error in batch_get_novelty_stream: {e}")
            print(f"Query vectors shape: {query_vectors.shape}")
            print(f"Feature vectors shape: {self.feature_vectors.shape}")
            return torch.zeros(query_vectors.shape[0], device=self.device)

    def add_feature_vectors(self, new_feature_vectors, step=0):
        """
        Add new feature vectors and recalculate optimal k using elbow method
        """
        try:
            new_feature_vectors = new_feature_vectors.to(self.device)
            
            # Ensure correct shape
            if len(new_feature_vectors.shape) == 1:
                new_feature_vectors = new_feature_vectors.unsqueeze(0)
            elif len(new_feature_vectors.shape) == 3:
                new_feature_vectors = new_feature_vectors.squeeze(1)
                
            # Validate new vectors
            if torch.isnan(new_feature_vectors).any():
                print("Warning: NaN values in new feature vectors")
                return
                
            if self.feature_vectors.numel() == 0:
                self.feature_vectors = new_feature_vectors
            else:
                self.feature_vectors = torch.cat((self.feature_vectors, new_feature_vectors), dim=0)
            
            #Random chance of 75% of not updating k
            if random.random() > 0.75:
                # Recalculate optimal k if we have enough vectors every 3 steps
                if step % 3 == 0 and self.feature_vectors.shape[0] >= 3:
                    if not self.last_update_step == step:
                        self.last_update_step = step
                        if self.feature_vectors.shape[0] >= 3:  # Need at least 3 points for elbow method
                            self.k = self.calculate_k_elbow()
                            #print(f"Recalculated optimal k = {self.k}")
            
        except Exception as e:
            print(f"Error adding feature vectors: {e}")

    def get_memory_usage(self):
        """
        Get current GPU memory usage.
        :return: Dictionary containing memory usage statistics
        """
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0)//1024**2,
                "cached": torch.cuda.memory_reserved(0)//1024**2
            }
        return {"allocated": 0, "cached": 0}
    
    def save(self, path):
        """
        Save the kNN instance to a file.
        :param path: The path where the kNN instance should be saved
        """
        # Move to CPU before saving
        cpu_state = {
            'k': self.k,
            'feature_vectors': self.feature_vectors.cpu()
        }
        torch.save(cpu_state, path)
        
    @staticmethod
    def load(path):
        """
        Load a kNN instance from a file.
        :param path: The path from which to load the kNN instance
        :return: The loaded kNN instance
        """
        state = torch.load(path)
        instance = kNN(k=state['k'])
        instance.feature_vectors = state['feature_vectors'].to(instance.device)
        return instance

if __name__ == "__main__":
    # Example usage
    try:
        # Create instance
        knn = kNN(k=5)
        
        # Generate test data
        test_vectors = torch.randn(1000, 64).to(knn.device)
        query = torch.randn(64).to(knn.device)
        
        # Add vectors
        knn.add_feature_vectors(test_vectors)
        
        # Find neighbors
        neighbors = knn.find_nearest_neighbors(query)
        
        print(f"Found {len(neighbors)} nearest neighbors")
        print(f"Memory usage: {knn.get_memory_usage()}")
        
    except RuntimeError as e:
        print(f"Error during testing: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()