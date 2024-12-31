import torch
import numpy as np

class kNN:
    def __init__(self, agent_id=None):
        """
        Initialize the kNN instance with GPU support.
        :param k: The number of nearest neighbors to consider
        """
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_vectors = torch.tensor([], device=self.device)
        self.k = min(max(int(self.feature_vectors.shape[0] * 0.1), 5), 50)  # 10% of memory size, bounded between 5 and 50
        print(f"Initialized kNN with k={self.k} for agent {agent_id}")
            
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
                
            self.k = min(max(int(self.feature_vectors.shape[0] * 0.1), 5), 50)
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
                
            self.k = min(max(int(self.feature_vectors.shape[0] * 0.1), 5), 50)
            
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
                
            self.k = min(max(int(self.feature_vectors.shape[0] * 0.1), 5), 50)
            
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

    def add_feature_vectors(self, new_feature_vectors):
        """Add new feature vectors with validation"""
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