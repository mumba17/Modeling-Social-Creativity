import torch
import numpy as np

class kNN:
    def __init__(self, k=3, agent_id=None):
        """
        Initialize the kNN instance with GPU support.
        :param k: The number of nearest neighbors to consider
        """
        self.k = k
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"kNN initialized on device: {self.device}")
        self.feature_vectors = torch.tensor([], device=self.device)
        print(f"Initialized kNN with k={k} for agent {agent_id}")
        
    def add_feature_vectors(self, new_feature_vectors):
        """Add new feature vectors with validation"""
        try:
            new_feature_vectors = new_feature_vectors.to(self.device)
            
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
            
            print(f"Removed features. Memory size: {self.feature_vectors.shape[0]}")
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
        """Calculate distances with proper error handling"""
        try:
            if self.feature_vectors.shape[0] < 2:  # Need at least 2 vectors
                return {}
                
            distances_dict = {}
            
            for i in range(self.feature_vectors.shape[0]):
                query = self.feature_vectors[i]
                
                # Create mask for all vectors except current
                mask = torch.ones(self.feature_vectors.shape[0], dtype=torch.bool, device=self.device)
                mask[i] = False
                other_vectors = self.feature_vectors[mask]
                
                if other_vectors.shape[0] == 0:
                    continue
                    
                # Calculate cosine similarity instead of Euclidean distance
                query_normalized = query / (torch.norm(query) + 1e-8)
                others_normalized = other_vectors / (torch.norm(other_vectors, dim=1, keepdim=True) + 1e-8)
                similarities = torch.matmul(others_normalized, query_normalized)
                
                # Convert to distances (1 - similarity)
                distances = 1 - similarities
                
                # Get k nearest
                k = min(self.k, len(distances))
                if k > 0:
                    topk_distances, _ = torch.topk(distances, k, largest=False)
                    distances_dict[i] = topk_distances
                    
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
                    agg_value = torch.mean(distances).item()
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