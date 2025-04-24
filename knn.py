import logging
import torch
import numpy as np
import random

from timing_utils import time_it
import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class kNN:
    """
    k-Nearest Neighbors data structure that supports:
      - Adding feature vectors
      - Calculating novelty as average distance to k nearest neighbors
      - Automatic k update via the elbow method
    """
    def __init__(self, agent_id=None):
        """
        Parameters
        ----------
        agent_id : int, optional
            Identifier for debugging/logging
        """
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_vectors = torch.tensor([], device=self.device)
        self.k = config.KNN_DEFAULT_K
        self.last_update_step = -1
        self.random_gen = random.Random(agent_id)
        self.random_threshold = self.random_gen.random()
        
        # Add support for approximate kNN
        self.use_approx = False
        if hasattr(config, 'ENABLE_KNN_APPROX'):
            self.use_approx = config.ENABLE_KNN_APPROX
            
        # Index for fast lookups - lazily initialized when needed
        self._index = None
        
        # Cache distances between common feature vectors
        self._distance_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Control frequency of k updates
        self.k_update_frequency = 3
        if hasattr(config, 'KNN_UPDATE_FREQUENCY'):
            self.k_update_frequency = config.KNN_UPDATE_FREQUENCY

        logger.debug(f"Initialized kNN for agent {agent_id} with k={self.k}")

    @time_it
    def get_max_k(self, n_samples, n_features):
        """
        Determine maximum feasible k value based on dataset size.

        Parameters
        ----------
        n_samples : int
        n_features : int

        Returns
        -------
        int
            Maximum recommended k
        """
        theoretical_max = int(np.sqrt(n_samples/2))
        absolute_max = min(
            theoretical_max,
            n_samples - 1,
            100
        )
        return max(3, absolute_max)

    @time_it
    def generate_k_values(self, max_k):
        """
        Generate candidate k values (logarithmic spacing for large k).

        Parameters
        ----------
        max_k : int

        Returns
        -------
        list
            Sequence of candidate k values
        """
        if max_k <= 3:
            return [3]
        if max_k <= 10:
            return list(range(3, max_k+1))

        k_values = [3]
        if max_k <= 20:
            k_values.extend(range(4, max_k + 1, 2))
        else:
            log_space = np.logspace(
                np.log10(3),
                np.log10(max_k),
                min(15, max_k - 2)
            ).astype(int)
            k_values.extend(range(3, 12, 2))
            k_values = sorted(set(k_values + list(log_space)))
            if len(k_values) < 5:
                step = max(2, (max_k - 3) // 10)
                k_values = list(range(3, max_k + 1, step))

        if k_values[-1] != max_k:
            k_values.append(max_k)
        return k_values

    @time_it
    def calculate_k_elbow(self):
        """
        Use elbow method to find optimal k by computing "distortion" over sample set.

        Returns
        -------
        int
            Optimal k for this dataset
        """
        try:
            n_samples = self.feature_vectors.shape[0]
            if n_samples < 2:
                return config.KNN_DEFAULT_K

            max_k = self.get_max_k(n_samples, self.feature_vectors.shape[1])
            k_values = self.generate_k_values(max_k)
            if not k_values:
                return config.KNN_DEFAULT_K

            MAX_SAMPLE_SIZE = 1000
            if n_samples > MAX_SAMPLE_SIZE:
                indices = torch.randperm(n_samples)[:MAX_SAMPLE_SIZE]
                vectors_sample = self.feature_vectors[indices]
            else:
                vectors_sample = self.feature_vectors

            distortions = torch.zeros(len(k_values), device=self.device)
            normalized_vectors = vectors_sample / (
                torch.norm(vectors_sample, dim=1, keepdim=True) + 1e-8
            )
            similarities = torch.matmul(normalized_vectors, normalized_vectors.T)
            distances = 1 - similarities
            distances.fill_diagonal_(float('inf'))

            for idx, k in enumerate(k_values):
                top_k_distances, _ = torch.topk(distances, k, largest=False, dim=1)
                distortion = torch.mean(top_k_distances).item()
                distortions[idx] = distortion
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            coords = torch.stack([torch.tensor(k_values, device=self.device), distortions], dim=1)
            coords_min = coords.min(0)[0]
            coords_max = coords.max(0)[0]
            coords_normalized = (coords - coords_min) / (coords_max - coords_min + 1e-12)

            vectors = coords_normalized[1:] - coords_normalized[:-1]
            angles = []
            for i in range(len(vectors) - 1):
                v1 = vectors[i]
                v2 = vectors[i + 1]
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-12)
                angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
                angles.append(angle.item())
            if not angles:
                return config.KNN_DEFAULT_K

            optimal_idx = angles.index(max(angles)) + 1
            optimal_k = k_values[optimal_idx]
            return optimal_k

        except Exception as e:
            logger.error(f"Error in calculate_k_elbow: {e}")
            return config.KNN_DEFAULT_K

    def remove_feature_vectors(self, indices):
        """
        Remove feature vectors by index from local memory.
        """
        try:
            if not self.feature_vectors.shape[0]:
                return
            mask = torch.ones(self.feature_vectors.shape[0], dtype=torch.bool, device=self.device)
            indices = torch.tensor(indices, device=self.device)
            mask[indices] = False
            self.feature_vectors = self.feature_vectors[mask]
            
            # Invalidate the index
            self._index = None
            
            # Clear distance cache when removing vectors
            self._distance_cache = {}
        except RuntimeError as e:
            logger.error(f"Error during removal: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise

    def _build_index(self):
        """
        Build an index for approximate nearest neighbor search.
        Only builds if we're using the approximate mode.
        """
        if not self.use_approx or self._index is not None:
            return
            
        if self.feature_vectors.shape[0] < 50:
            # Don't bother with index for small datasets
            return
            
        try:
            # Import faiss only if we need it
            import faiss
            
            # Move data to CPU for faiss
            vectors_cpu = self.feature_vectors.cpu().numpy()
            
            # Normalize vectors
            norms = np.linalg.norm(vectors_cpu, axis=1, keepdims=True)
            normalized_vectors = vectors_cpu / (norms + 1e-8)
            
            # Build index
            d = normalized_vectors.shape[1]
            index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity for normalized vectors
            index.add(normalized_vectors)
            self._index = (index, normalized_vectors)
            
        except ImportError:
            logger.warning("FAISS library not available, falling back to brute force kNN")
            self.use_approx = False
        except Exception as e:
            logger.error(f"Error building index: {e}")
            self.use_approx = False
            
    @time_it
    def _find_neighbors_approx(self, query_vector, k):
        """
        Find nearest neighbors using approximate search.
        """
        if self._index is None:
            self._build_index()
            
        if self._index is None:
            # If index building failed, fall back to brute force
            return self.find_nearest_neighbors(query_vector, k)
            
        try:
            # Convert query to numpy
            query_np = query_vector.cpu().numpy().reshape(1, -1)
            
            # Normalize query
            query_norm = query_np / (np.linalg.norm(query_np) + 1e-8)
            
            # Search in index
            index, _ = self._index
            _, indices = index.search(query_norm, min(k, index.ntotal))
            
            return torch.tensor(indices[0], device=self.device)
            
        except Exception as e:
            logger.error(f"Error in approximate search: {e}")
            return self.find_nearest_neighbors(query_vector, k)

    @time_it
    def find_nearest_neighbors(self, query_vector, batch_size=1000):
        """
        Find indices of k nearest neighbors for query_vector.
        """
        # Use approximate search if enabled and index exists
        if self.use_approx and self.feature_vectors.shape[0] >= 50:
            return self._find_neighbors_approx(query_vector, self.k)
            
        # Fall back to brute force method
        try:
            query_vector = query_vector.to(self.device)
            if self.feature_vectors.shape[0] <= batch_size:
                distances = torch.cdist(query_vector.unsqueeze(0), self.feature_vectors).squeeze(0)
                _, indices = torch.topk(distances, min(self.k, len(distances)), largest=False)
                return indices.cpu()
            else:
                min_distances = torch.full((self.k,), float('inf'), device=self.device)
                min_indices = torch.zeros(self.k, dtype=torch.long, device=self.device)
                for i in range(0, self.feature_vectors.shape[0], batch_size):
                    batch = self.feature_vectors[i:i + batch_size]
                    batch_distances = torch.cdist(query_vector.unsqueeze(0), batch).squeeze(0)
                    combined_distances = torch.cat([min_distances, batch_distances])
                    combined_indices = torch.cat([
                        min_indices,
                        torch.arange(i, i + len(batch_distances), device=self.device)
                    ])
                    _, top_k_indices = torch.topk(combined_distances, self.k, largest=False)
                    min_distances = combined_distances[top_k_indices]
                    min_indices = combined_indices[top_k_indices]
                return min_indices.cpu()

        except RuntimeError as e:
            logger.error(f"Error during neighbor search: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise

    def _get_cache_key(self, query_vector):
        """
        Generate a cache key for the query vector.
        """
        if len(query_vector.shape) == 1:
            vec = query_vector
        else:
            vec = query_vector.reshape(-1)
        return hash(vec.cpu().numpy().tobytes())

    @time_it
    def get_novelty(self, query_vector):
        """
        Compute novelty as average distance to k nearest neighbors.
        """
        try:
            if self.feature_vectors.shape[0] < 2:
                return 0.0

            # Try cache first
            cache_key = self._get_cache_key(query_vector)
            if cache_key in self._distance_cache:
                self._cache_hits += 1
                return self._distance_cache[cache_key]
                
            self._cache_misses += 1
                
            query_vector = query_vector.to(self.device)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.unsqueeze(0)

            query_norm = query_vector / (torch.norm(query_vector, dim=1, keepdim=True) + 1e-8)
            indices = self.find_nearest_neighbors(query_vector.squeeze(), self.k)
            if indices is None or len(indices) == 0:
                return 0.0

            neighbors = self.feature_vectors[indices]
            if len(neighbors.shape) == 1:
                neighbors = neighbors.unsqueeze(0)

            neighbors_norm = neighbors / (torch.norm(neighbors, dim=1, keepdim=True) + 1e-8)
            similarities = torch.matmul(query_norm, neighbors_norm.T)
            distances = 1 - similarities
            novelty_score = torch.mean(distances).item()
            novelty_score = min(max(novelty_score, 0.0), 1.0)
            
            # Store in cache
            self._distance_cache[cache_key] = novelty_score
            
            # Limit cache size
            if len(self._distance_cache) > 10000:
                # Remove random 20% of entries
                keys = list(self._distance_cache.keys())
                to_remove = random.sample(keys, int(len(keys) * 0.2))
                for key in to_remove:
                    del self._distance_cache[key]
                    
            return novelty_score

        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 0.0

    @time_it
    def batch_get_novelty(self, query_vectors):
        """
        Batch version of get_novelty for multiple queries.
        """
        try:
            if self.feature_vectors.shape[0] < 2:
                return torch.zeros(query_vectors.shape[0], device=self.device)

            query_vectors = query_vectors.to(self.device)
            if len(query_vectors.shape) == 3:
                query_vectors = query_vectors.squeeze(1)
            elif len(query_vectors.shape) == 1:
                query_vectors = query_vectors.unsqueeze(0)

            query_norm = query_vectors / (torch.norm(query_vectors, dim=1, keepdim=True) + 1e-8)
            neighbors_norm = self.feature_vectors / (torch.norm(self.feature_vectors, dim=1, keepdim=True) + 1e-8)
            similarities = torch.matmul(query_norm, neighbors_norm.T)
            topk_sim, _ = torch.topk(similarities, min(self.k, similarities.shape[1]), dim=1, largest=True)
            distances = 1 - topk_sim
            novelty_scores = torch.mean(distances, dim=1)
            novelty_scores = torch.clamp(novelty_scores, 0.0, 1.0)
            return novelty_scores

        except Exception as e:
            logger.error(f"Error in batch_get_novelty: {e}")
            return torch.zeros(query_vectors.shape[0], device=self.device)

    @time_it
    def batch_get_novelty_stream(self, query_vectors, stream):
        """
        CUDA stream-aware batch novelty for parallel processing.
        """
        try:
            if self.feature_vectors.shape[0] < 2:
                return torch.zeros(query_vectors.shape[0], device=self.device)

            with torch.cuda.stream(stream):
                query_vectors = query_vectors.to(self.device, non_blocking=True)
                if len(query_vectors.shape) == 3:
                    query_vectors = query_vectors.squeeze(1)
                elif len(query_vectors.shape) == 1:
                    query_vectors = query_vectors.unsqueeze(0)

                # Check cache for each vector in the batch
                novelty_scores = torch.zeros(query_vectors.shape[0], device=self.device)
                uncached_indices = []
                
                for i, query in enumerate(query_vectors):
                    cache_key = self._get_cache_key(query)
                    if cache_key in self._distance_cache:
                        self._cache_hits += 1
                        novelty_scores[i] = self._distance_cache[cache_key]
                    else:
                        self._cache_misses += 1
                        uncached_indices.append(i)
                        
                # If all queries were cached, return early
                if not uncached_indices:
                    return novelty_scores
                    
                # Process only uncached queries
                uncached_queries = query_vectors[uncached_indices]
                query_norm = uncached_queries / (torch.norm(uncached_queries, dim=1, keepdim=True) + 1e-8)
                neighbors_norm = self.feature_vectors / (torch.norm(self.feature_vectors, dim=1, keepdim=True) + 1e-8)

                similarities = torch.matmul(query_norm, neighbors_norm.T)
                _, top_k_indices = torch.topk(similarities, min(self.k, similarities.shape[1]), dim=1, largest=True)
                top_k_similarities = torch.gather(similarities, 1, top_k_indices)
                distances = 1 - top_k_similarities
                uncached_scores = torch.mean(distances, dim=1)
                uncached_scores = torch.clamp(uncached_scores, 0.0, 1.0)
                
                # Update cache
                for i, idx in enumerate(uncached_indices):
                    cache_key = self._get_cache_key(query_vectors[idx])
                    score = uncached_scores[i].item()
                    self._distance_cache[cache_key] = score
                    novelty_scores[idx] = score
                
                # Limit cache size occasionally
                if random.random() < 0.01 and len(self._distance_cache) > 10000:
                    keys = list(self._distance_cache.keys())
                    to_remove = random.sample(keys, int(len(keys) * 0.2))
                    for key in to_remove:
                        del self._distance_cache[key]
                
                return novelty_scores

        except Exception as e:
            logger.error(f"Error in batch_get_novelty_stream: {e}")
            return torch.zeros(query_vectors.shape[0], device=self.device)

    @time_it
    def add_feature_vectors(self, new_feature_vectors, step=0):
        """
        Add new vectors to the kNN dataset. Possibly recalc k using elbow method.
        """
        try:
            new_feature_vectors = new_feature_vectors.to(self.device)
            if len(new_feature_vectors.shape) == 1:
                new_feature_vectors = new_feature_vectors.unsqueeze(0)
            elif len(new_feature_vectors.shape) == 3:
                new_feature_vectors = new_feature_vectors.squeeze(1)

            if torch.isnan(new_feature_vectors).any():
                logger.warning("NaN values detected in new feature vectors.")
                return

            if self.feature_vectors.numel() == 0:
                self.feature_vectors = new_feature_vectors
            else:
                self.feature_vectors = torch.cat((self.feature_vectors, new_feature_vectors), dim=0)

            # Invalidate the index - will be rebuilt when needed
            self._index = None
            
            # Clear some of the cache occasionally
            if random.random() < 0.1 and len(self._distance_cache) > 5000:
                keys = list(self._distance_cache.keys())
                to_remove = random.sample(keys, int(len(keys) * 0.2))
                for key in to_remove:
                    del self._distance_cache[key]

            # Update k less frequently based on config
            should_update = (step % self.k_update_frequency == 0) and self.random_threshold > 0.7
            
            self.random_threshold = self.random_gen.random()
            if should_update:
                if step % self.k_update_frequency == 0 and self.feature_vectors.shape[0] >= 3:
                    if self.last_update_step != step:
                        self.last_update_step = step
                        self.k = self.calculate_k_elbow()
                        logger.debug(f"Recalculated optimal k={self.k} for agent {self.agent_id}")

        except Exception as e:
            logger.error(f"Error adding feature vectors: {e}")

    def clear_caches(self):
        """
        Clear all internal caches.
        """
        self._distance_cache = {}
        self._index = None
        logger.debug(f"Cleared kNN caches for agent {self.agent_id}. Hits: {self._cache_hits}, Misses: {self._cache_misses}")

    def get_memory_usage(self):
        """
        Return current GPU memory usage stats.
        """
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0)//1024**2,
                "reserved": torch.cuda.memory_reserved(0)//1024**2
            }
        return {"allocated": 0, "reserved": 0}

    def save(self, path):
        """
        Save kNN state to disk (moves feature_vectors to CPU first).
        """
        cpu_state = {
            'k': self.k,
            'feature_vectors': self.feature_vectors.cpu()
        }
        torch.save(cpu_state, path)

    @staticmethod
    def load(path):
        """
        Load kNN state from disk.
        """
        state = torch.load(path)
        instance = kNN()
        instance.k = state['k']
        instance.feature_vectors = state['feature_vectors'].to(instance.device)
        return instance


if __name__ == "__main__":
    # Basic usage example
    try:
        knn = kNN(agent_id=0)
        test_vectors = torch.randn(1000, 64).to(knn.device)
        query = torch.randn(64).to(knn.device)
        knn.add_feature_vectors(test_vectors)
        neighbors = knn.find_nearest_neighbors(query)
        logger.info(f"Found {len(neighbors)} nearest neighbors for query.")
        logger.info(f"Memory usage: {knn.get_memory_usage()}")
    except RuntimeError as e:
        logger.error(f"Error in knn testing: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
