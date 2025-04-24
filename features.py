import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from PIL import Image

from timing_utils import time_it

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeatureExtractor(nn.Module):
    """
    A ResNet-based feature extraction module with a dimension-reduction head.
    Extracts features from images or image batches.
    """
    def __init__(self, output_dims=64):
        super(FeatureExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Allocated: {torch.cuda.memory_allocated(0)//1024**2}MB, "
                        f"Reserved: {torch.cuda.memory_reserved(0)//1024**2}MB")

        # Load a ResNet-18 pre-trained
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Adjust initial conv to smaller kernel for 32x32
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model = self.model.to(self.device)

        # Extract only certain layers
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.layer1,
            self.model.layer2
        ).to(self.device)

        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

        self.pooling = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)

        # Determine feature dimensions by passing dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            dummy_features = self.features(dummy_input)
            dummy_pooled = self.pooling(dummy_features)
            dummy_flat = torch.flatten(dummy_pooled, 1)
            self.feature_dims = dummy_flat.shape[1]

        self.feature_norm = nn.BatchNorm1d(self.feature_dims).to(self.device)

        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_dims, self.feature_dims // 2),
            nn.BatchNorm1d(self.feature_dims // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dims // 2, output_dims),
            nn.BatchNorm1d(output_dims),
            nn.LayerNorm(output_dims, elementwise_affine=False)
        ).to(self.device)
        
        # Set up feature cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000  # Maximum cache entries

    @time_it
    def standardize_features(self, features):
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True) + 1e-8
        return (features - mean) / std

    @time_it
    def forward(self, x):
        x = x.to(self.device)
        
        # Process batch efficiently
        batch_size = x.size(0)
        if batch_size > 1:
            # For batches, process all at once
            features = self.features(x)
            features = self.pooling(features)
            features = torch.flatten(features, 1)
            features = self.feature_norm(features)
            features = self.standardize_features(features)
            features = self.dim_reduction(features)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features
        
        # For single image, check cache first (only cache single images)
        img_hash = hash(x.cpu().numpy().tobytes())
        if img_hash in self.cache:
            self.cache_hits += 1
            return self.cache[img_hash].to(self.device)
            
        self.cache_misses += 1
        features = self.features(x)
        features = self.pooling(features)
        features = torch.flatten(features, 1)
        features = self.feature_norm(features)
        features = self.standardize_features(features)
        features = self.dim_reduction(features)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # Store in cache if not too large
        if len(self.cache) < self.max_cache_size:
            self.cache[img_hash] = features.cpu()
        
        return features

    @torch.no_grad()
    @time_it
    def extract_features(self, x):
        """
        Extract features from a PIL Image or a torch.Tensor.

        Parameters
        ----------
        x : PIL.Image or torch.Tensor
            If PIL.Image, it is converted to (1,3,32,32).
            If torch.Tensor, expected shape is [N,3,H,W] or [3,H,W].

        Returns
        -------
        torch.Tensor
            Extracted and normalized feature vectors on CPU.
        """
        self.eval()
        
        # For PIL images, hash the image and check cache
        if isinstance(x, Image.Image):
            img_hash = hash(x.tobytes())
            if img_hash in self.cache:
                self.cache_hits += 1
                return self.cache[img_hash]
                
            self.cache_misses += 1
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            x = transform(x).unsqueeze(0)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        feats = self.forward(x)
        
        # Store in cache if single image and cache not too large
        if isinstance(x, Image.Image) and x.dim() == 4 and x.size(0) == 1:
            if len(self.cache) < self.max_cache_size:
                self.cache[img_hash] = feats.cpu()
        
        return feats.cpu()
    
    def clear_cache(self):
        """
        Clear the feature cache to free memory
        """
        cache_size = len(self.cache)
        self.cache = {}
        logger.info(f"Cleared feature cache with {cache_size} entries. Hits: {self.cache_hits}, Misses: {self.cache_misses}")
        
    def get_cache_stats(self):
        """
        Get cache statistics
        """
        return {
            "size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses + 1e-8)
        }

    def get_memory_usage(self):
        """
        Returns the current GPU memory usage as dict.
        """
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0)//1024**2,
                "reserved": torch.cuda.memory_reserved(0)//1024**2
            }
        return {"allocated": 0, "reserved": 0}


if __name__ == "__main__":
    extractor = FeatureExtractor(output_dims=32)
    image = torch.randn(1, 3, 32, 32)
    features = extractor.extract_features(image)
    logger.info(f"Feature shape: {features.shape}")
    usage = extractor.get_memory_usage()
    logger.info(f"Memory usage: {usage}")
