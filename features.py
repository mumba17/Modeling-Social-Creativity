import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self, output_dims=64):
        super(FeatureExtractor, self).__init__()
        
        # Determine device availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)//1024**2}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)//1024**2}MB")
        
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model = self.model.to(self.device) 
                
        # Define feature extraction layers
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3
        ).to(self.device) 
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)).to(self.device) 
        
        # Calculate feature dimensions using GPU if available
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 48, 48).to(self.device)
            dummy_features = self.features(dummy_input)
            dummy_pooled = self.pooling(dummy_features)
            dummy_flat = torch.flatten(dummy_pooled, 1)
            self.feature_dims = dummy_flat.shape[1]
        
        # Dimension reduction layers
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_dims, self.feature_dims // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dims // 2, output_dims)
        ).to(self.device)
        
        # Move model to appropriate device
        self.to(self.device)
            
    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)
        features = self.features(x)
        features = self.pooling(features)
        features = torch.flatten(features, 1)
        features = self.dim_reduction(features)
        return features

    @torch.no_grad()
    def extract_features(self, x):
        self.eval()
        # Handle both single images and batches
        with torch.no_grad():
            # Convert PIL Image to tensor if needed
            if isinstance(x, Image.Image):
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((48, 48)),
                    torchvision.transforms.ToTensor(),
                ])
                x = transform(x).unsqueeze(0)
                
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        features = self.forward(x)
        return features.cpu()  # Return to CPU for further processing

    def get_memory_usage(self):
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0)//1024**2,
                "cached": torch.cuda.memory_reserved(0)//1024**2
            }
        return {"allocated": 0, "cached": 0}

if __name__ == "__main__":
    # Set up model
    extractor = FeatureExtractor(output_dims=48)
    
    # Test single image
    try:
        image = torch.randn(1, 3, 48, 48)
        features = extractor.extract_features(image)
        print(f"Single image feature shape: {features.shape}")
        
        # Print memory usage
        memory_usage = extractor.get_memory_usage()
        print("\nCurrent GPU Memory Usage:")
        print(f"Allocated: {memory_usage['allocated']}MB")
        print(f"Cached: {memory_usage['cached']}MB")
        
    except RuntimeError as e:
        print(f"Error during processing: {e}")
        if "out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cleared GPU cache. Consider reducing batch size or model size.")