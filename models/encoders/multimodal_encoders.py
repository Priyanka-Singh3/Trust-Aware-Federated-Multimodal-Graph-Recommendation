import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer
import numpy as np

class TextEncoder(nn.Module):
    """Text encoder using TF-IDF or small embedding model"""
    
    def __init__(self, input_dim: int = 1000, output_dim: int = 128, use_transformer: bool = False):
        super(TextEncoder, self).__init__()
        
        self.use_transformer = use_transformer
        self.output_dim = output_dim
        
        if use_transformer:
            # Use a small transformer model
            self.model_name = "distilbert-base-uncased"
            self.transformer = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Project to desired dimension
            self.projection = nn.Linear(self.transformer.config.hidden_size, output_dim)
        else:
            # Simple MLP for TF-IDF features
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim)
            )
    
    def forward(self, text_features):
        if self.use_transformer:
            # text_features should be raw text strings
            if isinstance(text_features, list):
                inputs = self.tokenizer(text_features, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512)
                with torch.no_grad():
                    outputs = self.transformer(**inputs)
                # Use [CLS] token representation
                text_emb = outputs.last_hidden_state[:, 0, :]
                return self.projection(text_emb)
            else:
                # Handle tensor input (already tokenized)
                return self.projection(text_features)
        else:
            # TF-IDF features
            return self.encoder(text_features)

class ImageEncoder(nn.Module):
    """Image encoder using pretrained CNN"""
    
    def __init__(self, output_dim: int = 128, pretrained: bool = True):
        super(ImageEncoder, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze early layers if pretrained
        if pretrained:
            for param in list(self.backbone.parameters())[:-2]:
                param.requires_grad = False
        
        # Project to desired dimension
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images):
        # images should be preprocessed tensors
        with torch.no_grad():
            features = self.backbone(images)
        
        # Flatten and project
        features = features.view(features.size(0), -1)
        return self.projection(features)

class MultimodalEncoder(nn.Module):
    """Combined multimodal encoder"""
    
    def __init__(self, text_input_dim: int = 1000, image_output_dim: int = 128, 
                 text_output_dim: int = 128, fusion_dim: int = 256, 
                 use_transformer: bool = False):
        super(MultimodalEncoder, self).__init__()
        
        self.text_encoder = TextEncoder(text_input_dim, text_output_dim, use_transformer)
        self.image_encoder = ImageEncoder(image_output_dim)
        
        # Fusion layer - input dimensions should match text_output_dim + image_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(text_output_dim + image_output_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Modality weights for attention
        self.text_weight = nn.Parameter(torch.ones(1))
        self.image_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, text_features, images):
        # Encode each modality
        text_emb = self.text_encoder(text_features)
        image_emb = self.image_encoder(images)
        
        # Normalize embeddings
        text_emb = F.normalize(text_emb, p=2, dim=1)
        image_emb = F.normalize(image_emb, p=2, dim=1)
        
        # Attention-based fusion
        text_weight = torch.sigmoid(self.text_weight)
        image_weight = torch.sigmoid(self.image_weight)
        
        # Concatenate instead of weighted combination to maintain dimensions
        combined = torch.cat([text_emb, image_emb], dim=1)
        
        # Final fusion
        fused_emb = self.fusion(combined)
        
        return fused_emb, text_emb, image_emb

class UserItemEncoder(nn.Module):
    """Encoder for user and item IDs"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super(UserItemEncoder, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        return user_emb, item_emb

class RecommendationEncoder(nn.Module):
    """Complete encoder for recommendation system"""
    
    def __init__(self, num_users: int, num_items: int, text_input_dim: int = 1000,
                 multimodal_dim: int = 256, embedding_dim: int = 64, 
                 output_dim: int = 128, use_transformer: bool = False):
        super(RecommendationEncoder, self).__init__()
        
        self.user_item_encoder = UserItemEncoder(num_users, num_items, embedding_dim)
        self.multimodal_encoder = MultimodalEncoder(
            text_input_dim, embedding_dim, embedding_dim, multimodal_dim, use_transformer
        )
        
        # Final projection for recommendation
        self.final_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2 + multimodal_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, user_ids, item_ids, text_features, images):
        # Get user and item embeddings
        user_emb, item_emb = self.user_item_encoder(user_ids, item_ids)
        
        # Get multimodal embeddings
        fused_emb, text_emb, image_emb = self.multimodal_encoder(text_features, images)
        
        # Combine all embeddings
        combined = torch.cat([user_emb, item_emb, fused_emb], dim=1)
        
        # Final projection
        final_emb = self.final_projection(combined)
        
        return final_emb, {
            'user_emb': user_emb,
            'item_emb': item_emb,
            'text_emb': text_emb,
            'image_emb': image_emb,
            'fused_emb': fused_emb
        }

# Test function
def test_encoders():
    """Test the encoders with dummy data"""
    
    print("Testing encoders...")
    
    # Dummy parameters
    batch_size = 32
    num_users = 100
    num_items = 50
    text_input_dim = 1000
    
    # Dummy data
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    text_features = torch.randn(batch_size, text_input_dim)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Test individual encoders
    text_encoder = TextEncoder(text_input_dim, 128)
    image_encoder = ImageEncoder(128)
    
    text_emb = text_encoder(text_features)
    image_emb = image_encoder(images)
    
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Image embedding shape: {image_emb.shape}")
    
    # Test multimodal encoder - ensure text and image output dimensions match
    multimodal_encoder = MultimodalEncoder(text_input_dim, 128, 128)
    fused_emb, text_out, image_out = multimodal_encoder(text_features, images)
    
    print(f"Fused embedding shape: {fused_emb.shape}")
    
    # Test complete encoder
    rec_encoder = RecommendationEncoder(num_users, num_items, text_input_dim)
    final_emb, embeddings_dict = rec_encoder(user_ids, item_ids, text_features, images)
    
    print(f"Final embedding shape: {final_emb.shape}")
    print("All encoders working correctly!")

if __name__ == "__main__":
    test_encoders()
