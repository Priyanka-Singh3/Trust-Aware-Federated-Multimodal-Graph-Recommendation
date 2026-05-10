import os
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import glob

def extract_image_features():
    print("Starting ResNet18 feature extraction for Yelp images...")
    
    # 1. Load Pretrained ResNet18 backbone
    # Remove the final classification layer to get 512-dim features
    print("Loading pretrained ResNet18...")
    resnet = models.resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval()
    
    # Move to GPU/MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    backbone = backbone.to(device)
    
    # 2. Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Process Images
    img_dir = "data/raw/yelp_multimodal_final/images"
    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    
    print(f"Found {len(image_paths)} images. Extracting 512-d features...")
    
    features_dict = {}
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting"):
            photo_id = os.path.basename(path).replace(".jpg", "")
            try:
                # Load and convert to RGB (to handle greyscale/RGBA)
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Forward pass
                feature = backbone(img_tensor)
                feature = feature.view(-1).cpu() # Flatten to 1D (512,)
                
                features_dict[photo_id] = feature
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
    # 4. Save Features
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/image_features.pt"
    torch.save(features_dict, save_path)
    
    print(f"✅ Successfully extracted features for {len(features_dict)} images.")
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    extract_image_features()
