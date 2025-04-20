import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import pearsonr

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
TRAIN_IMAGES_DIR = 'images/train'
TEST_IMAGES_DIR = 'images/test'
TRAIN_RESPONSES_PATH = 'neural_responses_train.npy'

# Parameters
BATCH_SIZE = 10
IMAGE_SIZE = 224

# Load neural responses
train_responses = np.load(TRAIN_RESPONSES_PATH)

# Create dataset
class NeuralResponseDataset(Dataset):
    def __init__(self, image_dir, responses=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.responses = responses
        self.transform = transform
        self.is_test = is_test
        
        # Get sorted list of image files
        self.image_files = sorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Return image and response if training, just image if testing
        if self.is_test:
            return image, img_name
        else:
            return image, self.responses[idx]

# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def extract_features():
    # Load pre-trained ResNet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-2])  # Remove last two layers
    model.eval()
    
    # Create datasets
    train_dataset = NeuralResponseDataset(
        image_dir=TRAIN_IMAGES_DIR,
        responses=train_responses,
        transform=transform
    )
    
    test_dataset = NeuralResponseDataset(
        image_dir=TEST_IMAGES_DIR,
        transform=transform,
        is_test=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract features for training set
    train_features = []
    train_targets = []
    
    with torch.no_grad():
        for images, responses in train_loader:
            images = images.to(device)
            features = model(images)
            # Global average pooling
            features = torch.mean(features, dim=[2, 3])
            train_features.append(features.cpu().numpy())
            train_targets.append(responses.numpy())
    
    train_features = np.vstack(train_features)
    train_targets = np.vstack(train_targets)
    
    # Extract features for test set
    test_features = []
    test_filenames = []
    
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            features = model(images)
            # Global average pooling
            features = torch.mean(features, dim=[2, 3])
            test_features.append(features.cpu().numpy())
            test_filenames.extend(filenames)
    
    test_features = np.vstack(test_features)
    
    return train_features, train_targets, test_features, test_filenames

def train_ridge_models(train_features, train_targets):
    # Train separate Ridge models for each neuron
    ridge_models = []
    correlation_scores = []
    
    for n in range(train_targets.shape[1]):
        print(f"Training Ridge model for Neuron {n+1}")
        
        # Try different alpha values
        best_alpha = 1.0
        best_score = -float('inf')
        
        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]:
            # Define pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha))
            ])
            
            # Fit model and calculate Pearson correlation
            pipeline.fit(train_features, train_targets[:, n])
            predictions = pipeline.predict(train_features)
            correlation, _ = pearsonr(predictions, train_targets[:, n])
            
            print(f"  Alpha={alpha}, Pearson r: {correlation:.4f}")
            
            if correlation > best_score:
                best_score = correlation
                best_alpha = alpha
        
        # Train final model with best alpha
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=best_alpha))
        ])
        final_pipeline.fit(train_features, train_targets[:, n])
        
        ridge_models.append(final_pipeline)
        correlation_scores.append(best_score)
        
        print(f"Neuron {n+1}: Best alpha={best_alpha}, Best Pearson r: {best_score:.4f}")
    
    return ridge_models, correlation_scores

def generate_predictions(ridge_models, test_features):
    # Generate predictions for each neuron
    test_preds = np.zeros((test_features.shape[0], len(ridge_models)))
    
    for n, model in enumerate(ridge_models):
        test_preds[:, n] = model.predict(test_features)
    
    return test_preds

if __name__ == "__main__":
    # Extract features
    print("Extracting features...")
    train_features, train_targets, test_features, test_filenames = extract_features()
    
    # Train Ridge models
    print("\nTraining Ridge models...")
    ridge_models, correlation_scores = train_ridge_models(train_features, train_targets)
    
    # Generate predictions
    print("\nGenerating predictions...")
    test_preds = generate_predictions(ridge_models, test_features)
    
    # Save predictions
    np.save('neural_responses_test_pred.npy', test_preds)
    
    # Save filenames for reference
    with open('test_image_order.txt', 'w') as f:
        for filename in test_filenames:
            f.write(f"{filename}\n")
    
    # Visualize predictions
    plt.figure(figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    
    for i in range(5):  # Show first 5 test samples
        plt.subplot(1, 5, i+1)
        bars = plt.bar(range(test_preds.shape[1]), test_preds[i], color=colors)
        plt.title(f'Image {i+1}')
        plt.xlabel('Neuron')
        plt.ylabel('Response')
        plt.xticks(range(test_preds.shape[1]), [f'{j+1}' for j in range(test_preds.shape[1])])
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_predictions_samples.png')
    plt.show()
    
    # Print average Pearson correlation across neurons
    mean_correlation = np.mean(correlation_scores)
    print(f"\nAverage Pearson correlation across neurons: {mean_correlation:.4f}")