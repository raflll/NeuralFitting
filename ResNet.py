import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
TRAIN_IMAGES_DIR = 'images/train'
TEST_IMAGES_DIR = 'images/test'
TRAIN_RESPONSES_PATH = 'neural_responses_train.npy'

# Parameters
NUM_NEURONS = 5
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_FOLDS = 5
IMAGE_SIZE = 224
WEIGHT_DECAY = 1e-5

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

# ResNet-based neural response predictor
class ResNetEncoder(nn.Module):
    def __init__(self, num_neurons=5):
        super(ResNetEncoder, self).__init__()
        
        # Load pre-trained ResNet
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Freeze the backbone parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Global pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head for each neuron
        self.regression_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_neurons)
        )
        
    def forward(self, x):
        # Get features from ResNet
        features = self.features(x)
        
        # Apply global average pooling
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        
        # Predict responses
        responses = self.regression_head(pooled)
        
        return responses

# Cross-validation and training
def train_and_validate():
    # Prepare datasets
    full_train_dataset = NeuralResponseDataset(
        image_dir=TRAIN_IMAGES_DIR,
        responses=train_responses,
        transform=transform
    )
    
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # Lists to store results
    fold_r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_train_dataset)):
        print(f"Fold {fold+1}/{NUM_FOLDS}")
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=train_subsampler
        )
        
        val_loader = DataLoader(
            full_train_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_subsampler
        )
        
        # Initialize model, loss function, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNetEncoder(num_neurons=NUM_NEURONS).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_r2 = -float('inf')
        best_model_state = None
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            
            for images, responses in train_loader:
                images = images.to(device)
                responses = responses.to(device).float()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, responses)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
                model.eval()
                all_preds = []
                all_targets = []
                val_loss = 0.0
                
                with torch.no_grad():
                    for images, responses in val_loader:
                        images = images.to(device)
                        responses = responses.to(device).float()
                        
                        outputs = model(images)
                        loss = criterion(outputs, responses)
                        val_loss += loss.item()
                        
                        all_preds.append(outputs.cpu().numpy())
                        all_targets.append(responses.cpu().numpy())
                
                # Calculate R² score
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                r2 = r2_score(all_targets, all_preds)
                
                # Update learning rate based on validation R²
                scheduler.step(r2)
                
                # Save best model
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_state = model.state_dict().copy()
                
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}, R² Score: {r2:.4f}")
        
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        
        # Calculate final validation R² for this fold
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, responses in val_loader:
                images = images.to(device)
                responses = responses.float()
                
                outputs = model(images)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(responses.numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Calculate R² for each neuron
        neuron_r2 = []
        for n in range(NUM_NEURONS):
            r2 = r2_score(all_targets[:, n], all_preds[:, n])
            neuron_r2.append(r2)
            print(f"Fold {fold+1}, Neuron {n+1} R²: {r2:.4f}")
        
        fold_r2 = np.mean(neuron_r2)
        fold_r2_scores.append(fold_r2)
        print(f"Fold {fold+1} Mean R²: {fold_r2:.4f}")
        
        # Save the model
        torch.save(model.state_dict(), f'resnet_encoder_fold_{fold+1}.pth')

    # Print average R² across all folds
    mean_r2 = np.mean(fold_r2_scores)
    print(f"Average R² across all folds: {mean_r2:.4f}")
    
    return fold_r2_scores

# Generate predictions for test set
def generate_test_predictions(best_fold):
    # Load test dataset
    test_dataset = NeuralResponseDataset(
        image_dir=TEST_IMAGES_DIR,
        transform=transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Load the best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetEncoder(num_neurons=NUM_NEURONS).to(device)
    model.load_state_dict(torch.load(f'resnet_encoder_fold_{best_fold}.pth'))
    model.eval()
    
    # Generate predictions
    all_preds = []
    all_filenames = []
    
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Combine predictions
    all_preds = np.vstack(all_preds)
    
    # Save predictions
    np.save('neural_responses_test_pred.npy', all_preds)
    
    # Save filenames for reference
    with open('test_image_order.txt', 'w') as f:
        for filename in all_filenames:
            f.write(f"{filename}\n")
    
    print(f"Generated predictions for {len(all_filenames)} test images")
    return all_preds

if __name__ == "__main__":
    # Train and validate the ResNet model
    fold_r2_scores = train_and_validate()
    
    # Find the best fold
    best_fold = np.argmax(fold_r2_scores) + 1
    print(f"Best fold: {best_fold} with R² = {fold_r2_scores[best_fold-1]:.4f}")
    
    # Generate test predictions using the best fold
    test_preds = generate_test_predictions(best_fold)
    
    # Enhanced visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Cross-Validation Performance Plot
    plt.subplot(2, 1, 1)
    plt.bar(range(1, len(fold_r2_scores) + 1), fold_r2_scores, color='blue')
    plt.axhline(y=np.mean(fold_r2_scores), color='r', linestyle='--', label='Mean R²')
    plt.xlabel('Fold Number')
    plt.ylabel('R² Score')
    plt.title('ResNet Model Cross-Validation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, score in enumerate(fold_r2_scores):
        plt.text(i + 1, score, f'{score:.4f}', ha='center', va='bottom')
    
    # 2. Neural Response Predictions
    plt.subplot(2, 1, 2)
    for i in range(5):  # Show first 5 test samples
        plt.plot(range(NUM_NEURONS), test_preds[i], 
                marker='o', label=f'Test Image {i+1}')
    
    plt.xlabel('Neuron Index (1 to 5)')
    plt.ylabel('Predicted Neural Response')
    plt.title('ResNet Model: Predicted Neural Responses for Test Images')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                "Top: Cross-validation performance across folds. Higher R² scores indicate better predictions.\n"
                "Bottom: Predicted neural responses for 5 test images. Each line shows how different neurons respond to the same image.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('resnet_model_performance.png')
    plt.show()
    
    # Print detailed explanation
    print("\nVisualization Explanation:")
    print("1. Cross-Validation Performance Plot (Top):")
    print("   - Shows the R² score for each fold in cross-validation")
    print("   - The red dashed line shows the mean R² score across all folds")
    print("   - Higher R² scores (closer to 1) indicate better predictions")
    print("\n2. Neural Response Predictions (Bottom):")
    print("   - Shows predicted responses for 5 different test images")
    print("   - Each line represents one test image")
    print("   - X-axis shows the 5 different neurons being predicted")
    print("   - Y-axis shows the predicted response strength for each neuron")
    print("   - This helps visualize how different images activate different neurons")