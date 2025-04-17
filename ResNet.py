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
from torchvision.models import resnet18, ResNet18_Weights
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
BATCH_SIZE = 2
NUM_EPOCHS = 150
LEARNING_RATE = 5e-5
NUM_FOLDS = 3
IMAGE_SIZE = 260
WEIGHT_DECAY = 2e-5
EARLY_STOP_PATIENCE = 5

# Load neural responses
train_responses = np.load(TRAIN_RESPONSES_PATH)
response_means = train_responses.mean(axis=0)
response_stds = train_responses.std(axis=0)
train_responses = (train_responses - response_means) / response_stds

# Create dataset
class NeuralResponseDataset(Dataset):
    def __init__(self, image_dir, responses=None, transform=None, is_test=False, neuron_idx=None):
        self.image_dir = image_dir
        self.responses = responses
        self.transform = transform
        self.is_test = is_test
        self.neuron_idx = neuron_idx
        
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
            if self.neuron_idx is not None:
                # Return response for a specific neuron
                return image, self.responses[idx, self.neuron_idx]
            else:
                # Return responses for all neurons
                return image, self.responses[idx]

# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ResNet-based single neuron predictor
class SingleNeuronResNetEncoder(nn.Module):
    def __init__(self):
        super(SingleNeuronResNetEncoder, self).__init__()
        
        # Load pre-trained ResNet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Freeze the backbone parameters
        for name, param in self.features.named_parameters():
            param.requires_grad = False
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
                
        # Global pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head for a single neuron
        self.regression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Get features from ResNet
        features = self.features(x)
        
        # Apply global average pooling
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        
        # Predict response
        response = self.regression_head(pooled)
        
        return response

# Train and validate models for individual neurons
def train_and_validate_single_neuron_models():
    # Store results
    all_models = []
    all_r2_scores = []
    best_fold_per_neuron = []
    
    for neuron_idx in range(NUM_NEURONS):
        print(f"\n{'='*50}")
        print(f"Training model for Neuron {neuron_idx+1}/{NUM_NEURONS}")
        print(f"{'='*50}")
        
        # Prepare dataset for this neuron
        neuron_dataset = NeuralResponseDataset(
            image_dir=TRAIN_IMAGES_DIR,
            responses=train_responses,
            transform=transform,
            neuron_idx=neuron_idx
        )
        
        # Initialize k-fold cross-validation
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        
        # Lists to store results for this neuron
        fold_r2_scores = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(neuron_dataset)):
            print(f"Neuron {neuron_idx+1}, Fold {fold+1}/{NUM_FOLDS}")
            
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            no_improve_counter = 0
            
            train_loader = DataLoader(
                neuron_dataset, 
                batch_size=BATCH_SIZE, 
                sampler=train_subsampler
            )
            
            val_loader = DataLoader(
                neuron_dataset,
                batch_size=BATCH_SIZE,
                sampler=val_subsampler
            )
            
            # Initialize model, loss function, and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SingleNeuronResNetEncoder().to(device)
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
                    responses = responses.to(device).float().unsqueeze(1)  # Add dimension for single neuron
                    
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
                            responses = responses.to(device).float().unsqueeze(1)  # Add dimension for single neuron
                            
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
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1

                    if (no_improve_counter >= EARLY_STOP_PATIENCE):
                        print(f"Early stopping at epoch {epoch+1}, best R² was {best_r2:.4f}")
                        break
                    
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
                    responses = responses.float().unsqueeze(1)  # Add dimension for single neuron
                    
                    outputs = model(images)
                    
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(responses.numpy())
            
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            r2 = r2_score(all_targets, all_preds)
            fold_r2_scores.append(r2)
            fold_models.append(model)
            
            print(f"Neuron {neuron_idx+1}, Fold {fold+1} R²: {r2:.4f}")
            
            # Save the model
            torch.save(model.state_dict(), f'single_neuron_{neuron_idx+1}_fold_{fold+1}.pth')
        
        # Find the best fold for this neuron
        best_fold_idx = np.argmax(fold_r2_scores)
        best_fold = best_fold_idx + 1
        best_r2 = fold_r2_scores[best_fold_idx]
        
        print(f"Best model for Neuron {neuron_idx+1}: Fold {best_fold} with R² = {best_r2:.4f}")
        
        all_r2_scores.append(fold_r2_scores)
        best_fold_per_neuron.append(best_fold)
    
    return all_r2_scores, best_fold_per_neuron

# Generate predictions for test set
def generate_test_predictions(best_fold_per_neuron):
    # Load test dataset (without neuron index as we'll predict for all)
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
    
    # Store all predictions
    all_preds = np.zeros((len(test_dataset), NUM_NEURONS))
    all_filenames = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For each neuron, use the best model to generate predictions
    for neuron_idx in range(NUM_NEURONS):
        best_fold = best_fold_per_neuron[neuron_idx]
        
        # Load the best model for this neuron
        model = SingleNeuronResNetEncoder().to(device)
        model.load_state_dict(torch.load(f'single_neuron_{neuron_idx+1}_fold_{best_fold}.pth'))
        model.eval()
        
        # Generate predictions
        neuron_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                
                neuron_preds.append(outputs.cpu().numpy())
                
                # Only collect filenames once (from the first neuron)
                if neuron_idx == 0:
                    all_filenames.extend(filenames)
        
        # Combine predictions for this neuron
        neuron_preds = np.vstack(neuron_preds)
        all_preds[:, neuron_idx] = neuron_preds.flatten()
    
    # Save predictions
    np.save('neural_responses_test_pred_per_neuron.npy', all_preds)
    
    # Save filenames for reference
    with open('test_image_order.txt', 'w') as f:
        for filename in all_filenames:
            f.write(f"{filename}\n")
    
    print(f"Generated predictions for {len(all_filenames)} test images using {NUM_NEURONS} individual neuron models")
    return all_preds

# Visualize results
def visualize_results(all_r2_scores, test_preds):
    plt.figure(figsize=(15, 12))
    
    # 1. Cross-Validation Performance Plot per Neuron
    plt.subplot(2, 1, 1)
    
    neuron_mean_r2 = [np.mean(scores) for scores in all_r2_scores]
    x = np.arange(NUM_NEURONS)
    width = 0.2
    
    for fold in range(NUM_FOLDS):
        fold_scores = [scores[fold] if fold < len(scores) else 0 for scores in all_r2_scores]
        plt.bar(x + (fold - 1) * width, fold_scores, width, 
                label=f'Fold {fold+1}')
    
    plt.plot(x, neuron_mean_r2, 'ro-', linewidth=2, label='Mean R²')
    
    plt.xlabel('Neuron Index')
    plt.ylabel('R² Score')
    plt.title('Individual Neuron Models: Cross-Validation Performance')
    plt.xticks(x, [f'Neuron {i+1}' for i in range(NUM_NEURONS)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of mean points
    for i, score in enumerate(neuron_mean_r2):
        plt.text(i, score, f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Neural Response Predictions
    plt.subplot(2, 1, 2)
    
    # Show different test images
    for i in range(5):  # Show first 5 test samples
        plt.plot(range(NUM_NEURONS), test_preds[i], 
                marker='o', label=f'Test Image {i+1}')
    
    plt.xlabel('Neuron Index')
    plt.ylabel('Predicted Neural Response')
    plt.title('Individual Neuron Models: Predicted Neural Responses for Test Images')
    plt.xticks(range(NUM_NEURONS), [f'N{i+1}' for i in range(NUM_NEURONS)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
                "Top: Cross-validation performance for individual neuron models. Each neuron has its own specialized model.\n"
                "Bottom: Predicted neural responses for 5 test images across all neurons.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('individual_neuron_models_performance.png')
    plt.show()
    
    # Print detailed explanation
    print("\nVisualization Explanation:")
    print("1. Cross-Validation Performance Plot (Top):")
    print("   - Shows the R² score for each fold for each neuron")
    print("   - The red line shows the mean R² score across folds for each neuron")
    print("   - Allows comparison of prediction accuracy across different neurons")
    print("\n2. Neural Response Predictions (Bottom):")
    print("   - Shows predicted responses for 5 different test images")
    print("   - Each line represents one test image")
    print("   - X-axis shows the 5 different neurons (each with its own model)")
    print("   - Y-axis shows the predicted response strength")
    print("   - This helps visualize how different images activate different neurons")

if __name__ == "__main__":
    # Train and validate individual neuron models
    all_r2_scores, best_fold_per_neuron = train_and_validate_single_neuron_models()
    
    # Print summary of best models for each neuron
    print("\nSummary of Best Models:")
    for neuron_idx in range(NUM_NEURONS):
        best_fold = best_fold_per_neuron[neuron_idx]
        best_r2 = all_r2_scores[neuron_idx][best_fold-1]
        print(f"Neuron {neuron_idx+1}: Best fold = {best_fold}, R² = {best_r2:.4f}")
    
    # Generate test predictions using the best model for each neuron
    test_preds = generate_test_predictions(best_fold_per_neuron)
    
    # Compare mean R² of individual models vs. joint model approach
    mean_r2_per_neuron = [np.mean(scores) for scores in all_r2_scores]
    overall_mean_r2 = np.mean(mean_r2_per_neuron)
    
    print(f"\nOverall mean R² across all neurons and folds: {overall_mean_r2:.4f}")
    print(f"Mean R² per neuron: {mean_r2_per_neuron}")
    
    # Visualize results
    visualize_results(all_r2_scores, test_preds)