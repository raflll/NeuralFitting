# Neural Response Prediction System

## Project Overview
This project implements a deep learning system to predict neural responses to visual stimuli using a ResNet-based encoder. The system takes images as input and predicts how different neurons in the visual cortex will respond to these images.

## Team Members
Justin Bonner, Ethan Tieu, Gabriel Huang, Devon Mason

## Course Information
- Course: CSC550
- Institution: University of Miami

## Project Structure
```
.
├── images/
│   ├── train/          # Training images
│   └── test/           # Test images
├── ResNet.py           # Main implementation file
├── requirements.txt    # Python dependencies
├── neural_responses_train.npy  # Training neural responses
└── README.md           # This file
```

## Installation
1. Clone the repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Model Architecture
The project implements a ResNet-based encoder:
- Uses pre-trained ResNet50 as backbone
- Implements custom encoder architecture
- Includes cross-validation for model evaluation
- Uses adaptive pooling for feature extraction

## Results
The system generates:
1. Cross-validation performance visualization
2. Neural response predictions for test images
3. R² scores for model evaluation
4. Cross-validation results across multiple folds

## Visualization
The system produces two main visualizations:
1. Cross-Validation Performance Plot
   - Shows R² scores for each fold
   - Displays mean performance across all folds
   - Helps evaluate model consistency

2. Neural Response Predictions
   - Shows predicted responses for test images
   - Visualizes how different neurons respond to the same image
   - Helps understand the model's predictions

## Future Improvements
- Implement additional model architectures
- Add more sophisticated feature extraction methods
- Improve visualization capabilities
- Add support for different image formats and sizes

## License
[Add your chosen license here]

## Acknowledgments
- Course instructor and teaching assistants
- Original authors of ResNet architecture
- Contributors to PyTorch and scikit-learn 
