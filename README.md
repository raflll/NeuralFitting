# Neural Response Prediction System

This project implements a system that predicts neural responses to visual stimuli using deep learning and machine learning techniques. The system uses ResNet50 features combined with Ridge regression to predict how different neurons will respond to various images.

## Project Structure

```
.
├── images/
│   ├── train/          # Training images
│   └── test/           # Test images
├── neural_responses_train.npy    # Training neural responses
├── neural_responses_test_pred.npy # Predicted test responses
├── test_image_order.txt          # Order of test images
├── test_predictions_samples.png  # Visualization of predictions
├── ResNet.py           # Main implementation
└── requirements.txt    # Python dependencies
```

## Dependencies

All required packages are listed in `requirements.txt`. The main dependencies include:
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- Pillow

## Installation

1. Create a virtual environment (recommended):
   ```bash
   # For Unix/macOS
   python3 -m venv venv
   source venv/bin/activate

   # For Windows
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Implementation Details

The system uses a two-stage approach:

1. **Feature Extraction**:
   - Uses pre-trained ResNet50 model
   - Extracts features from the penultimate layer
   - Applies global average pooling to reduce dimensionality

2. **Response Prediction**:
   - Trains separate Ridge regression models for each neuron
   - Uses Pearson correlation coefficient to evaluate model performance
   - Optimizes regularization parameter (alpha) for each neuron

## Usage

1. Place your training images in `images/train/`
2. Place your test images in `images/test/`
3. Ensure `neural_responses_train.npy` contains the training responses
4. Run the script:
   ```bash
   python ResNet.py
   ```

The script will:
- Extract features from all images
- Train Ridge regression models
- Generate predictions for test images
- Save predictions to `neural_responses_test_pred.npy`
- Create a visualization of predictions in `test_predictions_samples.png`

## Results

The system generates:
1. Predicted neural responses for test images
2. A visualization showing how different neurons respond to each test image
3. Average Pearson correlation score across all neurons

The visualization (`test_predictions_samples.png`) shows:
- 5 different test images
- For each image, the responses of 5 different neurons
- Color-coded bars representing different neurons
- Response strengths ranging from -1 (inhibitory) to 1 (excitatory)

## Notes

- The system uses Pearson correlation coefficient instead of R² for evaluation
- Each neuron has its own optimized Ridge regression model
- The visualization helps understand how different images activate different neurons
