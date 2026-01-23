# ECG Heartbeat Classification using CNN-BiLSTM-Attention

Deep learning model for automated classification of ECG heartbeats into 5 arrhythmia categories achieving **98.25% test accuracy**.

---

## Dataset

This model uses the **MIT-BIH Arrhythmia Database** from PhysioNet, which was preprocessed and split into train/test sets by **Shayan Fazeli**. The dataset can be found here: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

### Class Imbalance

The dataset exhibits severe class imbalance, which is typical of real-world medical data:

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 72,471 | 82.77% |
| Supraventricular | 2,223 | 2.54% |
| Ventricular | 5,788 | 6.61% |
| Fusion | 641 | 0.73% |
| Unclassifiable | 6,431 | 7.35% |

**Solution:** I use **balanced class weights** (computed via sklearn's `compute_class_weight`) to assign higher penalties to minority classes during training, ensuring the model learns all arrhythmia types.

### Data Augmentation

I apply two augmentation techniques to the **training set**:

1. **Gaussian Noise** (σ=0.03, probability=0.5)
   - Simulates real-world electrical interference and sensor noise
   - Makes the model robust to signal quality variations

2. **Time Shifting** (±10 samples, probability=0.5)
   - Accounts for slight variations in heartbeat segmentation
   - Makes the model invariant to minor timing misalignments

---

##  Model Architecture

My model uses a hybrid CNN-BiLSTM with multi-head attention that combines three complementary approaches:

### Architecture Flow

```
Input: (batch, 1, 187)
    ↓
┌─────────────────────────────────┐
│   CNN Feature Extractor         │
│   • Conv1D(1→64, k=5)           │
│   • Conv1D(64→128, k=5)         │
│   • Conv1D(128→128, k=3)        │
│   • BatchNorm + Dropout(0.2)    │
└─────────────────────────────────┘
    ↓ (batch, 128, 23)
┌─────────────────────────────────┐
│   Bidirectional LSTM            │
│   • Hidden size: 64             │
│   • Output: 128 (64×2)          │
│   • BatchNorm + Dropout(0.3)    │
└─────────────────────────────────┘
    ↓ (batch, 23, 128)
┌─────────────────────────────────┐
│   Multi-Head Attention          │
│   • 4 attention heads           │
│   • Self-attention mechanism    │
│   • Dropout(0.1)                │
└─────────────────────────────────┘
    ↓ (batch, 23, 128)
┌─────────────────────────────────┐
│   Global Average Pooling        │
└─────────────────────────────────┘
    ↓ (batch, 128)
┌─────────────────────────────────┐
│   Fully Connected               │
│   • Dense(128→64)               │
│   • Dense(64→5)                 │
│   • BatchNorm + Dropout(0.4)    │
└─────────────────────────────────┘
    ↓
Output: (batch, 5)
```

### What Each Block Does

**1. Convolutional Layers**
- Extract **local morphological features** from ECG waveforms
- Detect key ECG components: QRS complex, P-wave, T-wave shapes
- BatchNorm and Dropout prevent overfitting

**2. Bidirectional LSTM**
- Captures **temporal dependencies** by processing the sequence in both forward and backward directions
- Understands how earlier parts of the heartbeat influence later parts (and vice versa)
- Output concatenates both directions: 64 (forward) + 64 (backward) = 128 features

**3. Multi-Head Attention**
- Learns to **focus on important timesteps** in the heartbeat
- 4 parallel attention heads each learn different patterns:
  - e.g., R-peak amplitude, QRS duration, P-wave presence, T-wave abnormalities

**4. Global Average Pooling**
- Aggregates information from all timesteps into a single representation
- Reduces overfitting compared to using a single timestep

**5. Fully Connected Classifier**
- Maps the 128-dimensional representation to 5 class probabilities

---

## Results

### Test Performance


**Overall Test Accuracy: **  **98.25%** 

### Per-Class Accuracy

| Class | Test Accuracy |
|-------|--------------|---------|
| Normal | 99% |
| Supraventricular | 87% |
| Ventricular | 95% |
| Fusion | 94% |
| Unclassifiable | 99% |

---

## Usage

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/heartbeat-arrhythmia-classifier.git
cd heartbeat-arrhythmia-classifier

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Download the MIT-BIH dataset from Kaggle: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
2. Extract `mitbih_train.csv` and `mitbih_test.csv`
3. Upload to Google Drive (if using Colab) or place in project directory

### Training

**Google Colab (Recommended):**
1. Upload `heart_arrhythmia_classifier.ipynb` to Google Colab
2. Mount your Google Drive containing the dataset
3. Update file paths in the notebook to point to your data
4. Run all cells

The notebook includes:
- Data loading and preprocessing
- Model architecture
- Training loop with early stopping
- Evaluation and visualization

---

### Inference on new data

**Note: This model was trained on and performs best when classifying individual heartbeats**

```python
import torch
import numpy as np

# Load trained model
model = CNN_BiLSTM_MultiHeadAttention()
model.load_state_dict(torch.load('best_ecg_model.pth'))
model.eval()

# Preprocess your data (must be 187 samples, normalized)
# heartbeat shape: (187,)
heartbeat = scaler.transform(your_heartbeat.reshape(1, -1))
heartbeat = torch.FloatTensor(heartbeat).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 187)

# Predict
with torch.no_grad():
    output = model(heartbeat)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

# Map to class names
class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']
print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class].item():.2%}")
```


## References

**Dataset:**
- Preprocessed by Shayan Fazeli: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

---

## Author

Created to explore deep learning in signal analysis
---

## License

MIT License

---

