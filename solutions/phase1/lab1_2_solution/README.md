# Lab 1.2 Solution: PyTorch Training Script

Complete, working solution for Lab 1.2 - Building a PyTorch training script for MNIST image classification.

## Files Included

1. **train_mnist.py** - Complete MNIST training script with:
   - SimpleCNN model architecture
   - Data loading with DataLoader
   - Full training loop with validation
   - Model checkpointing
   - Progress tracking

2. **test_mnist.py** - Model testing and evaluation script with:
   - Checkpoint loading
   - Test set evaluation
   - Per-class accuracy analysis
   - Sample predictions visualization

## Prerequisites

- Python 3.8+
- PyTorch installed (`pip install torch torchvision`)
- Virtual environment activated

## Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment
source ~/venv/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision

# Or for GPU support:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Navigate to Solution Directory

```bash
cd /home/user/mlops-learning-plan/solutions/phase1/lab1_2_solution
```

### 3. Train the Model

```bash
python train_mnist.py
```

**Expected runtime:** 5-10 minutes on CPU, 1-2 minutes on GPU

**Expected accuracy:** >97% on test set after 5 epochs

### 4. Test the Trained Model

```bash
# Test the best model
python test_mnist.py --checkpoint ./models/staging/mnist/best_model.pt

# Or test a specific epoch
python test_mnist.py --checkpoint ./models/staging/mnist/checkpoint_epoch_5.pt

# Show more sample predictions
python test_mnist.py --show-samples 20
```

## Detailed Usage

### Training Script (train_mnist.py)

The training script includes:

**Model Architecture:**
```
SimpleCNN:
  - Conv1: 1 -> 32 channels (3x3 kernel)
  - MaxPool + ReLU
  - Conv2: 32 -> 64 channels (3x3 kernel)
  - MaxPool + ReLU + Dropout(0.25)
  - Flatten: 64*7*7 = 3136
  - FC1: 3136 -> 128
  - ReLU + Dropout(0.5)
  - FC2: 128 -> 10 (output)
```

**Hyperparameters:**
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 5
- Loss function: CrossEntropyLoss

**Output Structure:**
```
./models/staging/mnist/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── checkpoint_epoch_3.pt
├── checkpoint_epoch_4.pt
├── checkpoint_epoch_5.pt
└── best_model.pt          # Best validation accuracy
```

**Expected Output:**

```
Using device: cpu
Loading MNIST dataset...
Downloading http://yann.lecun.com/exdb/mnist/...
Train samples: 60000
Test samples: 10000

Model architecture:
SimpleCNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)

Total trainable parameters: 421,226

Starting training for 5 epochs...

============================================================
Epoch 1/5
============================================================
Epoch 1 [0/60000 (0%)]  Loss: 2.304586
Epoch 1 [6400/60000 (11%)]  Loss: 0.234567
...
Training: Loss: 0.2345, Accuracy: 93.12%

Validation: Average loss: 0.0987, Accuracy: 9654/10000 (96.54%)

Checkpoint saved: ./models/staging/mnist/checkpoint_epoch_1.pt
New best accuracy: 96.54%

============================================================
Epoch 2/5
============================================================
...

============================================================
Training completed!
Best validation accuracy: 98.23%
============================================================
```

### Testing Script (test_mnist.py)

**Basic Usage:**

```bash
# Test best model
python test_mnist.py

# Test specific checkpoint
python test_mnist.py --checkpoint ./models/staging/mnist/checkpoint_epoch_3.pt

# Custom batch size
python test_mnist.py --batch-size 128

# Show more sample predictions
python test_mnist.py --show-samples 25
```

**Command-line Arguments:**

- `--checkpoint`: Path to model checkpoint (default: `./models/staging/mnist/best_model.pt`)
- `--batch-size`: Batch size for testing (default: 64)
- `--data-dir`: Directory containing MNIST data (default: `./data`)
- `--show-samples`: Number of sample predictions to display (default: 10)

**Expected Output:**

```
Using device: cpu
Loading MNIST test set...
Test samples: 10000
Loaded model from: ./models/staging/mnist/best_model.pt
Checkpoint info:
  - Epoch: 4
  - Validation Loss: 0.0567
  - Validation Accuracy: 98.23%

Testing model...

============================================================
TEST RESULTS
============================================================

Overall Accuracy: 98.23%
Correct predictions: 9823/10000

Per-Class Accuracy:
----------------------------------------
  Digit 0: 99.18%
  Digit 1: 99.03%
  Digit 2: 97.87%
  Digit 3: 98.22%
  Digit 4: 98.17%
  Digit 5: 97.76%
  Digit 6: 98.54%
  Digit 7: 97.76%
  Digit 8: 97.33%
  Digit 9: 97.43%

============================================================
✓ Model meets target accuracy (>97%)
============================================================

Sample Predictions (first 10):
----------------------------------------
  Sample 1: Predicted=7, Actual=7 ✓
  Sample 2: Predicted=2, Actual=2 ✓
  Sample 3: Predicted=1, Actual=1 ✓
  Sample 4: Predicted=0, Actual=0 ✓
  Sample 5: Predicted=4, Actual=4 ✓
  Sample 6: Predicted=1, Actual=1 ✓
  Sample 7: Predicted=4, Actual=4 ✓
  Sample 8: Predicted=9, Actual=9 ✓
  Sample 9: Predicted=5, Actual=6 ✗
  Sample 10: Predicted=9, Actual=9 ✓
```

## Understanding the Code

### Key Components

**1. Model Definition (SimpleCNN)**
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        # Define layers

    def forward(self, x):
        # Define forward pass
```

**2. Data Loading**
```python
def get_dataloaders(batch_size=64, data_dir='./data'):
    # Create transforms
    # Load MNIST dataset
    # Return DataLoaders
```

**3. Training Loop**
```python
def train_epoch(...):
    model.train()  # Set to training mode
    for data, target in train_loader:
        optimizer.zero_grad()  # Clear gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)
        loss.backward()        # Backward pass
        optimizer.step()       # Update weights
```

**4. Validation**
```python
def validate(...):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        # Test on validation set
```

**5. Checkpointing**
```python
def save_checkpoint(...):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
```

## Customization

### Change Hyperparameters

Edit the values in the `main()` function:

```python
# In train_mnist.py
def main():
    batch_size = 128        # Increase batch size
    learning_rate = 0.0001  # Decrease learning rate
    num_epochs = 10         # Train longer
```

### Modify Model Architecture

Edit the `SimpleCNN` class:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        # Add more layers
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        # Increase FC layer size
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
```

### Use Different Optimizer

```python
# Replace Adam with SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9
)
```

## Common Issues and Troubleshooting

### Issue 1: "No module named 'torch'"

**Solution:**
```bash
# Install PyTorch
pip install torch torchvision

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Issue 2: CUDA out of memory

**Solution:**
```python
# Reduce batch size
batch_size = 32  # or even 16

# Or force CPU
device = torch.device('cpu')
```

### Issue 3: Model accuracy is low (<90%)

**Possible causes:**
- Training didn't converge (try more epochs)
- Learning rate too high (decrease it)
- Model too simple (add more layers)

**Solution:**
```python
num_epochs = 10  # Train longer
learning_rate = 0.0001  # Lower learning rate
```

### Issue 4: "Checkpoint not found" error

**Solution:**
```bash
# Make sure you trained the model first
python train_mnist.py

# Check if checkpoint exists
ls -lh ./models/staging/mnist/
```

### Issue 5: Training is very slow

**Tips for faster training:**
```python
# Increase batch size (if memory allows)
batch_size = 128

# Use GPU if available (automatic in the script)
# Check GPU availability:
python -c "import torch; print(torch.cuda.is_available())"

# Reduce num_workers if CPU is bottleneck
num_workers = 0  # in get_dataloaders()
```

## Performance Benchmarks

**Expected Results:**

| Metric | Value |
|--------|-------|
| Training time (5 epochs, CPU) | 5-10 minutes |
| Training time (5 epochs, GPU) | 1-2 minutes |
| Final test accuracy | >97% |
| Best test accuracy | 98-99% |
| Model size | ~1.6 MB |
| Total parameters | 421,226 |

**Epoch-by-epoch progression:**

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1 | 93-95% | 96-97% | 0.10-0.15 |
| 2 | 96-97% | 97-98% | 0.08-0.10 |
| 3 | 97-98% | 98-99% | 0.06-0.08 |
| 4 | 98-99% | 98-99% | 0.05-0.07 |
| 5 | 98-99% | 98-99% | 0.04-0.06 |

## Key Concepts Demonstrated

### PyTorch Fundamentals
- ✅ `nn.Module` for model definition
- ✅ `forward()` method implementation
- ✅ Convolutional and fully connected layers
- ✅ Activation functions (ReLU)
- ✅ Regularization (Dropout)

### Training Best Practices
- ✅ Separate training and validation
- ✅ `model.train()` and `model.eval()` modes
- ✅ `optimizer.zero_grad()` before backprop
- ✅ Gradient computation with `.backward()`
- ✅ Parameter updates with `optimizer.step()`

### Data Handling
- ✅ `transforms.Compose()` for preprocessing
- ✅ Normalization with dataset statistics
- ✅ `DataLoader` for batching and shuffling
- ✅ Efficient data loading with `num_workers`

### Model Management
- ✅ Saving model state with `state_dict()`
- ✅ Loading checkpoints with `load_state_dict()`
- ✅ Tracking best model during training
- ✅ Saving optimizer state for resuming

### Device Management
- ✅ Device-agnostic code (CPU/GPU)
- ✅ `.to(device)` for model and data
- ✅ Automatic GPU detection

## Next Steps

After completing this lab:

1. **Experiment**: Try different architectures, learning rates, optimizers
2. **Extend**: Add learning rate scheduling, early stopping
3. **Visualize**: Add TensorBoard logging
4. **Compare**: Run multiple experiments with different configurations

These extensions are covered in **Lab 1.3: Config-Driven Training**.

## Success Criteria

You've successfully completed Lab 1.2 if:

- ✅ Training script runs without errors
- ✅ Model achieves >97% test accuracy
- ✅ Checkpoints are saved correctly
- ✅ You can load and test a saved model
- ✅ You understand the training loop structure
- ✅ You can modify hyperparameters

---

**Congratulations on building your first PyTorch training pipeline!** 🎉

Proceed to **Lab 1.3: Config-Driven Training** to learn how to manage experiments with configuration files.
