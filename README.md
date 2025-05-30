# Neural Network From Scratch 🧠

This project implements a simple feedforward neural network **from scratch** in Python, **without using high-level machine learning libraries like TensorFlow, PyTorch, or NumPy**. It is designed for educational purposes and demonstrates the core concepts of neural networks, including forward propagation, backpropagation, and training on real-world data.

## Features ⚙️

- Custom `Matrix` and `Vector` classes for all linear algebra operations (see `my_tools/VectorMatrixClass.py`)
- Manual implementation of:
  - Forward propagation
  - Backpropagation
  - Weight and bias updates
  - Cost calculation (binary cross-entropy)
  - Early stopping (optional)
- Data loading from CSV (Breast Cancer Wisconsin dataset)
- Configurable network architecture (number of layers, neurons per layer)
- Training and validation split
- Visualization of training cost over epochs
- Validation accuracy reporting

## File Structure 📂

```
Neural_Network_FromScratch/
├── archive/
│   └── data.csv                # Breast Cancer dataset (CSV)
├── my_tools/
│   ├── VectorMatrixClass.py    # Custom Matrix and Vector classes
│   └── my_random.py            # Random number utilities
├── neuron.py                   # Main neural network implementation
```

## How to Run ▶️

1. **Install requirements**  
   Only standard Python libraries are used (no external dependencies).

2. **Prepare the data**  
   Ensure `archive/data.csv` is present (already included).

3. **Run the main script**  
   ```bash
   python3 neuron.py
   ```

4. **What you will see**  
   - Training cost plotted over epochs
   - Final validation accuracy printed

## Main Components 🧩

- **neuron.py**  
  Main script: loads data, initializes the network, trains, validates, and plots results.

- **my_tools/VectorMatrixClass.py**  
  Custom classes for matrix and vector operations (addition, multiplication, transpose, etc.).

- **my_tools/my_random.py**  
  Utilities for generating random numbers for weight initialization.

- **archive/data.csv**  
  Breast Cancer Wisconsin dataset (features and labels).

## Customization 🛠️

- **Network architecture:**  
  Change `layer_nb` and the starting number of neurons in `init_neuron_layers()`.

- **Learning rate and epochs:**  
  Adjust `alpha` and `epochs` in `main()`.

- **Early stopping:**  
  The `train()` function supports early stopping via `patience` and `min_delta` parameters.

## Example Output 📋

```
[30, 64, 32, 16, 1]
455 samples in training set
epoch 0: cost = 0.693147
...
Final validation accuracy: 0.95
```

## Notes 📝

- **No external ML libraries** are used; all math is implemented manually.
- **For best results**, consider normalizing your input features.
- This project is for learning and demonstration; for production use, prefer established libraries.
- With fast parameters, the accuracy is ~65%
- I’ve been able to increase the accuracy to 85–90% with other parameters, but the whole process becomes very slow.
- Feel free to tweak the parameters as you like to try having a better accuracy !

---

**Author:**  
Dorian Boiré

2025
