from my_tools import my_random as mr
import random
from my_tools.VectorMatrixClass import Matrix, Vector
from math import log, exp
from matplotlib import pyplot as plt
import csv
import os
import kagglehub


def main():
    

    # path = "archive/data.csv"
    dir_path = kagglehub.dataset_download("uciml/iris")
    csv_path = os.path.join(dir_path, "Iris.csv")  # This is the actual CSV file
    X, y, X_val, y_val = load_data(csv_path, validation_ratio=0.2)

    layer_nb = 4 # Number of hidden layers in the neural network
    alpha = 0.05  # Learning rate
    epochs = 2000 # Number of epochs for training
    
    output_activation = "softmax"  # "sigmoid" or "softmax"

    num_classes = 3  # Number of classes in the Iris dataset (setosa, versicolor, virginica)
    n = init_neuron_layers(layer_nb, X, num_classes)  # Initialize the number of neurons in each layer
    
    weight_array, bias_array = neuron_init(n, X.num_rows)  # Initialize weights and biases for the network
    
    A0 = X.transpose()  # Transpose X to match the expected input shape for the first layer
    
    m = len(y)
    print(m, "samples in training set")
    Y = y
    
    
    costs = train(weight_array, bias_array, A0, Y, n, m, alpha, epochs, output_activation=output_activation)
    plt.plot(costs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost over Epochs')
    plt.show()
    
    accuracy = validate(weight_array, bias_array, X_val, y_val, n)
    print(f"Final validation accuracy: {accuracy:.4f}")

def load_data(csv_path, validation_ratio=0.2):
    # Read CSV
    if not os.path.exists(csv_path):
        print("File does not exist.")
        exit()
        
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = []
        for row in reader:
            if '' in row:
                continue
            features = list(map(float, row[1:5]))
            label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
            label = label_map[row[5]]
            data.append((features, label))
        
    random.shuffle(data)
        
    # Split features and labels
    X_data = [features for features, label in data]
    y_data = [label for features, label in data]

    # Split into train/validation
    split_idx = int(len(data) * (1 - validation_ratio))
    X_train = X_data[:split_idx]
    y_train = y_data[:split_idx]
    X_val = X_data[split_idx:]
    y_val = y_data[split_idx:]

    # Convert to Matrix/Vector
    X = Matrix(X_train)
    y = Vector(y_train)
    X_val = Matrix(X_val)
    y_val = Vector(y_val)

    return X, y, X_val, y_val

def one_hot_encode(Y, num_classes):
    # Y: Vector or list of class indices, num_classes: int
    m = len(Y)
    one_hot = []
    for c in range(num_classes):
        one_hot.append([1 if int(Y[i]) == c else 0 for i in range(m)])
    return Matrix(one_hot)

def init_neuron_layers(layer_nb, X, num_classes):
    n = [X.num_cols]  # Input layer
    neurons = 64  # or any power of 2 you want to start with
    for i in range(layer_nb - 1):
        n.append(neurons)
        neurons = max(2, neurons // 2)  # Halve, but not less than 2
    n.append(num_classes)  # Output layer
    print(n)
    return n

def neuron_init(n, m):
    """
    Initialize weights and biases for a neural network with n layers.
    n: list of integers, where n[i] is the number of neurons in layer i
    m: number of samples
    Returns: weight_array, bias_array
    """
    weight_array = []
    bias_array = []
    
    for i in range(len(n) - 1):
        W = Matrix(mr.randn(n[i + 1], n[i]))
        b = Vector(mr.randn_vec(n[i + 1]))
        weight_array.append(W)
        bias_array.append(b)
    
    return weight_array, bias_array

def train(weight_array, bias_array, A0, Y, n, m, alpha, epochs, patience=200, min_delta=1e-6, output_activation='sigmoid'):
    costs = []
    best_cost = float('inf')
    epochs_no_improve = 0

    for e in range(epochs):
        y_hat, AL = layer_calculations(m, weight_array, bias_array, A0, n)
        if output_activation == 'softmax':
            error = cost_softmax(y_hat, Y)
        else:
            error = cost_sigmoid(y_hat, Y)
        costs.append(error)
        
        if output_activation == "softmax":
            Y_one_hot = one_hot_encode(Y, n[-1])
            dC_dW, dC_db = backpropagation_loop(y_hat, Y_one_hot, AL, weight_array, n, m)
        else:
            dC_dW, dC_db = backpropagation_loop(y_hat, Y, AL, weight_array, n, m)

        
        weight_array, bias_array = update_weights_and_biases(weight_array, bias_array, dC_dW, dC_db, m, alpha)

        if e % 20 == 0:
            print(f"epoch {e}: cost = {error:4f}")

        # Early stopping logic
        if best_cost - error > min_delta:
            best_cost = error
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {e} (cost did not improve for {patience} epochs)")
            break

    return costs

def layer_calculations(m, weight_matrices, bias_vector, A0, n, output_activation='sigmoid'):
        AL = [A0]
        A_prev = A0
        num_layers = len(weight_matrices)
        for l in range(num_layers):
            Z = weight_matrices[l].mul_mat(A_prev)
            Z = Z.add_vector_to_matrix(bias_vector[l])
            assert Z.shape() == (n[l + 1], m), f"Z{l + 1} shape mismatch"
            if l == len(weight_matrices) - 1:
                # Output layer
                if output_activation == "softmax":
                    A = my_softmax(Z)
                else:
                    A = my_sigmoid(Z)
            else:
                A = my_sigmoid(Z)
            assert A.shape() == (n[l + 1], m), f"A{l + 1} shape mismatch"
            AL.append(A)
            A_prev = A
        y_hat = AL[-1]
        return y_hat, AL

def cost_sigmoid(y_hat, y):
    """Compute the cost function with numerical stability (epsilon)."""

    eps = 1e-8
    y_hat_vals = [v for row in y_hat.rows for v in row]
    y_vals = [v for row in y.rows for v in row]

    losses = [
        - (y_true * log(y_pred + eps) + (1 - y_true) * log(1 - y_pred + eps))
        for y_true, y_pred in zip(y_vals, y_hat_vals)
    ]
    m = len(losses)
    summed_loss = sum(losses) / m
    return summed_loss

def cost_softmax(y_hat, y):
    """Compute the cost function for softmax output with numerical stability."""

    if isinstance(y, Matrix):
        # If shape is (1, m) or (m, 1), flatten to [Y[0][i] for i in range(m)] or [Y[i][0] for i in range(m)]
        if len(y.rows) == 1:
            y_flat = y.rows[0]
        elif len(y.rows[0]) == 1:
            y_flat = [row[0] for row in y.rows]
        else:
            raise ValueError("y Matrix shape not supported for cost_softmax")
    elif isinstance(y, list) and isinstance(y[0], list):
        y_flat = [item[0] for item in y]
    else:
        y_flat = list(y)

    m = len(y_flat)
    epsilon = 1e-12
    losses = []
    for i in range(m):
        y_pred = [row[i] for row in y_hat.rows]
        y_true = int(y_flat[i])
        losses.append(-log(y_pred[y_true] + epsilon))
    return sum(losses) / m

def my_sigmoid(x):
    """Sigmoid activation function with overflow protection."""
    from my_tools.VectorMatrixClass import Vector, Matrix

    def safe_sigmoid(v):
        v = max(min(v, 500), -500)  # Clip to avoid overflow
        return 1 / (1 + exp(-v))

    if isinstance(x, Vector):
        return Vector([safe_sigmoid(v) for v in x])
    elif isinstance(x, Matrix):
        return Matrix([[safe_sigmoid(v) for v in row] for row in x.rows])
    elif hasattr(x, "__iter__") and not isinstance(x, str):
        return type(x)([safe_sigmoid(v) for v in x])
    else:
        return safe_sigmoid(x)

def my_softmax(Z):
    """Softmax activation function with numerical stability."""

    if isinstance(Z, Matrix):
    # Compute max for each column for numerical stability
        num_cols = len(Z.rows[0])
        max_per_col = []
        for j in range(num_cols):
            max_per_col.append(max(row[j] for row in Z.rows))
    
        # Subtract max and exponentiate
        exp_rows = []
        for i, row in enumerate(Z.rows):
            exp_row = []
            for j, val in enumerate(row):
                exp_row.append(exp(val - max_per_col[j]))
            exp_rows.append(exp_row)
        exp_Z = Matrix(exp_rows)
    
        # Sum over columns
        sum_exp_per_col = []
        for j in range(num_cols):
            sum_exp_per_col.append(sum(exp_Z.rows[i][j] for i in range(len(exp_Z.rows))))
    
        # Divide each column by its sum
        softmax_rows = []
        for i, row in enumerate(exp_Z.rows):
            softmax_row = []
            for j, val in enumerate(row):
                softmax_row.append(val / sum_exp_per_col[j])
            softmax_rows.append(softmax_row)

    return Matrix(softmax_rows)

def backpropagation_loop(y_hat, Y, A_list, W_list, n, m):
    dC_dA = (1/m) * (y_hat - Y)
    dC_dW_list = []
    dC_db_list = []
    
    for l in reversed(range(1, len(n))):  # from output to first hidden
        dC_dW, dC_db, dC_dA_prev = backpropagation(
            propagator_dC_dA = dC_dA,
            A_prev = A_list[l-1],
            A = A_list[l],
            W = W_list[l-1],
            n_l = n[l],
            n_prev = n[l-1],
            m = m
        )
        dC_dW_list.insert(0, dC_dW)
        dC_db_list.insert(0, dC_db)
        dC_dA = dC_dA_prev  # propagate for next layer
    return dC_dW_list, dC_db_list


def backpropagation(propagator_dC_dA, A_prev, A, W, n_l, n_prev, m):
    """
    Modular backprop for a single layer.
    propagator_dC_dA: upstream gradient (Matrix, shape: (n_l, m))
    A_prev: activations from previous layer (Matrix, shape: (n_prev, m))
    A: activations from this layer (Matrix, shape: (n_l, m))
    W: weights for this layer (Matrix, shape: (n_l, n_prev))
    n_l: number of neurons in this layer
    n_prev: number of neurons in previous layer
    m: number of samples
    Returns: dC_dW, dC_db, dC_dA_prev
    """
    # Step 1: dC/dZ = dC/dA * sigmoid'(A)
    dA_dZ = A * (1 - A)
    dC_dZ = propagator_dC_dA * dA_dZ
    assert dC_dZ.shape() == (n_l, m)

    # Step 2: dC/dW = dC/dZ @ A_prev.T / m
    dC_dW = (dC_dZ @ A_prev.transpose())
    assert dC_dW.shape() == (n_l, n_prev)

    # Step 3: dC/db = sum over columns of dC/dZ, shape (n_l, 1)
    dC_db = Matrix([[sum(dC_dZ.rows[i])] for i in range(n_l)]) * (1 / m)
    assert dC_db.shape() == (n_l, 1)

    # Step 4: propagate dC/dA_prev = W.T @ dC/dZ
    dC_dA_prev = W.transpose() @ dC_dZ
    assert dC_dA_prev.shape() == (n_prev, m)

    return dC_dW, dC_db, dC_dA_prev

def update_weights_and_biases(weight_matrices, bias_vectors, dC_dW, dC_db, m, learning_rate):

    for i in range(len(weight_matrices)):
        weight_matrices[i] = weight_matrices[i] - (learning_rate * dC_dW[i])
        if isinstance(dC_db[i], Matrix) and dC_db[i].num_cols == 1:
            db_vec = Vector([row[0] for row in dC_db[i].rows])
        else:
            db_vec = dC_db[i]
        
        db_vec.scl(learning_rate)
        bias_vectors[i] = bias_vectors[i] - db_vec
        
    return weight_matrices, bias_vectors

def validate(weight_array, bias_array, X_val, y_val, n, output_activation="softmax"):
    m_val = X_val.num_rows
    A0_val = X_val.transpose()
    y_hat_val, _ = layer_calculations(m_val, weight_array, bias_array, A0_val, n, output_activation=output_activation)

    # For softmax: pick the class with the highest probability
    if output_activation == "softmax":
        y_pred = [max(enumerate([row[i] for row in y_hat_val.rows]), key=lambda x: x[1])[0] for i in range(m_val)]
    else:  # sigmoid
        y_pred = [1 if y_hat_val.rows[0][i] >= 0.5 else 0 for i in range(m_val)]

    # Flatten y_val if needed
    if hasattr(y_val, "rows"):
        if len(y_val.rows[0]) == 1:
            y_true = [row[0] for row in y_val.rows]
        else:
            y_true = y_val.rows[0]
    else:
        y_true = list(y_val)

    # Print predictions and true labels for debugging
    print("Predicted:", y_pred[:20])
    print("True:     ", y_true[:20])

    correct = sum(1 for yp, yt in zip(y_pred, y_true) if yp == yt)
    accuracy = correct / len(y_true)
    print(f"Validation accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    main()