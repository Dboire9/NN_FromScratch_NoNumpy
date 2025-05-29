from my_tools import my_random as mr
from my_tools.VectorMatrixClass import Matrix, Vector
from my_tools.misc import my_sigmoid
from math import log
from matplotlib import pyplot as plt


def main():
    
    
    n = [2,3,3,1]  # Number of neurons in each layer
    
    print("layer 0 / input layer size", n[0])
    print("layer 1 size", n[1])
    print("layer 2 size", n[2])
    print("layer 3 size", n[3])

    W1 = Matrix(mr.randn(n[1], n[0]))
    W2 = Matrix(mr.randn(n[2], n[1]))
    W3 = Matrix(mr.randn(n[3], n[2]))
    
    b1 = Vector(mr.randn_vec(n[1]))
    b2 = Vector(mr.randn_vec(n[2]))
    b3 = Vector(mr.randn_vec(n[3]))
    
    weight_array = [W1, W2, W3]
    bias_array = [b1, b2, b3]

    
    print("W1", W1.shape(), "\nW2", W2.shape(), "\nW3", W3.shape())
    print("b1", len(b1), "\nb2", len(b2), "\nb3", len(b3))

    # X is our training data
    X = Matrix([[150, 70], 
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]])

    print("X shape", X.shape())
    
    A0 = X.transpose()  # Transpose X to match the expected input shape for the first layer
    print("X shape after transpose", A0.shape())
    
    y = Vector([0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    1,
    0])
    
    m = len(y)
    print(m, "samples in training set")
    Y = y.reshape(n[3], m)  # Reshape y to match the output layer size
    print("Y shape", Y.shape())
    
    costs = train(weight_array, bias_array, A0, Y, n, m)
    plt.plot(costs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Cost over Epochs')
    plt.show()




def train(weight_array, bias_array, A0, Y, n, m):
    epochs = 2000 # Number of epochs for training
    alpha = 0.1  # Learning rate
    costs = []
    for e in range(epochs):
    
        y_hat, AL = layer_calculations(m, weight_array, bias_array, A0, n)
        error = cost(y_hat, Y)
        costs.append(error)
        dC_dW, dC_db = backpropagation_loop(y_hat, Y, AL, weight_array, n, m)
        weight_array, bias_array = update_weights_and_biases(weight_array, bias_array, dC_dW, dC_db, m, alpha)
        
        if e % 20 == 0:
            print(f"epoch {e}: cost = {error:4f}")
    
    return costs

def layer_calculations(m, weight_matrices, bias_vector, A0, n):
        AL = [A0]
        A_prev = A0
        num_layers = len(weight_matrices)
        for l in range(num_layers):
            Z = weight_matrices[l].mul_mat(A_prev)
            Z = Z.add_vector_to_matrix(bias_vector[l])
            assert Z.shape() == (n[l + 1], m), f"Z{l + 1} shape mismatch"
            A = my_sigmoid(Z)
            assert A.shape() == (n[l + 1], m), f"A{l + 1} shape mismatch"
            AL.append(A)
            A_prev = A
        y_hat = AL[-1]
        return y_hat, AL

def cost(y_hat, y):
    """Compute the cost function with numerical stability (epsilon)."""
    from math import log

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


def sigmoid_derivative(A):
    # A is the activation (already sigmoid-ed)
    return A * (1 - A)


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

if __name__ == "__main__":
    main()