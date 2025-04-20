"""Simple neural network implementation from scratch using NumPy."""

import numpy as np

# Try to import numba for JIT compilation
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Performance optimizations will be disabled.")

# Optimized activation functions with numba JIT
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True)
    def _numba_sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @numba.jit(nopython=True)
    def _numba_sigmoid_derivative(sigmoid_output):
        return sigmoid_output * (1.0 - sigmoid_output)

    @numba.jit(nopython=True)
    def _numba_relu(x):
        return np.maximum(0.0, x)

    @numba.jit(nopython=True)
    def _numba_relu_derivative(x):
        return (x > 0.0).astype(np.float64)

    @numba.jit(nopython=True)
    def _numba_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @numba.jit(nopython=True)
    def _numba_bce_safe(y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    # Matrix operations
    @numba.jit(nopython=True)
    def _numba_forward_pass(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, 
                            hidden_activation_type, output_activation_type):
        # Input to Hidden Layer
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        
        # Apply chosen hidden activation
        if hidden_activation_type == 1:  # ReLU
            hidden_layer_output = _numba_relu(hidden_layer_input)
        else:  # Sigmoid
            hidden_layer_output = _numba_sigmoid(hidden_layer_input)
        
        # Hidden to Output Layer
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        
        # Apply chosen output activation
        if output_activation_type == 1:  # ReLU
            final_output = _numba_relu(output_layer_input)
        elif output_activation_type == 2:  # Linear
            final_output = output_layer_input
        else:  # Sigmoid
            final_output = _numba_sigmoid(output_layer_input)
            
        return hidden_layer_input, hidden_layer_output, output_layer_input, final_output

class SimpleNN:
    """A simple feedforward neural network with one hidden layer, built from scratch using NumPy.
       Supports Sigmoid/ReLU activations and MSE/BCE loss.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 hidden_activation='sigmoid', output_activation='sigmoid', loss='mse'):
        """Initialize weights and biases randomly.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons.
            hidden_activation (str): Activation for hidden layer ('sigmoid' or 'relu'). Default: 'sigmoid'.
            output_activation (str): Activation for output layer ('sigmoid', 'relu', 'linear'). Default: 'sigmoid'.
            loss (str): Loss function ('mse' or 'bce'). Default: 'mse'.
        """
        # --- Weight Initialization (using He initialization for ReLU potential) ---
        # He initialization (good for ReLU): scale by sqrt(2 / n_inputs)
        # Xavier/Glorot initialization (good for Sigmoid/Tanh): scale by sqrt(1 / n_inputs) or sqrt(2 / (n_inputs + n_outputs))
        # We'll use a simplified He-like scaling based on input size
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size) # He/Xavier variant
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size) # He/Xavier variant

        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Store activation and loss choices
        self.hidden_activation_type = hidden_activation
        self.output_activation_type = output_activation
        self.loss_type = loss
        
        # For optimized code paths with numba
        self._hidden_activation_code = 1 if hidden_activation == 'relu' else 0
        self._output_activation_code = 1 if output_activation == 'relu' else (2 if output_activation == 'linear' else 0)
        self._use_numba = NUMBA_AVAILABLE

        # Validate choices
        if hidden_activation not in ['sigmoid', 'relu']:
            raise ValueError("Invalid hidden_activation. Choose 'sigmoid' or 'relu'.")
        if output_activation not in ['sigmoid', 'relu', 'linear']:
            raise ValueError("Invalid output_activation. Choose 'sigmoid', 'relu', or 'linear'.")
        if loss not in ['mse', 'bce']:
            raise ValueError("Invalid loss function. Choose 'mse' or 'bce'.")
        if loss == 'bce' and output_activation != 'sigmoid':
            print("Warning: Binary Cross-Entropy loss is typically used with Sigmoid output activation.")

        # Placeholders for intermediate values needed in backpropagation
        self.X = None
        self.hidden_layer_input = None
        self.hidden_layer_output = None
        self.output_layer_input = None
        self.final_output = None

    # --- Activation Functions and Derivatives --- 
    def _sigmoid(self, x):
        # Sigmoid: 1 / (1 + exp(-x))
        if self._use_numba:
            return _numba_sigmoid(x)
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, sigmoid_output):
        # Derivative of sigmoid: output * (1 - output)
        # Takes sigmoid output as input for efficiency
        if self._use_numba:
            return _numba_sigmoid_derivative(sigmoid_output)
        return sigmoid_output * (1 - sigmoid_output)

    def _relu(self, x):
        # ReLU: max(0, x)
        if self._use_numba:
            return _numba_relu(x)
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        # Derivative of ReLU: 1 if x > 0, 0 otherwise
        # Takes the input to ReLU (x) as input
        if self._use_numba:
            return _numba_relu_derivative(x)
        return (x > 0).astype(float)

    # --- Loss Functions --- 
    def _mean_squared_error(self, y_true, y_pred):
        # MSE Loss: mean((y_true - y_pred)^2)
        if self._use_numba:
            return _numba_mse(y_true, y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def _binary_cross_entropy(self, y_true, y_pred):
        # BCE Loss: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        # Add epsilon to avoid log(0)
        if self._use_numba:
            return _numba_bce_safe(y_true, y_pred)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # --- Forward Propagation --- 
    def forward(self, X):
        self.X = X # Store input for backprop
        
        if self._use_numba:
            # Use optimized numba implementation
            self.hidden_layer_input, self.hidden_layer_output, self.output_layer_input, self.final_output = \
                _numba_forward_pass(X, self.weights_input_hidden, self.bias_hidden, 
                                    self.weights_hidden_output, self.bias_output,
                                    self._hidden_activation_code, self._output_activation_code)
        else:
            # Input to Hidden Layer
            self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            # Apply chosen hidden activation
            if self.hidden_activation_type == 'relu':
                self.hidden_layer_output = self._relu(self.hidden_layer_input)
            else: # Default to sigmoid
                self.hidden_layer_output = self._sigmoid(self.hidden_layer_input)

            # Hidden Layer to Output
            self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
            # Apply chosen output activation
            if self.output_activation_type == 'relu':
                self.final_output = self._relu(self.output_layer_input)
            elif self.output_activation_type == 'linear':
                self.final_output = self.output_layer_input # No activation
            else: # Default to sigmoid
                self.final_output = self._sigmoid(self.output_layer_input)

        return self.final_output

    # --- Backpropagation --- 
    def backward(self, y_true, learning_rate):
        n_samples = self.X.shape[0]

        # --- Calculate Output Layer Gradient (delta_output) --- 
        # This depends on the loss function and output activation function

        if self.loss_type == 'bce' and self.output_activation_type == 'sigmoid':
            # Simplified gradient for BCE loss with Sigmoid output:
            # dy/dz = y_pred - y_true
            delta_output = self.final_output - y_true
        else:
            # For other loss/activation combinations, use chain rule:
            # delta = dL/dy * dy/dz
            
            # First calculate dL/dy (derivative of loss with respect to output)
            if self.loss_type == 'mse':
                # dMSE/dy = 2*(y_pred - y_true)/n => we can simplify the 2/n
                dL_dy = (self.final_output - y_true)
            else: # BCE loss
                # dBCE/dy = -y_true/y_pred + (1-y_true)/(1-y_pred)
                epsilon = 1e-12
                y_pred_safe = np.clip(self.final_output, epsilon, 1 - epsilon)
                dL_dy = -y_true / y_pred_safe + (1 - y_true) / (1 - y_pred_safe)
            
            # Then calculate dy/dz (derivative of activation with respect to its input)
            if self.output_activation_type == 'sigmoid':
                dy_dz = self._sigmoid_derivative(self.final_output)
            elif self.output_activation_type == 'relu':
                dy_dz = self._relu_derivative(self.output_layer_input)
            else: # Linear activation
                dy_dz = 1
                
            # Combine to get output delta
            delta_output = dL_dy * dy_dz
        
        # --- Calculate Hidden Layer Gradient (delta_hidden) ---
        # delta_hidden = delta_output * weights_T * derivative_of_hidden_activation
        
        # Propagate error backwards
        delta_hidden_z = np.dot(delta_output, self.weights_hidden_output.T)
        
        # Calculate derivative based on hidden activation type
        if self.hidden_activation_type == 'sigmoid':
            delta_hidden = delta_hidden_z * self._sigmoid_derivative(self.hidden_layer_output)
        else: # ReLU
            delta_hidden = delta_hidden_z * self._relu_derivative(self.hidden_layer_input)
        
        # --- Calculate Gradients for Weights and Biases ---
        # weight gradients: dW = X.T * delta
        d_weights_hidden_output = np.dot(self.hidden_layer_output.T, delta_output)
        d_weights_input_hidden = np.dot(self.X.T, delta_hidden)
        
        # bias gradients: just sum the deltas
        d_bias_output = np.sum(delta_output, axis=0, keepdims=True)
        d_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)
        
        # --- Update Weights and Biases ---
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_output -= learning_rate * d_bias_output
        self.bias_hidden -= learning_rate * d_bias_hidden

    def train_batch(self, X_batch, y_batch, learning_rate):
        """Train on a batch of data.
        
        Args:
            X_batch (numpy.ndarray): Batch of input features
            y_batch (numpy.ndarray): Batch of target values
            learning_rate (float): Learning rate for weight updates
            
        Returns:
            float: Loss value for the batch
        """
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Calculate loss
        if self.loss_type == 'mse':
            loss = self._mean_squared_error(y_batch, y_pred)
        else: # BCE
            loss = self._binary_cross_entropy(y_batch, y_pred)
        
        # Backward pass
        self.backward(y_batch, learning_rate)
        
        return loss

    def train(self, X_train, y_train, epochs, learning_rate, verbose=100, batch_size=None):
        """Train the neural network.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for weight updates
            verbose (int): Print loss every `verbose` epochs. Set to 0 to disable.
            batch_size (int, optional): Batch size for mini-batch training. If None, use full batch.
            
        Returns:
            list: History of loss values
        """
        n_samples = X_train.shape[0]
        loss_history = []
        
        # Use mini-batch training if batch_size is specified
        use_batches = batch_size is not None and batch_size < n_samples
        
        for epoch in range(epochs):
            # Mini-batch training
            if use_batches:
                # Shuffle data
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                # Process each batch
                batch_losses = []
                for i in range(0, n_samples, batch_size):
                    # Get batch
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    
                    # Train on batch
                    batch_loss = self.train_batch(X_batch, y_batch, learning_rate)
                    batch_losses.append(batch_loss)
                
                # Average batch losses
                epoch_loss = np.mean(batch_losses)
            else:
                # Full batch training
                epoch_loss = self.train_batch(X_train, y_train, learning_rate)
            
            loss_history.append(epoch_loss)
            
            # Print progress
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
                
        return loss_history

    def predict(self, X):
        """Make predictions for input data.
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        return self.forward(X)

    def save_weights(self, filepath="simple_nn_weights.npz"):
        """Save the model weights to a file.
        
        Args:
            filepath (str): Path to save the weights
        """
        np.savez(filepath, 
                 w1=self.weights_input_hidden, 
                 b1=self.bias_hidden,
                 w2=self.weights_hidden_output, 
                 b2=self.bias_output)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath="simple_nn_weights.npz"):
        """Load model weights from a file.
        
        Args:
            filepath (str): Path to the saved weights
        """
        weights = np.load(filepath)
        self.weights_input_hidden = weights['w1']
        self.bias_hidden = weights['b1']
        self.weights_hidden_output = weights['w2']
        self.bias_output = weights['b2']
        print(f"Weights loaded from {filepath}") 