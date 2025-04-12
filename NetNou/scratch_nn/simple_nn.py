import numpy as np

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
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, sigmoid_output):
        # Derivative of sigmoid: output * (1 - output)
        # Takes sigmoid output as input for efficiency
        return sigmoid_output * (1 - sigmoid_output)

    def _relu(self, x):
        # ReLU: max(0, x)
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        # Derivative of ReLU: 1 if x > 0, 0 otherwise
        # Takes the input to ReLU (x) as input
        return (x > 0).astype(float)

    # --- Loss Functions --- 
    def _mean_squared_error(self, y_true, y_pred):
        # MSE Loss: mean((y_true - y_pred)^2)
        return np.mean((y_true - y_pred) ** 2)

    def _binary_cross_entropy(self, y_true, y_pred):
        # BCE Loss: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        # Add epsilon to avoid log(0)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # --- Forward Propagation --- 
    def forward(self, X):
        self.X = X # Store input for backprop

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
            # delta_output = y_pred - y_true (or y_true - y_pred depending on convention, let's use y_pred - y_true)
            # We calculated y_pred (self.final_output) in the forward pass
            output_delta = self.final_output - y_true
        else: # Default to MSE or handle other combinations explicitly if needed
            # General case: error * derivative_of_output_activation(output_layer_input)
            output_error = self.final_output - y_true # For MSE: dLoss/dPred = y_pred - y_true

            if self.output_activation_type == 'sigmoid':
                # Derivative uses the activation *output* (self.final_output)
                output_delta = output_error * self._sigmoid_derivative(self.final_output)
            elif self.output_activation_type == 'relu':
                 # Derivative uses the activation *input* (self.output_layer_input)
                output_delta = output_error * self._relu_derivative(self.output_layer_input)
            elif self.output_activation_type == 'linear':
                output_delta = output_error * 1 # Derivative of linear is 1
            else:
                 # Should not happen due to init validation, but default to sigmoid
                 output_delta = output_error * self._sigmoid_derivative(self.final_output)

        # --- Calculate Hidden Layer Gradient (delta_hidden) ---
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        
        # Derivative depends on the hidden activation function
        if self.hidden_activation_type == 'relu':
            # Derivative uses the activation *input* (self.hidden_layer_input)
            hidden_delta = hidden_error * self._relu_derivative(self.hidden_layer_input)
        else: # Default to sigmoid
            # Derivative uses the activation *output* (self.hidden_layer_output)
            hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_layer_output)

        # --- Update Weights and Biases --- 
        # Normalize gradients by number of samples
        # Apply gradient descent: subtract the scaled gradient
        # Hidden -> Output
        self.weights_hidden_output -= np.dot(self.hidden_layer_output.T, output_delta) * learning_rate / n_samples
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate / n_samples

        # Input -> Hidden
        self.weights_input_hidden -= np.dot(self.X.T, hidden_delta) * learning_rate / n_samples
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate / n_samples

    # --- Training Loop --- 
    def train(self, X_train, y_train, epochs, learning_rate, verbose=100):
        print(f"Starting training for {epochs} epochs...")
        print(f" Params: LR={learning_rate}, HiddenAct={self.hidden_activation_type}, OutputAct={self.output_activation_type}, Loss={self.loss_type}")
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)

            # Calculate loss based on selected type
            if self.loss_type == 'bce':
                loss = self._binary_cross_entropy(y_train, y_pred)
            else: # Default to mse
                loss = self._mean_squared_error(y_train, y_pred)

            # Backward pass and update weights
            # Note: Learning rate sign convention change in weight update step
            self.backward(y_train, learning_rate)

            # Print loss periodically
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        print("Training finished.")

    # --- Prediction --- 
    def predict(self, X):
        return self.forward(X)

    # --- Save/Load Weights --- 
    def save_weights(self, filepath="simple_nn_weights.npz"):
        # (Save logic remains the same)
        np.savez(filepath,
                 w_ih=self.weights_input_hidden,
                 w_ho=self.weights_hidden_output,
                 b_h=self.bias_hidden,
                 b_o=self.bias_output)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath="simple_nn_weights.npz"):
        # (Load logic remains the same)
        try:
            data = np.load(filepath)
            self.weights_input_hidden = data['w_ih']
            self.weights_hidden_output = data['w_ho']
            self.bias_hidden = data['b_h']
            self.bias_output = data['b_o']
            print(f"Weights loaded from {filepath}")
        except FileNotFoundError:
            print(f"Error: Weight file not found at {filepath}")
            raise # Re-raise exception to indicate failure
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise # Re-raise exception

# --- Example Usage Updated --- 
if __name__ == '__main__':
    # Example: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define network parameters
    input_dim = X.shape[1]
    hidden_dim = 8 # Increased hidden units
    output_dim = y.shape[1]

    # --- Create and train with ReLU hidden layer and BCE loss ---
    print("\n--- Training with ReLU hidden and BCE loss ---")
    nn_relu_bce = SimpleNN(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim,
                           hidden_activation='relu', output_activation='sigmoid', loss='bce')
    nn_relu_bce.train(X, y, epochs=15000, learning_rate=0.1, verbose=1000)

    # Make predictions
    predictions = nn_relu_bce.predict(X)
    print("\nPredictions (ReLU hidden, BCE loss):")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Predicted: {predictions[i][0]:.4f} -> Rounded: {np.round(predictions[i][0])}")

    # Save weights
    nn_relu_bce.save_weights("xor_relu_bce_weights.npz")

    # --- Example with default Sigmoid/MSE (Original behavior) ---
    # print("\n--- Training with Sigmoid hidden and MSE loss ---")
    # nn_sigmoid_mse = SimpleNN(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
    # nn_sigmoid_mse.train(X, y, epochs=15000, learning_rate=0.5, verbose=1000)
    # predictions_sm = nn_sigmoid_mse.predict(X)
    # print("\nPredictions (Sigmoid hidden, MSE loss):")
    # for i in range(len(X)):
    #     print(f"Input: {X[i]}, Target: {y[i][0]}, Predicted: {predictions_sm[i][0]:.4f} -> Rounded: {np.round(predictions_sm[i][0])}")
    # nn_sigmoid_mse.save_weights("xor_sigmoid_mse_weights.npz") 