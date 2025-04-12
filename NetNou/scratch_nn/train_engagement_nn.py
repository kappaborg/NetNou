import numpy as np
from simple_nn import SimpleNN # Import our scratch NN class
import os

# --- 1. Create Artificial Dataset --- 
# Map emotions to numerical values (example mapping)
emotion_map = {
    'happy': 0.9,
    'neutral': 0.5,
    'sad': 0.1,
    'angry': 0.0,
    'surprise': 0.8,
    'fear': 0.2,
    'disgust': 0.0
}

# Input features (numerical representation of emotions)
X_train = np.array([
    [emotion_map['happy']],
    [emotion_map['surprise']],
    [emotion_map['neutral']],
    [emotion_map['neutral']], # More neutral examples
    [emotion_map['happy']],
    [emotion_map['sad']],
    [emotion_map['angry']],
    [emotion_map['fear']],
    [emotion_map['disgust']]
])

# Target output (engagement level: 0 = not engaged, 1 = engaged)
# Based on assumptions about emotions and engagement
y_train = np.array([
    [1.0], # happy -> engaged
    [1.0], # surprise -> engaged (could be argued)
    [0.7], # neutral -> somewhat engaged (or 0.5?)
    [0.6], # neutral -> somewhat engaged
    [0.9], # happy -> engaged
    [0.0], # sad -> not engaged
    [0.0], # angry -> not engaged
    [0.1], # fear -> likely not engaged
    [0.0]  # disgust -> not engaged
])

print("--- Artificial Dataset ---")
print("Input (Emotion Values):\n", X_train)
print("Target (Engagement Level):\n", y_train)
print("-------------------------")

# --- 2. Define and Initialize the Network ---
input_dim = X_train.shape[1]    # Should be 1
hidden_dim = 4                  # Number of neurons in the hidden layer
output_dim = y_train.shape[1]   # Should be 1

# Create the network instance with ReLU hidden layer and BCE loss
print("\n--- Initializing Network (ReLU Hidden, Sigmoid Output, BCE Loss) ---")
engagement_nn = SimpleNN(input_size=input_dim,
                         hidden_size=hidden_dim,
                         output_size=output_dim,
                         hidden_activation='relu', # Use ReLU for hidden layer
                         output_activation='sigmoid', # Keep Sigmoid for 0-1 output
                         loss='bce' # Use Binary Cross-Entropy loss
                         )

# --- 3. Train the Network ---
epochs = 15000
learning_rate = 0.05 # May need adjustment for ReLU/BCE
engagement_nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=1000)

# --- 4. Save the Trained Weights ---
# Ensure weights are saved in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_filename = "engagement_nn_weights.npz"
weights_filepath = os.path.join(script_dir, weights_filename)
engagement_nn.save_weights(filepath=weights_filepath)

# --- Optional: Test Predictions After Training ---
print("\n--- Testing Predictions After Training ---")
predictions = engagement_nn.predict(X_train)
for i in range(len(X_train)):
    emotion_val = X_train[i][0]
    target_eng = y_train[i][0]
    predicted_eng = predictions[i][0]
    # Find emotion name from value (for display)
    emotion_name = [name for name, val in emotion_map.items() if val == emotion_val][0]
    print(f"Emotion: {emotion_name} ({emotion_val:.1f}), Target: {target_eng:.1f}, Predicted: {predicted_eng:.4f}") 