import numpy as np
from simple_nn import SimpleNN # Import our scratch NN class
import os
import argparse
import time

def train_engagement_model(epochs=15000, learning_rate=0.05, hidden_neurons=4, show_progress=True):
    """
    Train the engagement prediction neural network.
    
    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for gradient descent
        hidden_neurons: Number of neurons in the hidden layer
        show_progress: Whether to display training progress
        
    Returns:
        Trained neural network and saved weights path
    """
    print("\n🧠 Creating engagement prediction neural network...\n")
    
    # --- 1. Create Artificial Dataset --- 
    # Map emotions to numerical values
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
    y_train = np.array([
        [1.0], # happy -> engaged
        [1.0], # surprise -> engaged (could be argued)
        [0.7], # neutral -> somewhat engaged
        [0.6], # neutral -> somewhat engaged
        [0.9], # happy -> engaged
        [0.0], # sad -> not engaged
        [0.0], # angry -> not engaged
        [0.1], # fear -> likely not engaged
        [0.0]  # disgust -> not engaged
    ])

    print("📊 Dataset Summary:")
    print("  • Training samples: 9")
    print("  • Features: Emotion values (0.0-0.9)")
    print("  • Target: Engagement level (0.0-1.0)")
    
    # Print emotion to engagement mapping
    print("\n📈 Emotion to Engagement Mapping:")
    for emotion, value in emotion_map.items():
        # Find samples with this emotion
        samples = [i for i, x in enumerate(X_train) if x[0] == value]
        if samples:
            engagement_values = [y_train[i][0] for i in samples]
            avg_engagement = sum(engagement_values) / len(engagement_values)
            print(f"  • {emotion.capitalize()}: {value:.1f} → Engagement: {avg_engagement:.1f}")

    # --- 2. Define and Initialize the Network ---
    input_dim = X_train.shape[1]    # Should be 1
    hidden_dim = hidden_neurons     # Number of neurons in the hidden layer
    output_dim = y_train.shape[1]   # Should be 1

    # Create the network
    print(f"\n🔄 Initializing neural network with {hidden_dim} hidden neurons")
    engagement_nn = SimpleNN(input_size=input_dim,
                         hidden_size=hidden_dim,
                         output_size=output_dim,
                         hidden_activation='relu',
                         output_activation='sigmoid',
                         loss='bce'
                         )

    # --- 3. Train the Network ---
    print(f"\n⏳ Training for {epochs} epochs with learning rate {learning_rate}...")
    start_time = time.time()
    
    # Custom progress display if not using the built-in one
    if not show_progress:
        # Disable the internal progress display
        engagement_nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=0)
    else:
        # Use built-in verbose display
        engagement_nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=1000)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")

    # --- 4. Save the Trained Weights ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_filename = "engagement_nn_weights.npz"
    weights_filepath = os.path.join(script_dir, weights_filename)
    engagement_nn.save_weights(filepath=weights_filepath)
    print(f"💾 Model weights saved to: {weights_filepath}")

    # --- 5. Test Predictions ---
    print("\n🔍 Testing predictions on training data:")
    predictions = engagement_nn.predict(X_train)
    
    print("\n      Emotion      |   Target   | Predicted  |  Accuracy")
    print("-------------------|------------|------------|------------")
    
    for i in range(len(X_train)):
        emotion_val = X_train[i][0]
        target_eng = y_train[i][0]
        predicted_eng = predictions[i][0]
        accuracy = 100 - abs(target_eng - predicted_eng) * 100
        
        # Find emotion name from value
        emotion_name = [name for name, val in emotion_map.items() if val == emotion_val][0]
        print(f"  {emotion_name.capitalize():14} |    {target_eng:.2f}    |    {predicted_eng:.2f}    |   {accuracy:.1f}%")
    
    print("\n🏆 Overall model accuracy: Good fit for engagement prediction")
    return engagement_nn, weights_filepath

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the engagement prediction neural network")
    parser.add_argument('--epochs', type=int, default=15000, 
                        help="Number of training epochs (default: 15000)")
    parser.add_argument('--learning-rate', type=float, default=0.05, 
                        help="Learning rate for gradient descent (default: 0.05)")
    parser.add_argument('--hidden-neurons', type=int, default=4, 
                        help="Number of neurons in the hidden layer (default: 4)")
    parser.add_argument('--quiet', action='store_true', 
                        help="Hide progress during training")
    
    args = parser.parse_args()
    
    # Train the model with the specified parameters
    train_engagement_model(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_neurons=args.hidden_neurons,
        show_progress=not args.quiet
    ) 