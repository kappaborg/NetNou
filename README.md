# NetNou - AI Student Engagement Analysis

This project aims to analyze student engagement using artificial intelligence techniques, particularly face and emotion analysis, combined with a neural network implemented from scratch to demonstrate foundational principles.

## Objective

The primary goal of this project is to perform the following analyses using camera input:

1.  **Liveness Detection:** Determine if the face presented to the camera is live or a spoof (e.g., photo, video replay). *Current performance is limited due to data constraints.*
2.  **Emotion Recognition:** Identify the dominant emotion (e.g., happy, sad, neutral) displayed on detected faces.
3.  **Engagement Analysis:** Estimate the student's engagement level ("Engaged", "Neutral", "Not Engaged") based on the detected emotion, using a simple neural network coded from scratch with NumPy.

## Modules

The project is organized into modules within the `NetNou` directory:

### 1. Liveness Detection (`NetNou/liveness_detection`)

*   **Purpose:** Detect presentation attacks (spoofing).
*   **Technology:** A simple Convolutional Neural Network (CNN) implemented using `TensorFlow/Keras` (`model.py`).
*   **Data:** Samples derived from the [Kaggle: Anti-Spoofing Dataset by tapakah68](https://www.kaggle.com/datasets/tapakah68/anti-spoofing). Frames were extracted from videos (`extract_frames.py`) and renamed (`rename_files.py`) into `data/real` and `data/spoof` subdirectories.
*   **Status:** The performance of the current model is **unreliable** due to insufficient data volume and significant class imbalance. Class weights and data augmentation were attempted during training with limited success.
*   **Usage:**
    *   **Training:** `python NetNou/liveness_detection/train.py --data_dir NetNou/liveness_detection/data` (Run from project root; saves model to root).
    *   **Prediction:** `python NetNou/liveness_detection/predict.py <image_path>` (Run from project root).

### 2. Emotion Recognition & Engagement Analysis (`NetNou/emotion_recognition`)

*   **Purpose:** Detect faces and emotions from a live camera feed, then predict engagement level using the scratch-built NN.
*   **Technology:**
    *   `opencv-python`: Camera access and basic image processing.
    *   `deepface`: A comprehensive library for face analysis, including detection and emotion recognition. Supports multiple detector backends (`--detector` argument).
    *   `NetNou/scratch_nn/simple_nn.py`: The simple feedforward neural network, **coded from scratch using NumPy**, used for engagement prediction.
*   **Status:** Functional. Displays live video with bounding boxes, detected emotion, and predicted engagement level (with score). FPS and analysis frequency are configurable.
*   **Usage:**
    *   **Live Analysis:** `python NetNou/emotion_recognition/live_emotion.py [ARGUMENTS]` (Run from project root).
        *   `--detector <model>`: Select face detector backend (e.g., `opencv`, `ssd`, `mediapipe`). Use `-h` for more info.
        *   `--analyze_every <N>`: Analyze only every Nth frame to improve FPS (e.g., `1`, `2`, `3`). Default: `1`.
    *   **Train Engagement NN:** `python NetNou/scratch_nn/train_engagement_nn.py` (Trains the scratch NN on artificial data and saves weights).

### 3. Scratch Neural Network (`NetNou/scratch_nn`)

*   **Purpose:** Demonstrate fundamental neural network concepts by implementing a simple feedforward network using only NumPy.
*   **File:** `simple_nn.py`
*   **Features:**
    *   Single hidden layer architecture.
    *   Selectable activation functions (Sigmoid, ReLU) via initialization arguments.
    *   Selectable loss functions (MSE, Binary Cross-Entropy) via initialization arguments.
    *   Forward propagation implementation.
    *   Backpropagation implementation for weight updates (Gradient Descent).
    *   Helper methods for saving and loading trained weights (`.npz` format).
*   **Application:** An instance of this `SimpleNN` class, trained via `train_engagement_nn.py`, is loaded by `live_emotion.py` to predict student engagement based on the emotion detected by DeepFace.

## Installation

1.  **Prerequisites:** Python 3.9+ recommended.
2.  **Clone Repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <project_root_directory> # e.g., cd ai-student-attendance-system
    ```
3.  **Install Dependencies:** From the project root directory, run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `tensorflow`, `opencv-python`, `deepface`, `tf-keras`, `numpy`, and their dependencies.
4.  **Potential Issues & Troubleshooting:**
    *   **TensorFlow Conflicts:** If you encounter errors related to `tensorflow` or `tensorflow.keras` during installation or runtime (especially `ModuleNotFoundError`), it might be due to conflicting versions (e.g., `tensorflow-macos`). Try cleaning the environment by uninstalling existing TensorFlow versions and performing a clean install: `pip uninstall tensorflow tensorflow-macos tf-keras -y` followed by `pip install --upgrade --force-reinstall tensorflow "numpy<2.0" tf-keras`. Ensure NumPy version compatibility with OpenCV (`numpy<2.0` is often required).
    *   **DeepFace Model Downloads:** The first time `live_emotion.py` is run, `deepface` will automatically download the required face detection and emotion recognition models. Ensure you have an active internet connection.

## Usage

Make sure you are in the project's root directory in your terminal.

*   **Live Emotion, Engagement, Age & Gender Analysis:**
    ```bash
    # Run with default settings (opencv detector, analyze every frame)
    python NetNou/demographic_analysis/live_demographics.py

    # Run with SSD detector, analyzing every 3rd frame for higher FPS
    python NetNou/demographic_analysis/live_demographics.py --detector ssd --analyze_every 3

    # Run with mediapipe detector (often faster)
    python NetNou/demographic_analysis/live_demographics.py --detector mediapipe

    # See all options
    python NetNou/demographic_analysis/live_demographics.py -h
    ```
    *   `--detector`: Choose face detector ('opencv', 'ssd', 'mediapipe' are generally faster; 'mtcnn', 'retinaface', 'dlib' might be slower but more accurate).
    *   `--analyze_every`: Set analysis frequency (1=every frame, 2=every other, etc.) to balance FPS and update rate.
    *   Press 'q' in the displayed window to quit.
    *   *Note:* If run without arguments, the script will prompt you to select these options interactively.

*   **Train Scratch NN for Engagement:** 
    ```bash
    python NetNou/scratch_nn/train_engagement_nn.py
    ```
    This creates/updates the `NetNou/scratch_nn/engagement_nn_weights.npz` file.

*   **Liveness Detection (Experimental):**
    ```bash
    # Data Preparation (if using the Kaggle dataset for the first time)
    # 1. Place archive.zip in NetNou/liveness_detection/data
    # 2. unzip ... (extract the archive)
    # 3. python NetNou/liveness_detection/extract_frames.py --output_dir NetNou/liveness_detection/data/spoof (extract frames)
    # 4. python NetNou/liveness_detection/rename_files.py (rename files)
    # 5. rm -rf ... (remove original video folders, csv, zip from data/)

    # Train the model
    python NetNou/liveness_detection/train.py --data_dir NetNou/liveness_detection/data

    # Make a prediction on an image
    python NetNou/liveness_detection/predict.py <path_to_image.jpg>
    ```

## Scratch NN Details (`simple_nn.py`)

This project features a basic feedforward neural network implemented from scratch in `NetNou/scratch_nn/simple_nn.py` using only NumPy. This serves as an educational example demonstrating core neural network mechanics:

*   **Architecture:** Input layer, single hidden layer, output layer.
*   **Activations:** Configurable during initialization (`hidden_activation`, `output_activation`). Supports 'sigmoid' and 'relu'. 'linear' is also available for the output.
*   **Loss Functions:** Configurable (`loss`). Supports Mean Squared Error ('mse') and Binary Cross-Entropy ('bce').
*   **Learning:** Implements backpropagation with standard gradient descent to update weights and biases based on the calculated gradients of the loss function.
*   **Integration:** The trained scratch NN is used in `live_emotion.py` to infer student engagement levels based on emotion data provided by the `deepface` library, showcasing how a foundational NN can be integrated into a larger application.

## Future Improvements

*   Improve the Liveness Detection model with more data, better data augmentation, or a more sophisticated architecture.
*   Add demographic prediction (age, gender) using `deepface` or other methods (considering ethical implications).
*   Enhance the scratch NN (`SimpleNN`) with features like multiple hidden layers, different optimizers (e.g., Adam, Momentum), or regularization techniques (L2, Dropout) - *note: this would significantly increase the complexity of the scratch implementation.*
*   Develop a user-friendly interface (Web UI or Desktop GUI).
*   Implement logging and visualization for tracking analysis results (emotion/engagement changes over time). 