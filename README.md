# NetNou - NN Engagement Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This project analyzes Face engagement in real-time using computer vision and a neural network implemented from scratch.

## Core Functionality

The main application (`NetNou/demographic_analysis/live_demographics.py`) uses a webcam to:

1.  **Detect Faces:** Identifies faces in the video stream using various backends provided by the `deepface` library.
2.  **Analyze Demographics & Emotion:** Estimates the **age**, **gender**, and dominant **emotion** (e.g., happy, sad, neutral) for each detected face using `deepface`.
3.  **Predict Engagement:** Uses a **simple neural network built from scratch** with NumPy (`NetNou/scratch_nn/simple_nn.py`) to predict the student's engagement level ("Engaged", "Neutral", "Not Engaged") based on the detected emotion.

The live video feed displays bounding boxes around faces along with the analyzed information (emotion, age, gender, engagement level).

## Key Features

*   **Real-time Analysis:** Processes webcam feed for immediate feedback.
*   **DeepFace Integration:** Leverages the powerful `deepface` library for robust face detection, emotion recognition, age, and gender estimation.
*   **Scratch Neural Network (`SimpleNN`):** Includes a feedforward neural network implemented purely with NumPy (`NetNou/scratch_nn/simple_nn.py`) to demonstrate fundamental concepts (forward/backward propagation). This network is trained to predict engagement from emotion.
*   **Configurable Performance:** Allows selection of different face detection backends and analysis frequency to balance accuracy and speed (FPS).

## Project Structure

```
.
├── NetNou/
│   ├── demographic_analysis/  # Main live analysis script
│   │   └── live_demographics.py
│   ├── scratch_nn/          # Neural Network from scratch
│   │   ├── simple_nn.py
│   │   ├── train_engagement_nn.py
│   │   └── engagement_nn_weights.npz # Trained weights
│   └── emotion_recognition/ # (Older/related code, main is demographic_analysis)
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   Git (for cloning)
2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd ai-student-attendance-system # Or your project directory name
    ```
3.  **Install Dependencies:**
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note:** This installs `opencv-python`, `deepface`, `tensorflow` (as a dependency of `deepface`), `numpy`, etc.
    *   **Troubleshooting:** If you encounter issues with TensorFlow versions (`tensorflow`, `tf-keras`), try cleaning and reinstalling:
        ```bash
        pip uninstall tensorflow tensorflow-macos tf-keras -y
        pip install --upgrade --force-reinstall tensorflow "numpy<2.0" # Ensure NumPy compatibility
        ```
4.  **Model Downloads:** The first time you run the main script, `deepface` will automatically download necessary pre-trained models. Ensure you have an internet connection.

## Usage

Navigate to the project's root directory in your terminal.

**1. Run Live Analysis:**

```bash
python NetNou/demographic_analysis/live_demographics.py [OPTIONS]
```

*   **Examples:**
    *   `python NetNou/demographic_analysis/live_demographics.py` (Runs with interactive prompts or defaults)
    *   `python NetNou/demographic_analysis/live_demographics.py --detector mediapipe --analyze_every 2` (Uses MediaPipe detector, analyzes every 2nd frame)
*   **Options:**
    *   `--detector <backend>`: Choose face detector (`opencv`, `ssd`, `mediapipe`, `mtcnn`, etc.). Use `-h` for details. `mediapipe` or `ssd` often offer a good speed/accuracy balance.
    *   `--analyze_every <N>`: Analyze every Nth frame (e.g., `1`, `2`, `3`). Higher numbers increase FPS but decrease update frequency.
    *   `--enforce`: Stop if no face is detected.
*   **Quit:** Press 'q' in the OpenCV window to stop the analysis.

**2. Train the Engagement Neural Network (Optional):**

The repository includes pre-trained weights (`engagement_nn_weights.npz`). If you want to retrain the scratch NN (e.g., after modifying `simple_nn.py` or `train_engagement_nn.py`):

```bash
python NetNou/scratch_nn/train_engagement_nn.py
```
This will update/create the `NetNou/scratch_nn/engagement_nn_weights.npz` file.

## Scratch Neural Network Details (`simple_nn.py`)

This project includes a simple feedforward neural network (`SimpleNN`) built from scratch using only NumPy as an educational component.

*   **Purpose:** Demonstrates core NN concepts like layers, activation functions (Sigmoid, ReLU), loss functions (MSE, BCE), forward propagation, and backpropagation (gradient descent).
*   **Usage:** It's loaded by `live_demographics.py` and uses the emotion detected by `deepface` to predict student engagement.

## Future Ideas

*   Enhance the scratch `SimpleNN` (more layers, optimizers, regularization).
*   Develop a graphical user interface (GUI).
*   Log analysis results over time.

## License

[MIT](https://opensource.org/licenses/MIT)