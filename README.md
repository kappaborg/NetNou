# NetNou: AI Student Attendance & Engagement Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful real-time face analysis system that leverages computer vision and neural networks to track student engagement and attendance.

## Features

- **Real-time Face Detection & Analysis**: Detect faces and analyze demographics in live video
- **Emotion Recognition**: Identify emotions (happy, sad, angry, etc.) of detected faces
- **Engagement Prediction**: Determine student engagement level using a custom neural network
- **Demographics Analysis**: Estimate age and gender of each detected person
- **Performance Optimization**: Multiple detection backends with adjustable analysis frequency
- **Web Application**: Easy-to-use interface with Flask (NetNou-WebApp)
- **Custom Neural Network**: Implementation from scratch using NumPy

## Project Structure

```
.
├── NetNou/                        # Core analysis module
│   ├── demographic_analysis/      # Real-time face & engagement analysis
│   │   ├── live_demographics.py   # Main analysis script
│   │   └── optimized_demo.py      # Optimized version for better performance
│   ├── scratch_nn/                # Neural network implementation
│   │   ├── simple_nn.py           # Neural network built from scratch with NumPy
│   │   └── train_engagement_nn.py # Training script for the engagement model
│   └── emotion_recognition/       # Additional emotion analysis tools
├── NetNou-WebApp/                 # Web application module
│   └── (Flask application files)
└── requirements.txt               # Project dependencies
```

## Installation

1. **Prerequisites**:
   - Python 3.9 or higher
   - Webcam (for live analysis)

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ai-student-attendance-system.git
   cd ai-student-attendance-system
   ```

3. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: On first run, DeepFace will automatically download required models.

## Usage

### 1. Real-time Engagement Analysis

Run the live analysis script to analyze faces, emotions, and engagement in real-time:

```bash
python NetNou/demographic_analysis/live_demographics.py
```

This will:
- Open your webcam
- Detect faces in the video feed
- Analyze emotions, age, and gender
- Predict engagement level

**Command-line Options**:

```bash
# Use a specific face detector (faster/more accurate options available)
python NetNou/demographic_analysis/live_demographics.py --detector mediapipe

# Analyze every Nth frame to improve performance
python NetNou/demographic_analysis/live_demographics.py --analyze_every 2

# Enforce face detection (stop if no face is found)
python NetNou/demographic_analysis/live_demographics.py --enforce

# Combine options
python NetNou/demographic_analysis/live_demographics.py --detector ssd --analyze_every 3
```

Available detectors (from fastest to most accurate):
- `opencv` (default, fast but less accurate)
- `ssd` (good balance of speed and accuracy)
- `mediapipe` (good balance of speed and accuracy)
- `mtcnn` (slower but more accurate)
- `retinaface` (slowest but most accurate)

### 2. Train the Engagement Neural Network

If you want to retrain the engagement prediction model:

```bash
python NetNou/scratch_nn/train_engagement_nn.py
```

This will train the neural network based on emotion-to-engagement mappings and save the weights to `engagement_nn_weights.npz`.

### 3. Web Application (NetNou-WebApp)

To run the web application:

```bash
cd NetNou-WebApp
python run.py
```

Access the web interface at `http://localhost:5000` in your browser.

## How It Works

### Facial Analysis Pipeline

1. **Face Detection**: Identifies faces in each frame using the selected backend
2. **Emotion Analysis**: DeepFace analyzes the dominant emotion
3. **Demographics**: Age and gender are estimated for each face
4. **Engagement Prediction**: The neural network predicts engagement based on emotional cues
5. **Visualization**: Results are displayed with bounding boxes and text

### Neural Network Architecture

The `SimpleNN` class implements a feedforward neural network with:
- One input layer (emotion values)
- One hidden layer with configurable neurons
- One output layer (engagement score)
- Support for different activation functions (ReLU, Sigmoid) and loss functions (MSE, BCE)
- Numba optimization for faster computation when available

## Troubleshooting

- **Performance Issues**: Try a faster detector (`opencv` or `ssd`) or increase `analyze_every` value
- **Detection Problems**: Try a more accurate detector (`retinaface` or `mtcnn`)
- **TensorFlow Errors**: Ensure compatible versions with:
  ```bash
  pip uninstall tensorflow tensorflow-macos tf-keras -y
  pip install --force-reinstall tensorflow "numpy<2.0"
  ```
- **Webcam Access**: Ensure your webcam is connected and not in use by another application

## License

[MIT License](LICENSE)

## Acknowledgments

- DeepFace library for facial analysis
- NumPy for mathematical operations
- OpenCV for computer vision functionality