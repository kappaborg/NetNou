import cv2
from deepface import DeepFace
import time
import argparse
import os
import numpy as np
# Import our scratch NN class relative to the project root structure
import sys
# Assuming scratch_nn is in NetNou/scratch_nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scratch_nn'))
try:
    from simple_nn import SimpleNN
except ImportError:
    print("Error: Could not import SimpleNN. Ensure simple_nn.py is in NetNou/scratch_nn")
    sys.exit(1)

# --- Engagement NN Configuration ---
# This should match the training configuration
ENGAGEMENT_NN_INPUT_DIM = 1
ENGAGEMENT_NN_HIDDEN_DIM = 4 # Must match the hidden_dim used in training
ENGAGEMENT_NN_OUTPUT_DIM = 1
WEIGHTS_FILENAME = "engagement_nn_weights.npz"
# Construct the path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
ENGAGEMENT_NN_WEIGHTS_PATH = os.path.join(script_dir, '..', 'scratch_nn', WEIGHTS_FILENAME)

# Map emotions to numerical values (must match the training map)
emotion_map = {
    'happy': 0.9,
    'neutral': 0.5,
    'sad': 0.1,
    'angry': 0.0,
    'surprise': 0.8,
    'fear': 0.2,
    'disgust': 0.0
}
DEFAULT_EMOTION_VALUE = 0.5 # Value to use if emotion is not in map

# --- Load Scratch Engagement NN ---
def load_engagement_model():
    # Initialize the model with the SAME parameters used during training
    model = SimpleNN(input_size=ENGAGEMENT_NN_INPUT_DIM,
                     hidden_size=ENGAGEMENT_NN_HIDDEN_DIM,
                     output_size=ENGAGEMENT_NN_OUTPUT_DIM,
                     hidden_activation='relu',    # Match training
                     output_activation='sigmoid', # Match training
                     loss='bce'              # Match training
                     )
    # Load the weights trained with these parameters
    model.load_weights(ENGAGEMENT_NN_WEIGHTS_PATH)
    return model

# Load the model once at the start
try:
    engagement_model = load_engagement_model()
except Exception as e:
    print(f"Failed to load engagement NN model: {e}")
    engagement_model = None # Continue without engagement prediction if loading fails

def get_engagement_label(score):
    """Converts engagement score (0-1) to a label."""
    if score > 0.75:
        return "Engaged"
    elif score > 0.4:
        return "Neutral"
    else:
        return "Not Engaged"

def analyze_live_emotions(detector_backend='opencv', enforce_detection=False, analyze_every=1):
    """Captures video from the default camera and analyzes emotions in real-time.

    Args:
        detector_backend (str): Face detector backend for DeepFace.
        enforce_detection (bool): If True, stop if no face is detected.
        analyze_every (int): Analyze only every Nth frame (1 = every frame).
    """

    cap = cv2.VideoCapture(0) # 0 is usually the default webcam

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print(f"Starting live emotion analysis... Analyzing every {analyze_every} frame(s). Press 'q' to quit.")

    # For calculating FPS (optional)
    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0 # Counter for frame skipping
    last_analysis_results = [] # Store results from last analysis to display smoothly

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1

        # --- Resize Frame --- (Keep the 640x480 or preferred resolution)
        target_width = 640
        target_height = 480
        display_frame = frame.copy() # Work on a copy for display
        display_frame = cv2.resize(display_frame, (target_width, target_height))

        current_analysis_results = [] # Results for the current frame if analyzed

        # --- Analyze Frame Conditionally ---
        if frame_count % analyze_every == 0:
            try:
                # Use the resized frame for analysis for consistency, or original 'frame' if preferred
                analysis_frame = display_frame # Analyze the resized frame
                # analysis_frame = frame # Or analyze the original frame (might affect detector performance)

                results = DeepFace.analyze(
                    img_path=analysis_frame,
                    actions=('emotion',),
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    silent=True
                )

                # Process DeepFace results (list or dict)
                face_results = []
                if isinstance(results, list):
                    face_results = results
                elif isinstance(results, dict) and 'region' in results:
                     face_results = [results] # Wrap single dict in a list

                for face_info in face_results:
                    if 'region' in face_info and 'dominant_emotion' in face_info:
                        # Adjust coordinates relative to the resized display_frame
                        x, y, w, h = face_info['region']['x'], face_info['region']['y'], face_info['region']['w'], face_info['region']['h']
                        emotion = face_info['dominant_emotion']

                        engagement_label = "N/A"
                        engagement_score = -1.0
                        if engagement_model:
                            emotion_value = emotion_map.get(emotion.lower(), DEFAULT_EMOTION_VALUE)
                            nn_input = np.array([[emotion_value]])
                            engagement_score = engagement_model.predict(nn_input)[0][0]
                            engagement_label = get_engagement_label(engagement_score)
                        
                        current_analysis_results.append((x, y, w, h, emotion, engagement_label, engagement_score))
                
                # Update last known results
                last_analysis_results = current_analysis_results

            except Exception as e:
                # Ignore DeepFace errors if enforce_detection is False
                pass
                # Don't clear last_analysis_results here, keep showing the last successful one
        
        # --- Draw results on the display frame using last known results ---
        # Draw FPS
        new_frame_time = time.time()
        # Avoid division by zero on first frame
        fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw face boxes, emotion, and engagement using the last valid results
        for (x, y, w, h, emotion, engagement_label, engagement_score) in last_analysis_results:
             # Draw face rectangle
             cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
             # Prepare text
             emotion_text = f"Emotion: {emotion}"
             engagement_text = f"Engagement: {engagement_label}"
             if engagement_score >= 0:
                 engagement_text += f" ({engagement_score:.2f})"

             # Position text
             text_y = y - 10 if y > 30 else y + h + 20
             cv2.putText(display_frame, emotion_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
             cv2.putText(display_frame, engagement_text, (x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Display the resulting frame
        cv2.imshow('Live Emotion Analysis', display_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    print("Analysis stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live webcam emotion analysis using DeepFace and scratch NN for engagement.')
    parser.add_argument('--detector', type=str, default='opencv',
                        choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'],
                        help='Face detector backend. Faster options: opencv, ssd, mediapipe. Slower but potentially more accurate: mtcnn, retinaface. Default: opencv')
    parser.add_argument('--enforce', action='store_true',
                        help='Enforce face detection (stop if no face found). Default: False')
    parser.add_argument('--analyze_every', type=int, default=1,
                        help='Analyze only every Nth frame to improve FPS (e.g., 1=every frame, 2=every other frame). Default: 1')

    args = parser.parse_args()

    if args.analyze_every < 1:
        print("Warning: --analyze_every must be 1 or greater. Setting to 1.")
        args.analyze_every = 1

    analyze_live_emotions(detector_backend=args.detector,
                        enforce_detection=args.enforce,
                        analyze_every=args.analyze_every) 