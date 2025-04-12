import cv2
from deepface import DeepFace
import time
import argparse
import os
import numpy as np
# Import our scratch NN class relative to the project root structure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scratch_nn'))
try:
    from simple_nn import SimpleNN
except ImportError:
    print("Error: Could not import SimpleNN. Ensure simple_nn.py is in NetNou/scratch_nn")
    sys.exit(1)

# --- Engagement NN Configuration ---
ENGAGEMENT_NN_INPUT_DIM = 1
ENGAGEMENT_NN_HIDDEN_DIM = 4
ENGAGEMENT_NN_OUTPUT_DIM = 1
WEIGHTS_FILENAME = "engagement_nn_weights.npz"
script_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust path assuming this script is in NetNou/demographic_analysis
ENGAGEMENT_NN_WEIGHTS_PATH = os.path.join(script_dir, '..', 'scratch_nn', WEIGHTS_FILENAME)

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
DEFAULT_EMOTION_VALUE = 0.5

# --- Load Scratch Engagement NN ---
def load_engagement_model():
    model = SimpleNN(input_size=ENGAGEMENT_NN_INPUT_DIM,
                     hidden_size=ENGAGEMENT_NN_HIDDEN_DIM,
                     output_size=ENGAGEMENT_NN_OUTPUT_DIM,
                     hidden_activation='relu', 
                     output_activation='sigmoid',
                     loss='bce'
                     )
    model.load_weights(ENGAGEMENT_NN_WEIGHTS_PATH)
    return model

try:
    engagement_model = load_engagement_model()
except Exception as e:
    print(f"Warning: Failed to load engagement NN model: {e}. Engagement prediction disabled.")
    engagement_model = None

def get_engagement_label(score):
    if score > 0.75:
        return "Engaged"
    elif score > 0.4:
        return "Neutral"
    else:
        return "Not Engaged"

# --- Interactive Input Helper ---
def get_user_choice(prompt, options_with_hints, default_value):
    """Prompts the user to choose from a list of options with hints."""
    print(prompt)
    # Keep track of the actual option values without hints
    option_values = []
    for i, (option, hint) in enumerate(options_with_hints):
        print(f"  {i+1}. {option} {hint}")
        option_values.append(option)

    # Find the default option's value for the prompt message
    default_display = default_value # Assume default is one of the option values

    while True:
        try:
            choice = input(f"Enter number (1-{len(option_values)}) or press Enter for default [{default_display}]): ")
            if not choice: # User pressed Enter
                return default_value
            choice_int = int(choice)
            if 1 <= choice_int <= len(option_values):
                return option_values[choice_int - 1] # Return the actual value
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_int(prompt, default_value):
    """Prompts the user to enter an integer >= 1."""
    while True:
        try:
            # Add hint about FPS impact to the prompt string itself
            full_prompt = f"{prompt} (Higher number = Higher FPS, less frequent updates. Press Enter for default [{default_value}]): "
            choice = input(full_prompt)
            if not choice:
                return default_value
            choice_int = int(choice)
            if choice_int >= 1:
                return choice_int
            else:
                print("Please enter a positive integer (>= 1).")
        except ValueError:
            print("Invalid input. Please enter an integer.")

def analyze_live_demographics(detector_backend='opencv', enforce_detection=False, analyze_every=1):
    """Captures video and analyzes emotions, engagement, age, and gender."""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print(f"Starting live analysis (Emotion, Engagement, Age, Gender)... Analyzing every {analyze_every} frame(s). Press 'q' to quit.")

    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    last_analysis_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        frame_count += 1

        target_width = 640
        target_height = 480
        display_frame = frame.copy()
        display_frame = cv2.resize(display_frame, (target_width, target_height))

        current_analysis_results = []

        if frame_count % analyze_every == 0:
            try:
                analysis_frame = display_frame
                # Request emotion, age, and gender analysis
                results = DeepFace.analyze(
                    img_path=analysis_frame,
                    actions=('emotion', 'age', 'gender'), # Add age and gender
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    silent=True
                )

                face_results = []
                if isinstance(results, list):
                    face_results = results
                elif isinstance(results, dict) and 'region' in results:
                     face_results = [results]

                for face_info in face_results:
                    # Extract common info
                    region = face_info.get('region', {})
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                    
                    # Get predictions (handle potential missing keys)
                    emotion = face_info.get('dominant_emotion', 'N/A')
                    age = face_info.get('age', 'N/A')
                    gender = face_info.get('dominant_gender', 'N/A') # DeepFace uses dominant_gender

                    # Predict engagement
                    engagement_label = "N/A"
                    engagement_score = -1.0
                    if engagement_model and emotion != 'N/A':
                        emotion_value = emotion_map.get(emotion.lower(), DEFAULT_EMOTION_VALUE)
                        nn_input = np.array([[emotion_value]])
                        engagement_score = engagement_model.predict(nn_input)[0][0]
                        engagement_label = get_engagement_label(engagement_score)
                    
                    current_analysis_results.append((x, y, w, h, emotion, engagement_label, engagement_score, age, gender))
                
                last_analysis_results = current_analysis_results

            except Exception as e:
                # print(f"Analysis error: {e}") # Optional debug
                pass
        
        # --- Draw results --- 
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (x, y, w, h, emotion, engagement_label, engagement_score, age, gender) in last_analysis_results:
             cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
             
             # Position text
             text_y = y - 10 if y > 50 else y + h + 20 # Adjust starting y pos
             line_height = 20

             # Display Info Line by Line
             cv2.putText(display_frame, f"Emotion: {emotion}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
             
             engagement_text = f"Engagement: {engagement_label}"
             if engagement_score >= 0:
                 engagement_text += f" ({engagement_score:.2f})"
             cv2.putText(display_frame, engagement_text, (x, text_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

             cv2.putText(display_frame, f"Age: {age}", (x, text_y + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Cyan for age
             cv2.putText(display_frame, f"Gender: {gender}", (x, text_y + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2) # Magenta for gender

        cv2.imshow('Live Analysis (Emotion, Engagement, Age, Gender)', display_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Analysis stopped.")

if __name__ == "__main__":
    # Define available detectors with hints
    available_detectors_with_hints = [
        ('opencv',    "(Fastest, Baseline Accuracy)"),
        ('ssd',       "(Fast, Good Balance)"),
        ('mediapipe', "(Fast, Good Accuracy)"),
        ('dlib',      "(Moderate Speed, Good Accuracy, Requires dlib install)"), # Added dlib hint
        ('mtcnn',     "(Slower, Accurate)"),
        ('retinaface',"(Slower, Very Accurate)")
    ]
    # Extract just the names for argparse choices
    available_detectors = [d[0] for d in available_detectors_with_hints]
    default_detector = 'opencv'
    default_analyze_every = 1

    parser = argparse.ArgumentParser(description='Live webcam analysis for Emotion, Engagement, Age, and Gender.')
    parser.add_argument('--detector', type=str, default=None,
                        choices=available_detectors,
                        help='Face detector backend. Options have different speed/accuracy trade-offs. Default: interactively select or opencv')
    parser.add_argument('--analyze_every', type=int, default=None,
                        help='Analyze only every Nth frame to improve FPS (e.g., 1=every frame, 2=every other frame). Default: interactively select or 1')
    parser.add_argument('--enforce', action='store_true',
                        help='Enforce face detection (stop if no face found). Default: False')

    args = parser.parse_args()

    # --- Interactive Selection if Arguments Not Provided ---
    selected_detector = args.detector
    if selected_detector is None:
        print("\n--- Configuration Selection ---")
        # Pass the list with hints to the updated get_user_choice function
        selected_detector = get_user_choice("Select Face Detector Backend:", available_detectors_with_hints, default_detector)

    selected_analyze_every = args.analyze_every
    if selected_analyze_every is None:
        if args.detector is None:
             print("---")
        else:
             print("\n--- Configuration Selection ---")
        # Pass the modified prompt hint implicitly via the updated get_user_int function
        selected_analyze_every = get_user_int("Analyze every Nth frame (e.g., 1, 2, 3,...)", default_analyze_every)

    # Validate selected_analyze_every again (in case default was invalid, though unlikely here)
    if selected_analyze_every < 1:
        print("Warning: analyze_every must be 1 or greater. Setting to 1.")
        selected_analyze_every = 1

    print("\nStarting analysis with:")
    print(f"  Detector Backend: {selected_detector}")
    print(f"  Analyze Every: {selected_analyze_every} frame(s)")
    print(f"  Enforce Detection: {args.enforce}")
    print("-------------------------")

    # Call the main function with the determined parameters
    analyze_live_demographics(detector_backend=selected_detector,
                              enforce_detection=args.enforce,
                              analyze_every=selected_analyze_every) 