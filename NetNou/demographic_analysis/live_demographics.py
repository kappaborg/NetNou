import cv2
from deepface import DeepFace
import time
import argparse
import os
import numpy as np
import threading
import queue
import psutil
import face_recognition
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

# --- Performance & Optimization Settings ---
# Size for analysis frames (smaller = faster)
ANALYSIS_FRAME_WIDTH = 320
ANALYSIS_FRAME_HEIGHT = 240
# Face tracking constants
TRACKING_MAX_FRAMES = 30  # Reset tracking after this many frames
# Worker threads
MAX_QUEUE_SIZE = 5  # Maximum frames waiting for analysis

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
    print(f"\n📋 {prompt}")
    # Keep track of the actual option values without hints
    option_values = []
    for i, (option, hint) in enumerate(options_with_hints):
        print(f"  {i+1}. {option} {hint}")
        option_values.append(option)
    
    # Find the default index for display
    default_index = option_values.index(default_value) if default_value in option_values else 0
    default_display = f"{default_index + 1} ({default_value})"

    while True:
        try:
            choice = input(f"\n👉 Enter number (1-{len(option_values)}) or press Enter for default [{default_display}]: ")
            if not choice: # User pressed Enter
                print(f"✓ Using default: {default_value}")
                return default_value
            choice_int = int(choice)
            if 1 <= choice_int <= len(option_values):
                selected = option_values[choice_int - 1]
                print(f"✓ Selected: {selected}")
                return selected
            else:
                print("❌ Invalid choice. Please enter a valid number.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def get_user_int(prompt, default_value):
    """Prompts the user to enter an integer >= 1."""
    while True:
        try:
            print(f"\n⚙️ {prompt}")
            full_prompt = f"👉 Enter a value (>=1) or press Enter for default [{default_value}]: "
            choice = input(full_prompt)
            if not choice:
                print(f"✓ Using default: {default_value}")
                return default_value
            choice_int = int(choice)
            if choice_int >= 1:
                print(f"✓ Selected: {choice_int}")
                return choice_int
            else:
                print("❌ Please enter a positive integer (>= 1).")
        except ValueError:
            print("❌ Invalid input. Please enter an integer.")

# --- Image Preprocessing Functions ---
def preprocess_for_analysis(frame):
    """Preprocess frame for analysis (resize, convert color if needed)"""
    # Resize for faster processing
    resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Convert to RGB if needed (DeepFace uses RGB)
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized

# --- Adaptive Analysis Rate ---
def get_adaptive_analyze_rate(base_rate=1):
    """Dynamically adjust analysis rate based on CPU load"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 90:
        return max(10, base_rate * 5)  # Very high load - analyze every 10 frames or 5x base rate
    elif cpu_percent > 70:
        return max(5, base_rate * 3)   # High load - analyze every 5 frames or 3x base rate
    elif cpu_percent > 50:
        return max(3, base_rate * 2)   # Medium load - analyze every 3 frames or 2x base rate
    else:
        return base_rate               # Low load - use base rate

def analyze_live_demographics(camera_index=0, output_width=640, output_height=480, 
                          display_analysis=True, display_fps_stats=True,
                          analyze_every=1, detector_backend='opencv', enforce_detection=False):
    """Captures video and analyzes emotions, engagement, age, and gender.
    
    Args:
        camera_index: Index of the camera to use (default: 0)
        output_width: Width of the output display (default: 640)
        output_height: Height of the output display (default: 480)
        display_analysis: Whether to display analysis results (default: True)
        display_fps_stats: Whether to display FPS statistics (default: True)
        analyze_every: Analyze every N frames (default: 1)
        detector_backend: Face detector backend (default: 'opencv')
        enforce_detection: Whether to enforce face detection (default: False)
    """

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Set up queues for multi-threaded processing
    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    result_queue = queue.Queue()
    
    # Flag to indicate when threads should terminate
    thread_stop_event = threading.Event()
    
    # Analysis worker thread function
    def analysis_worker():
        """Worker function that performs face analysis on frames in the queue"""
        print("Analysis worker started")
        try:
            while not thread_stop_event.is_set():
                try:
                    # Try to get a frame from the queue (non-blocking)
                    if frame_queue.empty():
                        time.sleep(0.01)  # Small sleep to prevent CPU thrashing
                        continue
                        
                    frame_id, frame = frame_queue.get(block=False)
                    
                    if frame is None or frame_id is None:
                        continue
                        
                    # Preprocess the frame for analysis
                    processed_frame = preprocess_for_analysis(frame)
                    
                    try:
                        # Detect faces using face_recognition
                        # This is much more reliable than DeepFace's detector
                        rgb_frame = processed_frame
                        face_locations = face_recognition.face_locations(rgb_frame)
                        
                        if not face_locations:
                            # No faces detected, put empty result to maintain frame sequence
                            result_queue.put((frame_id, []))
                            continue
                            
                        # Analyze faces with DeepFace
                        predictions = []
                        try:
                            predictions = DeepFace.analyze(
                                img_path=rgb_frame,
                                actions=['emotion', 'age', 'gender'],
                                detector_backend=detector_backend,
                                enforce_detection=False,
                                align=True
                            )
                        except Exception as e:
                            print(f"DeepFace analysis error: {e}")
                            # Continue with empty predictions rather than failing
                        
                        # Process predictions into our standard format
                        processed_results = process_predictions(predictions, face_locations)
                        
                        # Put result in the queue
                        result_queue.put((frame_id, processed_results))
                        
                    except Exception as e:
                        print(f"Error in face detection/analysis: {e}")
                        # Put empty result to maintain frame sequence
                        result_queue.put((frame_id, []))
                        
                except queue.Empty:
                    # Queue is empty, just continue the loop
                    continue
                except Exception as e:
                    print(f"Unexpected error in analysis worker: {e}")
                    # Put empty result to maintain frame sequence
                    try:
                        result_queue.put((frame_id, []))
                    except:
                        pass  # If frame_id is not defined, we can't put a result
                    
        except Exception as e:
            print(f"Fatal error in analysis worker: {e}")
        finally:
            print("Analysis worker stopped")
    
    # Start analysis worker thread
    analysis_thread = threading.Thread(target=analysis_worker)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    print(f"Starting live analysis (Emotion, Engagement, Age, Gender)... Analyzing based on adaptive rate (base: every {analyze_every} frame(s)). Press 'q' to quit.")

    prev_frame_time = 0
    new_frame_time = 0
    frame_count = 0
    
    # Initialize variables for face tracking
    trackers = []
    last_detection_frame = 0
    last_analysis_results = []
    need_new_detection = True
    
    # Dictionary to store the latest results for each frame ID
    results_cache = {}
    next_result_frame_id = 0
    
    adaptive_analyze_every = analyze_every
    last_adaptive_update = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        frame_count += 1
        current_frame_id = frame_count

        # Resize for display
        target_width = output_width
        target_height = output_height
        display_frame = cv2.resize(frame.copy(), (target_width, target_height))
        
        # Update adaptive analysis rate every 5 seconds
        current_time = time.time()
        if current_time - last_adaptive_update > 5.0:
            adaptive_analyze_every = get_adaptive_analyze_rate(analyze_every)
            last_adaptive_update = current_time
            
        # Check if we need to do face detection and analysis
        if frame_count % adaptive_analyze_every == 0 or need_new_detection:
            # Preprocess frame for analysis
            analysis_frame = preprocess_for_analysis(frame)
            
            # Reset trackers when we do a new detection
            trackers = []
            
            # Add frame to queue for processing if there's room
            if not frame_queue.full():
                frame_queue.put((current_frame_id, analysis_frame))
                need_new_detection = False
                last_detection_frame = frame_count
        
        # Update face trackers (if any)
        active_trackers = []
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                active_trackers.append((tracker, (x, y, w, h)))
        
        trackers = [t[0] for t in active_trackers]
        
        # Check for new analysis results
        try:
            while True:  # Get all available results
                result_frame_id, results = result_queue.get_nowait()
                results_cache[result_frame_id] = results
                result_queue.task_done()
        except queue.Empty:
            pass
        
        # Process results in order
        while next_result_frame_id in results_cache:
            results = results_cache[next_result_frame_id]
            if results:  # If we have face results
                last_analysis_results = results
                
                # Create new trackers for detected faces
                trackers = []
                for (x, y, w, h, _, _, _, _, _) in results:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    trackers.append(tracker)
            
            # Remove processed result and increment expected ID
            del results_cache[next_result_frame_id]
            next_result_frame_id += 1
        
        # Check if we need a new detection (if tracking is failing or it's been too long)
        if not trackers or (frame_count - last_detection_frame) > TRACKING_MAX_FRAMES:
            need_new_detection = True
        
        # --- Draw results --- 
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        # Add FPS and configuration info
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Analyze every: {adaptive_analyze_every} frames", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Detector: {detector_backend}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw tracked face boxes
        if trackers and not need_new_detection:
            for tracker, (x, y, w, h) in active_trackers:
                # Draw tracking box in blue
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(display_frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw analysis results
        for (x, y, w, h, emotion, engagement_label, engagement_score, age, gender) in last_analysis_results:
            # Draw detection box in green
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Determine text start position - making sure it doesn't go off screen
            frame_height = display_frame.shape[0]
            
            # Decide if text should go above or below the face
            if y > 70:  # Enough space above face
                text_y = y - 15
                direction = -1  # text goes upward from face
            else:
                text_y = y + h + 25
                direction = 1  # text goes downward from face
            
            line_height = 22  # Increased line height for better readability
            
            # Create semi-transparent background for text to improve readability
            def draw_text_with_background(img, text, position, font, scale, color, thickness=1, bg_color=(0, 0, 0), bg_alpha=0.6):
                x, y = position
                text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                text_w, text_h = text_size
                
                # Create overlay for semi-transparent background
                overlay = img.copy()
                cv2.rectangle(overlay, (x-5, y-text_h-5), (x+text_w+5, y+5), bg_color, -1)
                cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
                
                # Draw text
                cv2.putText(img, text, position, font, scale, color, thickness)
            
            # Display Info with backgrounds
            # Emotion
            emotion_text = f"Emotion: {emotion}"
            emotion_pos = (x, text_y)
            draw_text_with_background(display_frame, emotion_text, emotion_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Engagement
            engagement_text = f"Engagement: {engagement_label}"
            if engagement_score >= 0:
                engagement_text += f" ({engagement_score:.2f})"
            engagement_pos = (x, text_y + direction * line_height)
            draw_text_with_background(display_frame, engagement_text, engagement_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            
            # Age
            age_text = f"Age: {age}"
            age_pos = (x, text_y + direction * 2 * line_height)
            draw_text_with_background(display_frame, age_text, age_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Gender
            gender_text = f"Gender: {gender}"
            gender_pos = (x, text_y + direction * 3 * line_height)
            draw_text_with_background(display_frame, gender_text, gender_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow('Live Analysis (Emotion, Engagement, Age, Gender)', display_frame)
        
        # Make window more noticeable
        cv2.setWindowProperty('Live Analysis (Emotion, Engagement, Age, Gender)', 
                              cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    thread_stop_event.set()
    if analysis_thread.is_alive():
        analysis_thread.join(timeout=1.0)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Analysis stopped.")

def process_predictions(predictions, face_locations):
    """
    Process the predictions from DeepFace and face locations
    to extract meaningful data.
    
    Args:
        predictions: The predictions from DeepFace
        face_locations: The face locations from face_recognition
        
    Returns:
        List of tuples containing (x, y, w, h, emotion, engagement_label, 
        engagement_score, age, gender)
    """
    processed_results = []
    
    # Handle case when predictions is empty
    if not predictions:
        return processed_results
        
    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = [predictions]
        
    # Map emotions to engagement scores
    emotion_engagement_map = {
        'happy': 1.0,
        'surprise': 0.8,
        'neutral': 0.5,
        'fear': 0.3,
        'sad': 0.2, 
        'angry': 0.1,
        'disgust': 0.0
    }
    
    # Process each face
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Convert to x, y, w, h format
        x, y = left, top
        w, h = right - left, bottom - top
        
        # Default values if prediction fails
        emotion = "unknown"
        age = "unknown"
        gender = "unknown"
        engagement_score = -1  # Negative means unknown
        engagement_label = "unknown"
        
        # Try to match this face location with a prediction
        if i < len(predictions):
            face_data = predictions[i]
            
            # Extract emotion
            try:
                if isinstance(face_data, dict) and 'emotion' in face_data:
                    emotions = face_data['emotion']
                    # Get the emotion with highest score
                    emotion = max(emotions.items(), key=lambda x: x[1])[0]
                    
                    # Calculate engagement score
                    engagement_score = emotion_engagement_map.get(emotion, 0.0)
                    
                    # Determine engagement label based on score
                    if engagement_score >= 0.8:
                        engagement_label = "High"
                    elif engagement_score >= 0.4:
                        engagement_label = "Medium"
                    else:
                        engagement_label = "Low"
            except Exception as e:
                print(f"Error extracting emotion: {e}")
                
            # Extract age
            try:
                if isinstance(face_data, dict) and 'age' in face_data:
                    age = face_data['age']
                    if isinstance(age, (int, float)):
                        age = int(round(age))
            except Exception as e:
                print(f"Error extracting age: {e}")
                
            # Extract gender
            try:
                if isinstance(face_data, dict) and 'gender' in face_data:
                    gender_data = face_data['gender']
                    if isinstance(gender_data, dict) and 'dominant_gender' in gender_data:
                        gender = gender_data['dominant_gender']
                    elif isinstance(gender_data, str):
                        gender = gender_data
                    else:
                        gender = str(gender_data)
            except Exception as e:
                print(f"Error extracting gender: {e}")
        
        # Add processed result
        processed_results.append((x, y, w, h, emotion, engagement_label, engagement_score, age, gender))
    
    return processed_results

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

    parser = argparse.ArgumentParser(
        description='Real-time face analysis for student engagement tracking.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_demographics.py                           # Interactive mode
  python live_demographics.py --detector mediapipe      # Use MediaPipe detector
  python live_demographics.py --analyze_every 2         # Process every 2nd frame
  python live_demographics.py --detector ssd --analyze_every 3 --enforce  # Combined options
        """
    )
    
    parser.add_argument('--detector', type=str, default=None,
                      choices=available_detectors,
                      help='Face detector backend to use. Each option has different speed/accuracy trade-offs.')
                      
    parser.add_argument('--analyze_every', type=int, default=None,
                      help='Analyze only every Nth frame to improve performance. Higher values increase FPS but reduce analysis frequency.')
                      
    parser.add_argument('--enforce', action='store_true',
                      help='Stop analysis if no face is detected. Useful for attendance verification systems.')

    args = parser.parse_args()

    # --- Interactive Selection if Arguments Not Provided ---
    selected_detector = args.detector
    if selected_detector is None:
        print("\n=== Configuration Selection ===")
        # Pass the list with hints to the updated get_user_choice function
        selected_detector = get_user_choice("Select Face Detector Backend:", available_detectors_with_hints, default_detector)

    selected_analyze_every = args.analyze_every
    if selected_analyze_every is None:
        if args.detector is None:
             print("\n---")
        selected_analyze_every = get_user_int("Analyze every N frames (higher = faster but less frequent updates):", default_analyze_every)

    print(f"\n🚀 Starting analysis with:")
    print(f"  • Detector: '{selected_detector}'")
    print(f"  • Analysis frequency: every {selected_analyze_every} frame(s)")
    print(f"  • Enforcing face detection: {'Yes' if args.enforce else 'No'}")
    print("\n📝 Press 'q' in the video window to exit")
    
    analyze_live_demographics(camera_index=0, output_width=640, output_height=480, 
                          display_analysis=True, display_fps_stats=True,
                          analyze_every=selected_analyze_every, 
                          detector_backend=selected_detector,
                          enforce_detection=args.enforce) 