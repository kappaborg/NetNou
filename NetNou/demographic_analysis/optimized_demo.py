import cv2
from deepface import DeepFace
import time
import numpy as np
import face_recognition
import sys
import os

# Scratch NN modelini import edelim (engagement için)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scratch_nn'))
try:
    from simple_nn import SimpleNN
except ImportError:
    print("Error: Could not import SimpleNN. Ensure simple_nn.py is in NetNou/scratch_nn")
    engagement_model = None
else:
    # Engagement modelini yükleyelim
    ENGAGEMENT_NN_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'scratch_nn', 'engagement_nn_weights.npz')
    try:
        engagement_model = SimpleNN(input_size=1, hidden_size=4, output_size=1, 
                                    hidden_activation='relu', output_activation='sigmoid', loss='bce')
        engagement_model.load_weights(ENGAGEMENT_NN_WEIGHTS_PATH)
        print(f"Engagement model loaded from {ENGAGEMENT_NN_WEIGHTS_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load engagement model: {e}")
        engagement_model = None

# --- Configuration ---
# Analyze only every N frames for better performance
ANALYZE_EVERY = 20
# Maximum frames to track before re-detecting
MAX_TRACKING_FRAMES = 60
# Size reduction for analysis (smaller = faster, but less accurate)
RESIZE_FACTOR = 0.3
# Choose a fast detector backend: opencv, ssd, or mtcnn
DETECTOR = 'opencv'
# Analyze these features
ANALYZE_ACTIONS = ['emotion', 'age', 'gender']  # Age ekledik
# Emotion -> engagement mapping
EMOTION_ENGAGEMENT_MAP = {
    'happy': 0.9,
    'surprise': 0.8,
    'neutral': 0.5,
    'fear': 0.3,
    'sad': 0.2, 
    'angry': 0.1,
    'disgust': 0.0
}

# --- Runtime dynamic configuration ---
config = {
    'analyze_every': ANALYZE_EVERY,
    'resize_factor': RESIZE_FACTOR,
    'show_fps': True,
    'show_labels': True,
    'help_visible': False,
    'show_debug': False,  # Detaylı hata ayıklama bilgileri
}

# Keyboard controls help text
HELP_TEXT = """
KEYBOARD CONTROLS:
+/- : Increase/decrease analysis frequency
f   : Toggle FPS display
l   : Toggle labels display
d   : Toggle debug info
a   : Toggle advanced display
r   : Reset to default settings
h   : Toggle this help display
q   : Quit
"""

# --- Helper Functions ---
def create_tracker():
    """Create a tracker compatible with the installed OpenCV version"""
    # Try different tracker types based on OpenCV version
    tracker_types = ['CSRT', 'KCF', 'MOSSE']
    
    for tracker_type in tracker_types:
        try:
            if tracker_type == 'CSRT':
                return cv2.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                return cv2.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                return cv2.legacy.TrackerMOSSE_create()
        except:
            continue
            
    # If no tracker can be created, use a simple tracker implementation
    print("Warning: No built-in tracker available. Using simple manual tracking.")
    return None

class SimpleTracker:
    """A simple manual tracker for when OpenCV trackers aren't available"""
    def __init__(self, bbox):
        self.bbox = bbox
        
    def update(self, frame):
        """Very simple tracking - just return the original position"""
        return True, self.bbox

def get_engagement_score(emotion):
    """Convert emotion to engagement score"""
    return EMOTION_ENGAGEMENT_MAP.get(emotion, 0.5)

def get_engagement_label(score):
    """Convert engagement score to descriptive label"""
    if score >= 0.8:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"

def process_predictions(predictions, face_locations):
    """Process the predictions from DeepFace and face locations"""
    processed_results = []
    
    # Handle case when predictions is empty
    if not predictions:
        return processed_results
        
    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Process each face
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Convert to x, y, w, h format
        x, y = left, top
        w, h = right - left, bottom - top
        
        # Default values
        emotion = "unknown"
        age = "unknown"
        gender = "unknown"
        engagement_score = 0.5  # Default mid value
        
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
                    engagement_score = get_engagement_score(emotion)
            except Exception as e:
                if config['show_debug']:
                    print(f"Error extracting emotion: {e}")
            
            # Extract age
            try:
                if isinstance(face_data, dict) and 'age' in face_data:
                    age = face_data['age']
                    if isinstance(age, (int, float)):
                        age = int(round(age))
            except Exception as e:
                if config['show_debug']:
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
                if config['show_debug']:
                    print(f"Error extracting gender: {e}")
        
        # Get engagement label
        engagement_label = get_engagement_label(engagement_score)
        
        # Add processed result including age and engagement
        processed_results.append((x, y, w, h, emotion, engagement_score, engagement_label, age, gender))
    
    return processed_results

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                             font_scale=0.6, color=(255, 255, 255), thickness=2,
                             bg_color=(0, 0, 0), bg_alpha=0.7, padding=5):
    """Draw text with a semi-transparent background for better readability"""
    x, y = position
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Create overlay for background
    overlay = img.copy()
    cv2.rectangle(overlay, 
                 (x - padding, y - text_h - padding), 
                 (x + text_w + padding, y + padding), 
                 bg_color, -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, position, font, font_scale, color, thickness)

def optimized_live_demo():
    """Optimized version of demographics analysis with face tracking."""
    global config
    
    # Webcam erişim denemesi
    print("Trying to access webcam...")
    
    # Farklı kamera indekslerini dene
    available_camera = None
    for camera_index in range(3):  # 0, 1, 2 indekslerini dene
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    available_camera = camera_index
                    print(f"Successfully connected to camera index {camera_index}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"Error accessing camera {camera_index}: {e}")
    
    # Eğer kamera bulunamadıysa
    if available_camera is None:
        print("Error: Could not access any webcam. Please check camera permissions.")
        print("On MacOS, go to System Preferences -> Security & Privacy -> Camera")
        return
    
    # Use the found camera
    cap = cv2.VideoCapture(available_camera)
    
    # Try setting camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Webcam opened. Starting analysis (every {config['analyze_every']} frames)...")
    print("Press 'h' for help, 'q' to quit.")
    
    # FPS tracking
    prev_frame_time = 0
    fps_values = []
    
    # Frame tracking
    frame_count = 0
    last_analysis_frame = 0
    
    # Face tracking
    trackers = []
    last_results = []
    need_detection = True
    
    while True:
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Update face trackers
        if trackers and not need_detection:
            active_trackers = []
            for i, tracker in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # Store the updated tracker and results
                    active_trackers.append((tracker, (x, y, w, h), last_results[i][4:]))
            
            # Update the trackers list with only successful ones
            trackers = [t[0] for t in active_trackers]
            
            # Draw tracking results
            if config['show_labels']:
                for _, (x, y, w, h), result_data in active_trackers:
                    # Extract emotion, engagement, age and gender
                    if len(result_data) >= 5:  # Tüm değerler varsa
                        emotion, engagement_score, engagement_label, age, gender = result_data
                    else:  # Eksik değerler olması durumunda
                        emotion = result_data[0] if len(result_data) > 0 else "unknown"
                        engagement_label = result_data[2] if len(result_data) > 2 else "Medium"
                        age = result_data[3] if len(result_data) > 3 else "unknown"
                        gender = result_data[4] if len(result_data) > 4 else "unknown"
                    
                    # Draw tracking box in blue
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 120, 255), 2)
                    
                    # Decide text positioning - above or below the face
                    if y > 70:  # Enough space above face
                        direction = -1  # text goes upward from face
                        base_y = y - 15
                    else:
                        direction = 1  # text goes downward from face
                        base_y = y + h + 25
                    
                    line_height = 25
                    
                    # Draw labels with backgrounds
                    draw_text_with_background(
                        display_frame, 
                        f"Tracking: {emotion}", 
                        (x, base_y), 
                        color=(0, 120, 255)
                    )
                    
                    draw_text_with_background(
                        display_frame, 
                        f"Engagement: {engagement_label}", 
                        (x, base_y + direction * line_height), 
                        color=(0, 120, 255)
                    )
                    
                    draw_text_with_background(
                        display_frame, 
                        f"Age: {age}", 
                        (x, base_y + direction * 2 * line_height), 
                        color=(0, 120, 255)
                    )
                    
                    draw_text_with_background(
                        display_frame, 
                        f"Gender: {gender}", 
                        (x, base_y + direction * 3 * line_height), 
                        color=(0, 120, 255)
                    )
        
        # Do full detection+analysis only periodically or when tracking fails
        if frame_count % config['analyze_every'] == 0 or need_detection or not trackers or (frame_count - last_analysis_frame) > MAX_TRACKING_FRAMES:
            # Clear existing trackers
            trackers = []
            
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=config['resize_factor'], fy=config['resize_factor'])
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if face_locations:
                try:
                    # Analyze with DeepFace
                    predictions = DeepFace.analyze(
                        img_path=rgb_small_frame,
                        actions=ANALYZE_ACTIONS,
                        detector_backend=DETECTOR,
                        enforce_detection=False,
                        align=True,
                        silent=True  # Suppress progress bars
                    )
                    
                    # Process predictions
                    results = process_predictions(predictions, face_locations)
                    last_results = []
                    
                    # Create trackers for each face
                    for face_data in results:
                        # Unpack face data
                        x, y, w, h, emotion, engagement_score, engagement_label, age, gender = face_data
                        
                        # Scale coordinates back to original size
                        scale = 1.0 / config['resize_factor']
                        x_full, y_full = int(x * scale), int(y * scale)
                        w_full, h_full = int(w * scale), int(h * scale)
                        
                        # Create and initialize tracker
                        tracker = create_tracker()
                        if tracker is None:
                            # Use simple manual tracker as fallback
                            tracker = SimpleTracker((x_full, y_full, w_full, h_full))
                        else:
                            tracker.init(frame, (x_full, y_full, w_full, h_full))
                        trackers.append(tracker)
                        
                        # Save result with scaled coordinates
                        last_results.append((
                            x_full, y_full, w_full, h_full, 
                            emotion, engagement_score, engagement_label, age, gender
                        ))
                        
                        # Draw detection box in green
                        cv2.rectangle(display_frame, (x_full, y_full), (x_full+w_full, y_full+h_full), 
                                     (0, 255, 0), 2)
                        
                        if config['show_labels']:
                            # Decide text positioning - above or below the face
                            if y_full > 70:  # Enough space above face
                                direction = -1  # text goes upward from face
                                base_y = y_full - 15
                            else:
                                direction = 1  # text goes downward from face
                                base_y = y_full + h_full + 25
                            
                            line_height = 25
                            
                            # Draw labels with better backgrounds
                            draw_text_with_background(
                                display_frame, 
                                f"Emotion: {emotion}", 
                                (x_full, base_y), 
                                color=(0, 255, 0)
                            )
                            
                            draw_text_with_background(
                                display_frame, 
                                f"Engagement: {engagement_label}", 
                                (x_full, base_y + direction * line_height), 
                                color=(255, 165, 0)
                            )
                            
                            draw_text_with_background(
                                display_frame, 
                                f"Age: {age}", 
                                (x_full, base_y + direction * 2 * line_height), 
                                color=(0, 255, 255)
                            )
                            
                            draw_text_with_background(
                                display_frame, 
                                f"Gender: {gender}", 
                                (x_full, base_y + direction * 3 * line_height), 
                                color=(255, 0, 255)
                            )
                
                except Exception as e:
                    if config['show_debug']:
                        print(f"Analysis error: {e}")
            
            need_detection = False
            last_analysis_frame = frame_count
        
        # Calculate FPS
        new_frame_time = time.time()
        frame_time = new_frame_time - prev_frame_time if prev_frame_time > 0 else 1/30
        current_fps = 1 / frame_time
        fps_values.append(current_fps)
        
        # Show smoothed FPS (average of last 10 frames)
        if len(fps_values) > 10:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        
        # Show FPS and analysis info if enabled
        if config['show_fps']:
            info_bg = display_frame.copy()
            cv2.rectangle(info_bg, (5, 5), (250, 140), (0, 0, 0), -1)
            cv2.addWeighted(info_bg, 0.6, display_frame, 0.4, 0, display_frame)
            
            cv2.putText(display_frame, f"FPS: {int(avg_fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Analysis: Every {config['analyze_every']} frames", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, f"Detector: {DETECTOR}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_frame, f"Resize: {int(config['resize_factor']*100)}%", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show help if requested
        if config['help_visible']:
            # Draw semi-transparent background for help text
            help_overlay = display_frame.copy()
            overlay_color = (0, 0, 0)
            cv2.rectangle(help_overlay, (0, 150), (400, 400), overlay_color, -1)
            alpha = 0.7
            cv2.addWeighted(help_overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
            
            # Add help text line by line
            y_pos = 180
            for line in HELP_TEXT.strip().split('\n'):
                cv2.putText(display_frame, line, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
        
        # Display frame
        window_title = "Live Demographics Analysis (Optimized)"
        cv2.imshow(window_title, display_frame)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            config['analyze_every'] = max(1, config['analyze_every'] - 5)
            print(f"Analyzing every {config['analyze_every']} frames")
        elif key == ord('-'):
            config['analyze_every'] += 5
            print(f"Analyzing every {config['analyze_every']} frames")
        elif key == ord('f'):
            config['show_fps'] = not config['show_fps']
            print(f"FPS display: {'On' if config['show_fps'] else 'Off'}")
        elif key == ord('l'):
            config['show_labels'] = not config['show_labels']
            print(f"Labels display: {'On' if config['show_labels'] else 'Off'}")
        elif key == ord('d'):
            config['show_debug'] = not config['show_debug']
            print(f"Debug info: {'On' if config['show_debug'] else 'Off'}")
        elif key == ord('r'):
            config['analyze_every'] = ANALYZE_EVERY
            config['resize_factor'] = RESIZE_FACTOR
            config['show_fps'] = True
            config['show_labels'] = True
            config['show_debug'] = False
            print("Settings reset to defaults")
        elif key == ord('h'):
            config['help_visible'] = not config['help_visible']
        
        prev_frame_time = new_frame_time
        
        # Calculate if we need a new detection
        if not trackers:
            need_detection = True
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")

if __name__ == "__main__":
    optimized_live_demo() 