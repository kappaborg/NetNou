import cv2
from deepface import DeepFace
import time
import numpy as np
import face_recognition

def process_predictions(predictions, face_locations):
    """
    Process the predictions from DeepFace and face locations
    to extract meaningful data.
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

def simple_live_demo():
    """A simplified version of live demographics analysis."""
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    
    print("Webcam opened. Starting analysis...")
    print("Press 'q' to quit.")
    
    # Set up for FPS calculation
    prev_frame_time = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if face_locations:
            try:
                # Analyze with DeepFace
                predictions = DeepFace.analyze(
                    img_path=rgb_small_frame,
                    actions=['emotion', 'age', 'gender'],
                    detector_backend='opencv',
                    enforce_detection=False,
                    align=True
                )
                
                # Process predictions
                results = process_predictions(predictions, face_locations)
                
                # Draw results
                for (x, y, w, h, emotion, engagement_label, engagement_score, age, gender) in results:
                    # Scale coordinates back to original size
                    x, y, w, h = x*2, y*2, w*2, h*2
                    
                    # Draw face box
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Decide text positioning
                    if y > 70:  # Space above
                        text_y = y - 15
                        direction = -1  # text goes up
                    else:
                        text_y = y + h + 25
                        direction = 1  # text goes down
                    
                    line_height = 25
                    
                    # Helper function for text backgrounds
                    def add_text_with_background(img, text, pos, font_scale=0.7, color=(255, 255, 255), thickness=2):
                        x, y = pos
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        text_w, text_h = text_size
                        
                        # Background
                        cv2.rectangle(img, (x-5, y-text_h-5), (x+text_w+5, y+5), (0, 0, 0), -1)
                        
                        # Text
                        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                    
                    # Add all text labels
                    add_text_with_background(display_frame, f"Emotion: {emotion}", 
                                           (x, text_y), color=(0, 255, 0))
                    add_text_with_background(display_frame, f"Engagement: {engagement_label}", 
                                           (x, text_y + direction * line_height), color=(255, 150, 0))
                    add_text_with_background(display_frame, f"Age: {age}", 
                                           (x, text_y + direction * 2 * line_height), color=(0, 255, 255))
                    add_text_with_background(display_frame, f"Gender: {gender}", 
                                           (x, text_y + direction * 3 * line_height), color=(255, 0, 255))
            
            except Exception as e:
                print(f"Analysis error: {e}")
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        # Add FPS to display
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Simple Live Demographics", display_frame)
        
        # Make window stay on top
        cv2.setWindowProperty("Simple Live Demographics", cv2.WND_PROP_TOPMOST, 1)
        
        # Check for quit
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")

if __name__ == "__main__":
    simple_live_demo() 