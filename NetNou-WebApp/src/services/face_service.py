"""Face recognition service for the application."""

import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import base64
import os
from ..config import Config
from ..database.student_model import get_student, update_student_face
from ..database.face_model import save_face_embedding, get_all_face_embeddings
from ..core.nn.simple_nn import SimpleNN

# Load engagement model
def load_engagement_model():
    """Load the engagement neural network model."""
    try:
        model = SimpleNN(input_size=1, hidden_size=4, output_size=1,
                         hidden_activation='relu', output_activation='sigmoid', loss='bce')
        model.load_weights(Config.NN_WEIGHTS_PATH)
        return model
    except Exception as e:
        print(f"Warning: Failed to load engagement model: {e}")
        return None

# Map emotions to engagement scores
EMOTION_ENGAGEMENT_MAP = {
    'happy': 0.9,
    'surprise': 0.8,
    'neutral': 0.5,
    'fear': 0.3,
    'sad': 0.2,
    'angry': 0.1,
    'disgust': 0.0
}

# Initialize the engagement model
engagement_model = load_engagement_model()

def decode_image(image_data):
    """Decode base64 image data to OpenCV format.
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        numpy.ndarray: Image in OpenCV format
    """
    # Check if the image is a base64 string
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Extract the base64 part
        image_data = image_data.split(',')[1]
    
    # Decode the base64 string
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    
    # Decode image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    return image

def register_face(student_id, image_data):
    """Register a student's face in the system.
    
    Args:
        student_id (str): The ID of the student
        image_data (str): Base64 encoded image data
        
    Returns:
        dict: Result of registration with success status and message
    """
    # Check if student exists
    student = get_student(student_id)
    if not student:
        return {'success': False, 'message': 'Student not found'}
    
    try:
        # Decode image
        image = decode_image(image_data)
        if image is None:
            return {'success': False, 'message': 'Invalid image data'}
        
        # Convert image to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {'success': False, 'message': 'No face detected in image'}
        
        if len(face_locations) > 1:
            return {'success': False, 'message': 'Multiple faces detected. Please provide an image with one face only.'}
        
        # Generate face embedding
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # Save face embedding to database
        save_face_embedding(student_id, face_encoding.tolist())
        
        # Update student record
        update_student_face(student_id, True)
        
        return {'success': True, 'message': 'Face registered successfully'}
    
    except Exception as e:
        return {'success': False, 'message': f'Error registering face: {str(e)}'}

def identify_face(image_data):
    """Identify a face in the given image.
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        dict: Result of identification with success status, message, and student info if found
    """
    try:
        # Decode image
        image = decode_image(image_data)
        if image is None:
            return {'success': False, 'message': 'Invalid image data'}
        
        # Convert image to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {'success': False, 'message': 'No face detected in image'}
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Get all registered face embeddings
        registered_faces = get_all_face_embeddings()
        
        matches = []
        
        # Check each detected face against registered faces
        for face_encoding in face_encodings:
            match_found = False
            
            for registered_face in registered_faces:
                registered_encoding = np.array(registered_face['embedding'])
                
                # Compare faces
                distance = face_recognition.face_distance([registered_encoding], face_encoding)[0]
                
                # If match is found (distance less than threshold)
                if distance < 0.6:  # Adjust threshold as needed
                    student_id = registered_face['student_id']
                    student = get_student(student_id)
                    
                    if student:
                        # Analyze face for demographics and emotion
                        demographics = analyze_face(rgb_image, face_locations[0])
                        
                        matches.append({
                            'student_id': student_id,
                            'name': f"{student['first_name']} {student['last_name']}",
                            'match_confidence': float(1 - distance),
                            'demographics': demographics
                        })
                        match_found = True
                        break
            
            if not match_found:
                # If no match found, still analyze the face
                demographics = analyze_face(rgb_image, face_locations[0])
                matches.append({
                    'student_id': None,
                    'name': 'Unknown',
                    'match_confidence': 0.0,
                    'demographics': demographics
                })
        
        return {
            'success': True,
            'message': 'Face identification completed',
            'matches': matches
        }
    
    except Exception as e:
        return {'success': False, 'message': f'Error identifying face: {str(e)}'}

def analyze_face(image, face_location=None):
    """Analyze a face for demographics and emotion.
    
    Args:
        image (numpy.ndarray): Image containing a face
        face_location (tuple, optional): Location of the face in the image (top, right, bottom, left)
        
    Returns:
        dict: Demographics and emotion analysis results
    """
    try:
        # If face location is provided, crop the image to the face
        if face_location:
            top, right, bottom, left = face_location
            image = image[top:bottom, left:right]
        
        # Analyze with DeepFace
        analysis = DeepFace.analyze(
            img_path=image,
            actions=Config.FACE_ANALYZE_ACTIONS,
            detector_backend=Config.FACE_DETECTION_BACKEND,
            enforce_detection=False,
            align=True,
            silent=True
        )
        
        # Process results
        if isinstance(analysis, list):
            analysis = analysis[0]
            
        # Extract data
        result = {}
        
        # Get emotion
        if 'emotion' in analysis:
            emotions = analysis['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            result['emotion'] = dominant_emotion
            
            # Calculate engagement score
            engagement_score = EMOTION_ENGAGEMENT_MAP.get(dominant_emotion.lower(), 0.5)
            
            # Use neural network if available
            if engagement_model:
                nn_input = np.array([[engagement_score]])
                predicted_score = engagement_model.predict(nn_input)[0][0]
                engagement_score = float(predicted_score)
            
            # Get engagement label
            if engagement_score >= 0.8:
                engagement_label = "High"
            elif engagement_score >= 0.4:
                engagement_label = "Medium"
            else:
                engagement_label = "Low"
                
            result['engagement'] = {
                'score': engagement_score,
                'label': engagement_label
            }
        
        # Get age
        if 'age' in analysis:
            result['age'] = int(analysis['age'])
        
        # Get gender
        if 'gender' in analysis:
            gender_data = analysis['gender']
            if isinstance(gender_data, dict):
                result['gender'] = gender_data.get('dominant_gender', 'unknown')
            else:
                result['gender'] = str(gender_data)
        
        return result
    
    except Exception as e:
        print(f"Error analyzing face: {str(e)}")
        return {'error': str(e)}

def analyze_deepface_batch(frames, backend="opencv", actions=None):
    """Analyze a batch of frames using DeepFace for demographics.
    
    Args:
        frames (list): List of images to analyze
        backend (str): Face detection backend to use
        actions (list): DeepFace actions to perform
        
    Returns:
        list: Analysis results for each frame
    """
    if actions is None:
        actions = Config.FACE_ANALYZE_ACTIONS
        
    results = []
    
    for frame in frames:
        try:
            # Convert to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != np.uint8:
                    # Normalize and convert to uint8 if needed
                    frame = (frame * 255).astype(np.uint8)
                
                # Analyze with DeepFace
                analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=actions,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                    silent=True
                )
                
                results.append(analysis)
            else:
                results.append({'error': 'Invalid image format'})
        except Exception as e:
            results.append({'error': str(e)})
    
    return results 