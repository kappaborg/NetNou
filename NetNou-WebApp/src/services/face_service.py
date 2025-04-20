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
import io
from PIL import Image
from datetime import datetime
import json
import uuid

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

# Create directory for storing face embeddings if it doesn't exist
FACE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'faces')
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# In-memory face database
FACE_EMBEDDINGS = []

# Load existing face data at startup
def load_face_data():
    """Load existing face data from file storage."""
    global FACE_EMBEDDINGS
    try:
        face_data_file = os.path.join(FACE_DATA_DIR, 'face_embeddings.json')
        if os.path.exists(face_data_file):
            with open(face_data_file, 'r') as f:
                FACE_EMBEDDINGS = json.load(f)
            print(f"Loaded {len(FACE_EMBEDDINGS)} face embeddings from storage")
    except Exception as e:
        print(f"Error loading face data: {str(e)}")
        FACE_EMBEDDINGS = []

# Initialize by loading existing face data
load_face_data()

def save_face_data():
    """Save face embeddings to persistent storage."""
    try:
        face_data_file = os.path.join(FACE_DATA_DIR, 'face_embeddings.json')
        with open(face_data_file, 'w') as f:
            json.dump(FACE_EMBEDDINGS, f)
        print(f"Saved {len(FACE_EMBEDDINGS)} face embeddings to storage")
        return True
    except Exception as e:
        print(f"Error saving face data: {str(e)}")
        return False

def decode_image(base64_image):
    """Decode base64 image to OpenCV format.
    
    Args:
        base64_image (str): Base64 encoded image
        
    Returns:
        numpy.ndarray: OpenCV image or None if decoding fails
    """
    try:
        # Handle data URL format (e.g. "data:image/jpeg;base64,...")
        if isinstance(base64_image, str) and 'base64,' in base64_image:
            # Extract only the base64 part
            base64_image = base64_image.split('base64,')[1]
        
        # Handle potential padding issues
        padding = 4 - (len(base64_image) % 4) if len(base64_image) % 4 != 0 else 0
        if padding:
            base64_image += '=' * padding
            
        # Decode base64 to image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        return None

def analyze_face(image_data):
    """Analyze face in an image.
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        dict: Face analysis results
    """
    try:
        # Decode image
        image = decode_image(image_data)
        if image is None:
            return {'success': False, 'message': 'Invalid image data'}
        
        # Convert image to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {'success': False, 'message': 'No face detected in image'}
        
        if len(face_locations) > 1:
            return {'success': False, 'message': 'Multiple faces detected. Please provide an image with one face only.'}
        
        # Generate face encoding
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        return {
            'success': True,
            'face_detected': True,
            'face_location': face_locations[0],
            'face_encoding': face_encoding.tolist()
        }
    except Exception as e:
        print(f"Error analyzing face: {str(e)}")
        return {'success': False, 'message': f'Error analyzing face: {str(e)}'}

def register_face(student_id, image_data):
    """Register a student's face in the system.
    
    Args:
        student_id (str): The ID of the student
        image_data (str): Base64 encoded image data
        
    Returns:
        dict: Result of registration with success status and message
    """
    if not student_id:
        return {'success': False, 'message': 'Missing student ID'}
        
    if not image_data:
        return {'success': False, 'message': 'Missing image data'}
    
    if not isinstance(image_data, str):
        return {'success': False, 'message': 'Invalid image data format'}
        
    print(f"Processing face registration for student ID: {student_id}")
    print(f"Image data length: {len(image_data) if image_data else 0}")
    
    # Check if student already has a face registered
    for embedding in FACE_EMBEDDINGS:
        if embedding['student_id'] == student_id:
            # Update the existing face embedding instead of returning an error
            print(f"Updating existing face for student {student_id}")
            result = analyze_face(image_data)
            if not result['success']:
                print(f"Face analysis failed: {result.get('message', 'Unknown error')}")
                return result
                
            embedding['face_encoding'] = result['face_encoding']
            embedding['updated_at'] = datetime.now().isoformat()
            save_face_data()
            
            # Update student's face registration status in database
            update_student_face(student_id, True)
            return {'success': True, 'message': 'Face updated successfully'}
    
    try:
        # Analyze face and get encoding
        print(f"Analyzing face for new registration")
        result = analyze_face(image_data)
        if not result['success']:
            print(f"Face analysis failed: {result.get('message', 'Unknown error')}")
            return result
        
        # Create a new face embedding record
        embedding_record = {
            'id': str(uuid.uuid4()),
            'student_id': student_id,
            'face_encoding': result['face_encoding'],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Add to database
        FACE_EMBEDDINGS.append(embedding_record)
        
        # Save to persistent storage
        save_success = save_face_data()
        if not save_success:
            print("Warning: Face data saved to memory but failed to persist to storage")
        
        # Update student's face registration status in database
        update_student_face(student_id, True)
        
        print(f"Face registered successfully for student {student_id}")
        return {'success': True, 'message': 'Face registered successfully'}
    
    except Exception as e:
        error_message = str(e)
        print(f"Error registering face: {error_message}")
        
        # Provide more user-friendly error messages for common issues
        if "base64" in error_message.lower():
            return {'success': False, 'message': 'Invalid image format. Please ensure the image is properly encoded.'}
        elif "io" in error_message.lower() or "bytes" in error_message.lower():
            return {'success': False, 'message': 'Invalid image data. Please capture a new image.'}
        else:
            return {'success': False, 'message': f'Error registering face: {error_message}'}

def identify_face(image_data, tolerance=0.6):
    """Identify a face from an image.
    
    Args:
        image_data (str): Base64 encoded image data
        tolerance (float): Matching tolerance (lower is stricter)
        
    Returns:
        dict: Identification result with student_id if found
    """
    try:
        # Decode image
        image = decode_image(image_data)
        if image is None:
            return {'success': False, 'message': 'Invalid image data'}
        
        # Convert image to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {'success': False, 'message': 'No face detected in image'}
        
        if len(face_locations) > 1:
            return {'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in the frame.'}
        
        # Generate face encoding
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # No faces registered yet
        if not FACE_EMBEDDINGS:
            return {'success': False, 'message': 'No faces registered in the system. Please register students first.'}
        
        # Compare with stored face encodings
        known_encodings = [np.array(embedding['face_encoding']) for embedding in FACE_EMBEDDINGS]
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        
        # If we have a match
        if True in matches:
            match_index = matches.index(True)
            matched_student_id = FACE_EMBEDDINGS[match_index]['student_id']
            
            # Calculate confidence (1 - distance)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            confidence = 1 - float(face_distances[match_index])
            confidence_percentage = f"{confidence * 100:.2f}%"
            
            return {
                'success': True,
                'identified': True,
                'student_id': matched_student_id,
                'confidence': confidence_percentage
            }
        else:
            return {
                'success': True,
                'identified': False,
                'message': 'Face not recognized. Please register this student.'
            }
    
    except Exception as e:
        print(f"Error identifying face: {str(e)}")
        return {'success': False, 'message': f'Error identifying face: {str(e)}'}

def batch_identify_faces(image_data, tolerance=0.6):
    """Identify multiple faces in a single image.
    
    Args:
        image_data (str): Base64 encoded image data
        tolerance (float): Matching tolerance (lower is stricter)
        
    Returns:
        dict: Identification results with student IDs if found
    """
    try:
        # Decode image
        image = decode_image(image_data)
        if image is None:
            return {'success': False, 'message': 'Invalid image data'}
        
        # Convert image to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {'success': False, 'message': 'No faces detected in image'}
        
        # No faces registered yet
        if not FACE_EMBEDDINGS:
            return {'success': False, 'message': 'No faces registered in the system. Please register students first.'}
        
        # Extract known face encodings
        known_encodings = [np.array(embedding['face_encoding']) for embedding in FACE_EMBEDDINGS]
        known_student_ids = [embedding['student_id'] for embedding in FACE_EMBEDDINGS]
        
        # Process each detected face
        results = []
        for i, face_location in enumerate(face_locations):
            # Generate face encoding for this face
            face_encoding = face_recognition.face_encodings(rgb_image, [face_location])[0]
            
            # Compare with stored face encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
            
            # Find the closest match
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # If we have a match
            if matches[best_match_index]:
                matched_student_id = known_student_ids[best_match_index]
                confidence = 1 - float(face_distances[best_match_index])
                confidence_percentage = f"{confidence * 100:.2f}%"
                
                results.append({
                    'face_index': i,
                    'face_location': face_location,
                    'identified': True,
                    'student_id': matched_student_id,
                    'confidence': confidence_percentage
                })
            else:
                results.append({
                    'face_index': i,
                    'face_location': face_location,
                    'identified': False,
                    'message': 'Face not recognized'
                })
        
        return {
            'success': True,
            'face_count': len(face_locations),
            'results': results
        }
    
    except Exception as e:
        print(f"Error in batch face identification: {str(e)}")
        return {'success': False, 'message': f'Error identifying faces: {str(e)}'}

def face_recognition_attendance(class_id, image_data, tolerance=0.6):
    """Take attendance using face recognition for an entire class at once.
    
    Args:
        class_id (str): The class ID
        image_data (str): Base64 encoded image data
        tolerance (float): Matching tolerance (lower is stricter)
        
    Returns:
        dict: Attendance results with identified students
    """
    # Identify all faces in the image
    identification_results = batch_identify_faces(image_data, tolerance)
    
    if not identification_results['success']:
        return identification_results
    
    # Extract student IDs of recognized faces
    recognized_students = []
    unrecognized_faces = 0
    
    for result in identification_results['results']:
        if result['identified']:
            recognized_students.append({
                'student_id': result['student_id'],
                'confidence': result['confidence']
            })
        else:
            unrecognized_faces += 1
    
    return {
        'success': True,
        'class_id': class_id,
        'timestamp': datetime.now().isoformat(),
        'recognized_count': len(recognized_students),
        'unrecognized_count': unrecognized_faces,
        'recognized_students': recognized_students
    }

def delete_face_registration(student_id):
    """Delete a student's face registration.
    
    Args:
        student_id (str): The ID of the student
        
    Returns:
        dict: Result of deletion with success status and message
    """
    global FACE_EMBEDDINGS
    
    # Check if student has a face registered
    for i, embedding in enumerate(FACE_EMBEDDINGS):
        if embedding['student_id'] == student_id:
            # Remove from the list
            FACE_EMBEDDINGS.pop(i)
            
            # Save changes to persistent storage
            save_face_data()
            
            # Update student's face registration status in database
            update_student_face(student_id, False)
            
            return {'success': True, 'message': 'Face registration deleted successfully'}
    
    return {'success': False, 'message': 'No face registration found for this student'}

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