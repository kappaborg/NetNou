"""Face recognition data model."""

from datetime import datetime

# In-memory face database
FACE_EMBEDDINGS = []

def save_face_embedding(student_id, embedding):
    """Save a face embedding for a student.
    
    Args:
        student_id (str): The student ID
        embedding (list): Face embedding as a list of floats
        
    Returns:
        dict: The saved face embedding
    """
    # Check if student already has a face embedding
    for face in FACE_EMBEDDINGS:
        if face['student_id'] == student_id:
            # Update existing embedding
            face['embedding'] = embedding
            face['updated_at'] = datetime.now().isoformat()
            return face
    
    # Create new face embedding
    face_data = {
        'student_id': student_id,
        'embedding': embedding,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    FACE_EMBEDDINGS.append(face_data)
    return face_data

def get_face_embedding(student_id):
    """Get a face embedding by student ID.
    
    Args:
        student_id (str): The student ID
        
    Returns:
        dict: Face embedding if found, None otherwise
    """
    for face in FACE_EMBEDDINGS:
        if face['student_id'] == student_id:
            return face
    return None

def get_all_face_embeddings():
    """Get all face embeddings.
    
    Returns:
        list: All face embeddings
    """
    return FACE_EMBEDDINGS

def delete_face_embedding(student_id):
    """Delete a face embedding.
    
    Args:
        student_id (str): The student ID
        
    Returns:
        bool: True if deleted, False if not found
    """
    for i, face in enumerate(FACE_EMBEDDINGS):
        if face['student_id'] == student_id:
            del FACE_EMBEDDINGS[i]
            return True
    return False 