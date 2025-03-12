import numpy as np
import cv2
import base64
from typing import Optional, Tuple, List, Dict
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# Initialize the models
detector = MTCNN()
facenet = FaceNet()

class ImageQuality:
    MIN_FACE_SIZE = 100  # Minimum face size in pixels
    MIN_BRIGHTNESS = 0.4  # Minimum brightness (0-1)
    MAX_BRIGHTNESS = 0.9  # Maximum brightness (0-1)
    MIN_SHARPNESS = 50   # Minimum sharpness score
    MIN_CONFIDENCE = 0.9  # Minimum face detection confidence

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image for FaceNet model"""
    # Convert BGR to RGB
    if len(img.shape) == 4:
        img = img[0]
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    return img

def check_image_quality(img: np.ndarray, face_box: Dict) -> Tuple[bool, Dict[str, float]]:
    """Check image quality metrics"""
    quality_scores = {}
    
    # Check face size
    width = face_box['box'][2]
    height = face_box['box'][3]
    face_size = min(width, height)
    quality_scores['face_size'] = face_size
    
    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray) / 255.0
    quality_scores['brightness'] = brightness
    
    # Check sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    quality_scores['sharpness'] = sharpness
    
    # Check face detection confidence
    quality_scores['confidence'] = face_box['confidence']
    
    # Check if all metrics pass thresholds
    passes_threshold = (
        face_size >= ImageQuality.MIN_FACE_SIZE and
        ImageQuality.MIN_BRIGHTNESS <= brightness <= ImageQuality.MAX_BRIGHTNESS and
        sharpness >= ImageQuality.MIN_SHARPNESS and
        face_box['confidence'] >= ImageQuality.MIN_CONFIDENCE
    )
    
    return passes_threshold, quality_scores

def extract_faces(img: np.ndarray, required_size=(160, 160)) -> List[Tuple[np.ndarray, Dict]]:
    """Extract multiple faces from image using MTCNN"""
    # Detect faces
    results = detector.detect_faces(img)
    if not results:
        return []
    
    faces = []
    for face_box in results:
        x1, y1, width, height = face_box['box']
        x2, y2 = x1 + width, y1 + height
        
        # Extract face
        face = img[y1:y2, x1:x2]
        
        # Check quality
        passes_quality, quality_scores = check_image_quality(img, face_box)
        if not passes_quality:
            continue
        
        # Resize to required size
        face = cv2.resize(face, required_size)
        
        # Convert to float32 and normalize
        face = face.astype('float32')
        face = (face - face.mean()) / face.std()
        
        faces.append((face, quality_scores))
    
    return faces

def get_embeddings(faces: List[np.ndarray]) -> List[np.ndarray]:
    """Get face embeddings for multiple faces using FaceNet"""
    if not faces:
        return []
    
    # Stack faces into a batch
    samples = np.stack(faces)
    
    # Get embeddings
    return facenet.embeddings(samples)

def decode_base64_image(base64_image: str) -> Optional[np.ndarray]:
    """Decode base64 image to numpy array"""
    try:
        img_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return preprocess_image(img)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_embedding(embedding: np.ndarray) -> str:
    """Convert embedding to base64 string"""
    return base64.b64encode(embedding.tobytes()).decode()

def decode_embedding(encoded_embedding: str) -> np.ndarray:
    """Convert base64 string to embedding"""
    embedding_bytes = base64.b64decode(encoded_embedding)
    return np.frombuffer(embedding_bytes, dtype=np.float32)

def get_face_embeddings(image: str) -> List[Tuple[str, Dict[str, float]]]:
    """Get face embeddings from base64 image for multiple faces"""
    # Decode image
    img = decode_base64_image(image)
    if img is None:
        return []
    
    # Extract faces and quality scores
    faces_and_scores = extract_faces(img)
    if not faces_and_scores:
        return []
    
    # Separate faces and scores
    faces, quality_scores = zip(*faces_and_scores)
    
    # Get embeddings
    embeddings = get_embeddings(list(faces))
    
    # Encode embeddings and pair with quality scores
    return [(encode_embedding(emb), scores) for emb, scores in zip(embeddings, quality_scores)]

def compare_embeddings(
    known_embedding: str,
    unknown_embedding: str,
    threshold: float = 0.7
) -> Tuple[bool, float]:
    """Compare two face embeddings and return match status and similarity score"""
    # Decode embeddings
    emb1 = decode_embedding(known_embedding)
    emb2 = decode_embedding(unknown_embedding)
    
    # Reshape embeddings for cosine similarity
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    
    # Calculate similarity score
    similarity = cosine_similarity(emb1, emb2)[0][0]
    
    # Convert to percentage
    similarity_score = (similarity + 1) * 50  # Convert from [-1,1] to [0,100]
    
    # Check if faces match
    matches = similarity_score >= (threshold * 100)
    
    return matches, similarity_score

class FaceRecognitionModel:
    def __init__(self):
        """Initialize the face recognition model"""
        # Set memory growth for GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
    
    def get_face_embeddings(self, image: str) -> List[Tuple[str, Dict[str, float]]]:
        """Get face embeddings and quality scores from image"""
        return get_face_embeddings(image)
    
    def compare_faces(
        self,
        known_embedding: str,
        unknown_embedding: str,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """Compare two face embeddings"""
        return compare_embeddings(known_embedding, unknown_embedding, threshold)