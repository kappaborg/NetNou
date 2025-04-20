"""API routes for the application."""

from flask import Blueprint, request, jsonify
from ..services.auth_service import authenticate_user
from ..services.attendance_service import record_attendance
from ..services.face_service import register_face, identify_face
from ..services.class_service import get_classes, create_class

# Create blueprint for API routes
api = Blueprint('api', __name__, url_prefix='/api')

# Authentication routes
@api.route('/login', methods=['POST'])
def login():
    """User login endpoint."""
    username = request.json.get('username')
    password = request.json.get('password')
    
    result = authenticate_user(username, password)
    if result['success']:
        return jsonify(result), 200
    return jsonify(result), 401

# Attendance routes
@api.route('/record-attendance', methods=['POST'])
def attendance():
    """Record student attendance."""
    class_id = request.json.get('class_id')
    student_id = request.json.get('student_id')
    
    result = record_attendance(class_id, student_id)
    return jsonify(result), 200 if result['success'] else 400

# Face recognition routes
@api.route('/register-face', methods=['POST'])
def register():
    """Register student face."""
    student_id = request.json.get('student_id')
    image_data = request.json.get('image_data')
    
    result = register_face(student_id, image_data)
    return jsonify(result), 200 if result['success'] else 400

@api.route('/identify-face', methods=['POST'])
def identify():
    """Identify a face from image data."""
    image_data = request.json.get('image_data')
    
    result = identify_face(image_data)
    return jsonify(result), 200 if result['success'] else 400

# Class management routes
@api.route('/classes', methods=['GET'])
def classes():
    """Get all classes."""
    return jsonify(get_classes()), 200

@api.route('/classes', methods=['POST'])
def create():
    """Create a new class."""
    class_data = request.json
    
    result = create_class(class_data)
    return jsonify(result), 201 if result['success'] else 400 