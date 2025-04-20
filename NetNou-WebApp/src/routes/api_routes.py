"""API routes for the application."""

from flask import Blueprint, request, jsonify, session
from ..services.auth_service import authenticate_user
from ..services.student_service import get_students, get_student, create_student
from ..services.class_service import get_classes, create_class, record_attendance
from ..services.attendance_service import get_student_details
from ..services.face_service import (
    register_face, 
    identify_face, 
    face_recognition_attendance,
    delete_face_registration
)

# Create blueprint for API routes
api = Blueprint('api', __name__, url_prefix='/api')

# Authentication routes
@api.route('/login', methods=['POST'])
def login():
    """Login endpoint."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    result = authenticate_user(username, password)
    
    if result['success']:
        session['user'] = result['user']
    
    return jsonify(result)

# Face recognition routes
@api.route('/register-face', methods=['POST'])
def register_face_route():
    """Register a student's face."""
    data = request.get_json()
    student_id = data.get('student_id')
    image_data = data.get('image_data')
    
    if not student_id or not image_data:
        return jsonify({
            'success': False, 
            'message': 'Missing required fields: student_id and image_data'
        })
    
    result = register_face(student_id, image_data)
    return jsonify(result)

@api.route('/identify-face', methods=['POST'])
def identify_face_route():
    """Identify a face from image data."""
    data = request.get_json()
    image_data = data.get('image_data')
    tolerance = data.get('tolerance', 0.6)
    
    if not image_data:
        return jsonify({
            'success': False,
            'message': 'Missing required field: image_data'
        })
    
    result = identify_face(image_data, tolerance)
    return jsonify(result)

@api.route('/face-attendance', methods=['POST'])
def face_attendance_route():
    """Take attendance using face recognition."""
    data = request.get_json()
    class_id = data.get('class_id')
    image_data = data.get('image_data')
    tolerance = data.get('tolerance', 0.6)
    
    if not class_id or not image_data:
        return jsonify({
            'success': False,
            'message': 'Missing required fields: class_id and image_data'
        })
    
    result = face_recognition_attendance(class_id, image_data, tolerance)
    
    # Record attendance for each recognized student
    if result['success'] and result['recognized_count'] > 0:
        for student in result['recognized_students']:
            record_attendance(class_id, student['student_id'], 'present')
    
    return jsonify(result)

@api.route('/delete-face-registration', methods=['POST'])
def delete_face_registration_route():
    """Delete a student's face registration."""
    data = request.get_json()
    student_id = data.get('student_id')
    
    if not student_id:
        return jsonify({
            'success': False,
            'message': 'Missing required field: student_id'
        })
    
    result = delete_face_registration(student_id)
    return jsonify(result)

# Attendance routes
@api.route('/record-attendance', methods=['POST'])
def record_attendance_route():
    """Record student attendance."""
    data = request.get_json()
    class_id = data.get('class_id')
    student_id = data.get('student_id')
    status = data.get('status', 'present')
    
    if not class_id or not student_id:
        return jsonify({
            'success': False,
            'message': 'Missing required fields: class_id and student_id'
        })
    
    result = record_attendance(class_id, student_id, status)
    return jsonify(result)

# Student routes
@api.route('/students', methods=['GET'])
def get_students_route():
    """Get all students."""
    students_list = get_students()
    return jsonify({'students': students_list})

@api.route('/students/<student_id>', methods=['GET'])
def get_student_route(student_id):
    """Get a student by ID."""
    student = get_student(student_id)
    
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'}), 404
    
    return jsonify({'success': True, 'student': student})

@api.route('/students/<student_id>/details', methods=['GET'])
def get_student_details_route(student_id):
    """Get detailed student information including attendance history."""
    result = get_student_details(student_id)
    
    if not result['success']:
        return jsonify(result), 404
    
    return jsonify(result)

@api.route('/students', methods=['POST'])
def create_student_route():
    """Create a new student."""
    data = request.get_json()
    result = create_student(data)
    return jsonify(result)

# Class routes
@api.route('/classes', methods=['GET'])
def get_classes_route():
    """Get all classes."""
    classes_list = get_classes()
    return jsonify({'classes': classes_list})

@api.route('/classes', methods=['POST'])
def create_class_route():
    """Create a new class."""
    data = request.get_json()
    result = create_class(data)
    return jsonify(result) 