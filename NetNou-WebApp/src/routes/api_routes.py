"""API routes for the application."""

from flask import Blueprint, request, jsonify, session
from flask_login import login_required, current_user
from ..services.auth_service import authenticate_user
from ..services.student_service import get_students, get_student, create_student, delete_student
from ..services.class_service import get_classes, create_class, record_attendance, get_class
from ..services.attendance_service import get_student_details, take_attendance, get_attendance_for_class
from ..services.face_service import (
    register_face, 
    identify_face, 
    face_recognition_attendance,
    delete_face_registration,
    FACE_EMBEDDINGS
)
from ..database.student_model import update_student_face

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
@login_required
def register_face_route():
    """Register a student's face."""
    data = request.json
    if not data or 'student_id' not in data or 'image_data' not in data:
        return jsonify({'success': False, 'message': 'Missing required data'}), 400

    student_id = data['student_id']
    image_data = data['image_data']

    # Check if the student exists
    student = get_student(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'}), 404

    # Register the face
    result = register_face(student_id, image_data)

    if result['success']:
        # Update the student's face registration status
        update_student_face(student_id, True)
        return jsonify(result)
    else:
        return jsonify(result), 400

@api.route('/register-face/<student_id>', methods=['DELETE'])
@login_required
def delete_face_registration_route(student_id):
    """Delete a student's face registration."""
    # Check if student exists
    student = get_student(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'}), 404

    # Delete the face registration
    result = delete_face_registration(student_id)
    
    if result['success']:
        # Update the student's face registration status
        update_student_face(student_id, False)
        return jsonify(result)
    else:
        return jsonify(result), 404

@api.route('/register-face/<student_id>', methods=['PUT'])
@login_required
def update_face_registration_route(student_id):
    """Update a student's face registration."""
    # Check if student exists
    student = get_student(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'}), 404

    data = request.json
    if not data or 'image_data' not in data:
        return jsonify({'success': False, 'message': 'Missing image data'}), 400

    image_data = data['image_data']

    # Register face will handle updating if face already exists
    result = register_face(student_id, image_data)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400

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

# Keeping old method for backward compatibility
@api.route('/delete-face-registration', methods=['POST'])
@login_required
def legacy_delete_face_registration_route():
    """Legacy endpoint to delete a student's face registration."""
    data = request.json
    if not data or 'student_id' not in data:
        return jsonify({'success': False, 'message': 'Missing student ID'}), 400

    student_id = data['student_id']

    # Check if student exists
    student = get_student(student_id)
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'}), 404

    # Delete the face registration
    result = delete_face_registration(student_id)
    
    if result['success']:
        # Update the student's face registration status
        update_student_face(student_id, False)
        return jsonify(result)
    else:
        return jsonify(result), 404

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

@api.route('/take-attendance', methods=['POST'])
@login_required
def take_attendance_route():
    data = request.json
    if not data or 'class_id' not in data or 'attendance_data' not in data:
        return jsonify({'success': False, 'message': 'Missing required data'}), 400

    class_id = data['class_id']
    attendance_data = data['attendance_data']
    
    # Check if the class exists
    class_obj = get_class(class_id)
    if not class_obj:
        return jsonify({'success': False, 'message': 'Class not found'}), 404

    # Take attendance
    result = take_attendance(class_id, attendance_data)
    return jsonify(result)

@api.route('/class/<class_id>/attendance', methods=['GET'])
@login_required
def get_class_attendance_route(class_id):
    # Check if the class exists
    class_obj = get_class(class_id)
    if not class_obj:
        return jsonify({'success': False, 'message': 'Class not found'}), 404

    # Get attendance records for the class
    attendance = get_attendance_for_class(class_id)
    return jsonify({
        'success': True,
        'class': class_obj,
        'attendance': attendance
    })

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

@api.route('/students/<student_id>', methods=['DELETE'])
@login_required
def delete_student_route(student_id):
    """Delete a student by ID."""
    result = delete_student(student_id)
    
    if not result['success']:
        return jsonify(result), 404
    
    return jsonify(result)

@api.route('/student/<student_id>/details', methods=['GET'])
@login_required
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