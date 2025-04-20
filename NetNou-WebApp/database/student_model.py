"""Student model for the application."""

from datetime import datetime

# In-memory student database
STUDENTS = [
    {
        'id': '1001',
        'first_name': 'Fifi',
        'last_name': 'ONE',
        'email': 'cincancoe@example.com',
        'gender': 'Female',
        'date_of_birth': '2000-05-15',
        'has_face_registered': False,
        'created_at': datetime.now().isoformat()
    },
    {
        'id': '1002',
        'first_name': 'Kappasutra',
        'last_name': 'none',
        'email': 'demo@example.com',
        'gender': 'Male',
        'date_of_birth': '2001-08-22',
        'has_face_registered': False,
        'created_at': datetime.now().isoformat()
    }
]

def get_student(student_id):
    """Get a student by ID.
    
    Args:
        student_id (str): The student ID to search for
        
    Returns:
        dict: Student object if found, None otherwise
    """
    for student in STUDENTS:
        if student['id'] == student_id:
            return student
    return None

def get_students():
    """Get all students.
    
    Returns:
        list: List of all students
    """
    return STUDENTS

def create_student(student_data):
    """Create a new student.
    
    Args:
        student_data (dict): Student data including name, email, etc.
        
    Returns:
        dict: Created student object
    """
    # Check if student ID already exists
    if 'id' in student_data and get_student(student_data['id']):
        return None
    
    # Generate ID if not provided
    if 'id' not in student_data:
        # Generate a unique ID
        last_id = max(int(student['id']) for student in STUDENTS) if STUDENTS else 1000
        student_data['id'] = str(last_id + 1)
    
    # Create student with defaults
    new_student = {
        'id': student_data.get('id'),
        'first_name': student_data.get('first_name', ''),
        'last_name': student_data.get('last_name', ''),
        'email': student_data.get('email', ''),
        'gender': student_data.get('gender', ''),
        'date_of_birth': student_data.get('date_of_birth', ''),
        'has_face_registered': student_data.get('has_face_registered', False),
        'created_at': datetime.now().isoformat()
    }
    
    # Add to database
    STUDENTS.append(new_student)
    
    return new_student

def update_student(student_id, updates):
    """Update a student.
    
    Args:
        student_id (str): The student ID to update
        updates (dict): Field updates for the student
        
    Returns:
        dict: Updated student object, or None if not found
    """
    student = get_student(student_id)
    if not student:
        return None
    
    # Update fields
    for key, value in updates.items():
        if key in student and key != 'id':  # Don't allow changing ID
            student[key] = value
    
    return student

def update_student_face(student_id, has_face_registered):
    """Update a student's face registration status.
    
    Args:
        student_id (str): The student ID to update
        has_face_registered (bool): Whether the student has a face registered
        
    Returns:
        dict: Updated student object, or None if not found
    """
    return update_student(student_id, {'has_face_registered': has_face_registered}) 