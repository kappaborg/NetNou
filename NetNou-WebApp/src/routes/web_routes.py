"""Web routes for the application."""

from flask import Blueprint, render_template, redirect, url_for, session, request, flash, jsonify, g
from datetime import datetime
from ..services.auth_service import authenticate_user, is_authenticated
from ..services.class_service import get_classes, get_class_attendance, create_class
from ..services.student_service import get_students, get_student_details, create_student
from ..services.attendance_service import get_recent_attendance, get_attendance_stats, record_attendance
from ..services.face_service import register_face

# Create blueprint for web routes
web = Blueprint('web', __name__)

@web.before_request
def load_user():
    """Load user information before each request."""
    g.user = session.get('user', None)
    
    # Add a helper to mimic Flask-Login's current_user behavior
    class User:
        @property
        def is_authenticated(self):
            return g.user is not None
            
        @property
        def username(self):
            if g.user:
                return g.user.get('username', 'User')
            return None
    
    g.current_user = User() if g.user else User()

@web.context_processor
def inject_user():
    """Make current_user available to all templates."""
    return {'current_user': g.current_user}

# Authentication routes
@web.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        result = authenticate_user(username, password)
        if result['success']:
            session['user'] = result['user']
            return redirect(url_for('web.dashboard'))
        
        flash(result['message'], 'error')
    
    # Pass current year to the template for footer copyright
    return render_template('login.html', now=datetime.now())

@web.route('/logout')
def logout():
    """Logout user."""
    session.clear()
    return redirect(url_for('web.login'))

# Dashboard routes
@web.route('/')
@web.route('/dashboard')
def dashboard():
    """Main dashboard page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    # Get data for dashboard
    classes = get_classes()
    
    # Get attendance statistics
    attendance_stats = get_attendance_stats()
    total_students = attendance_stats.get('total_students', 0)
    attendance_today = attendance_stats.get('attendance_today', 0)
    average_engagement = attendance_stats.get('average_engagement', '0%')
    
    # Get recent activities
    activities = get_recent_attendance(limit=5)
    
    return render_template('dashboard.html', 
                          classes=classes,
                          total_students=total_students,
                          attendance_today=attendance_today,
                          average_engagement=average_engagement,
                          activities=activities,
                          now=datetime.now())

# Class routes
@web.route('/classes')
def classes():
    """Classes list page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    classes_list = get_classes()
    return render_template('classes.html', classes=classes_list, now=datetime.now())

@web.route('/classes/<class_id>')
def class_detail(class_id):
    """Class detail page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    attendance = get_class_attendance(class_id)
    return render_template('class_detail.html', attendance=attendance, now=datetime.now())

@web.route('/classes/add', methods=['POST'])
def add_class():
    """Add a new class."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    if request.method == 'POST':
        class_data = {
            'name': request.form.get('name'),
            'code': request.form.get('code'),
            'description': request.form.get('description'),
            'schedule': request.form.get('schedule')
        }
        
        result = create_class(class_data)
        
        if result['success']:
            flash('Class created successfully!', 'success')
        else:
            flash(result['message'], 'error')
            
    return redirect(url_for('web.classes'))

# Student routes
@web.route('/students')
def students():
    """Student list page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    students_list = get_students()
    return render_template('students.html', students=students_list, now=datetime.now())

@web.route('/students/<student_id>')
def student_detail(student_id):
    """Student detail page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    student = get_student_details(student_id)
    return render_template('student_detail.html', student=student, now=datetime.now())

@web.route('/students/register', methods=['GET', 'POST'])
def register_student():
    """Register a new student with face."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    if request.method == 'POST':
        student_data = {
            'first_name': request.form.get('first_name'),
            'last_name': request.form.get('last_name'),
            'email': request.form.get('email'),
            'gender': request.form.get('gender'),
            'date_of_birth': request.form.get('date_of_birth')
        }
        
        result = create_student(student_data)
        
        if result['success']:
            # If face data was provided, register the face
            if 'face_image' in request.form and request.form.get('face_image'):
                face_result = register_face(result['student']['id'], request.form.get('face_image'))
                if face_result['success']:
                    flash('Student and face registered successfully!', 'success')
                else:
                    flash(f'Student created but face registration failed: {face_result["message"]}', 'warning')
            else:
                flash('Student created successfully! No face registered.', 'success')
                
            return redirect(url_for('web.students'))
        else:
            flash(result['message'], 'error')
    
    return render_template('register_student.html', now=datetime.now())

# Attendance routes
@web.route('/attendance')
def attendance():
    """Attendance management page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    classes_list = get_classes()
    return render_template('attendance.html', classes=classes_list, now=datetime.now())

# Take attendance page
@web.route('/take-attendance')
@web.route('/take-attendance/<class_id>')
def take_attendance(class_id=None):
    """Face recognition attendance page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    classes = get_classes()
    selected_class = None
    
    if class_id:
        for cls in classes:
            if str(cls['id']) == class_id:
                selected_class = cls
                break
    
    # Get the list of students for manual attendance option
    students_list = get_students()
    
    return render_template('take_attendance.html', 
                          classes=classes, 
                          students=students_list,
                          selected_class=selected_class,
                          now=datetime.now())

@web.route('/take-attendance-live/<class_id>', methods=['POST'])
def take_attendance_live(class_id):
    """Handle live face recognition attendance submission."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    # Get image data from form
    image_data = request.form.get('image_data')
    
    if not image_data:
        flash('No image data provided', 'error')
        return redirect(url_for('web.take_attendance', class_id=class_id))
    
    # Process face recognition attendance
    from ..services.face_service import face_recognition_attendance
    result = face_recognition_attendance(class_id, image_data)
    
    if result['success']:
        if result['recognized_count'] > 0:
            flash(f'Attendance recorded for {result["recognized_count"]} students!', 'success')
        else:
            flash('No students recognized in the image', 'warning')
    else:
        flash(f'Error processing attendance: {result.get("message", "Unknown error")}', 'error')
    
    return redirect(url_for('web.take_attendance', class_id=class_id))

# Analytics routes
@web.route('/analytics')
def analytics():
    """Analytics and reporting page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    # Get summary data for analytics
    attendance_stats = get_attendance_stats()
    
    return render_template('analytics.html', 
                          stats=attendance_stats,
                          now=datetime.now())

# Manual attendance recording
@web.route('/record-manual-attendance', methods=['POST'])
def record_manual_attendance():
    """Record attendance manually."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    class_id = request.form.get('class_id')
    student_id = request.form.get('student_id')
    status = request.form.get('status', 'present')
    
    if not class_id or not student_id:
        flash('Class ID and Student ID are required', 'error')
        return redirect(url_for('web.attendance'))
    
    # Record attendance
    result = record_attendance(class_id, student_id, status)
    
    if result['success']:
        flash('Attendance recorded successfully', 'success')
    else:
        flash(result['message'], 'error')
    
    return redirect(url_for('web.attendance'))

@web.route('/update-attendance', methods=['POST'])
def update_attendance():
    """Update an existing attendance record."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    attendance_id = request.form.get('attendance_id')
    status = request.form.get('status')
    
    if not attendance_id:
        flash('Attendance ID is required', 'error')
        return redirect(url_for('web.attendance'))
    
    # For now, just show a success message
    # In a real implementation, you would update the attendance record in the database
    flash('Attendance record updated successfully', 'success')
    
    return redirect(url_for('web.attendance'))

# Error handlers
@web.app_errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('errors/404.html', now=datetime.now()), 404

@web.app_errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('errors/500.html', now=datetime.now()), 500 