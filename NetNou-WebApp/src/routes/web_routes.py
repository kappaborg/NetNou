"""Web routes for the application."""

from flask import Blueprint, render_template, redirect, url_for, session, request, flash
from ..services.auth_service import authenticate_user, is_authenticated
from ..services.class_service import get_classes, get_class_attendance
from ..services.student_service import get_students

# Create blueprint for web routes
web = Blueprint('web', __name__)

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
    
    return render_template('login.html')

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
    
    classes = get_classes()
    return render_template('dashboard.html', classes=classes)

# Class routes
@web.route('/classes/<class_id>')
def class_detail(class_id):
    """Class detail page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    attendance = get_class_attendance(class_id)
    return render_template('class_detail.html', attendance=attendance)

# Student routes
@web.route('/students')
def students():
    """Student list page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    students_list = get_students()
    return render_template('students.html', students=students_list)

# Face recognition routes
@web.route('/take-attendance')
def take_attendance():
    """Face recognition attendance page."""
    if not is_authenticated():
        return redirect(url_for('web.login'))
    
    classes = get_classes()
    return render_template('take_attendance.html', classes=classes) 