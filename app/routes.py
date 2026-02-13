from flask import Blueprint, render_template, Response, request, redirect, url_for, flash, current_app
from . import db
from .models import Student, Attendance
import cv2
import numpy as np
import os
from datetime import datetime, timedelta

main = Blueprint('main', __name__)

# Global vars
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not lead haarcascade!")
else:
    print(f"Haarcascade loaded from {cv2.data.haarcascades}")

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.environ.get('K_SERVICE') or os.environ.get('K_REVISION'):
    # Cloud Run Environment - use /tmp
    TRAINER_PATH = '/tmp/trainer.yml'
    FACES_DIR = '/tmp/known_faces'
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)
else:
    # Local Environment
    TRAINER_PATH = 'instance/trainer.yml'
    FACES_DIR = 'known_faces'

def train_model():
    """Trains the LBPH model with images from known_faces directory."""
    faces = []
    ids = []
    
    # Map internal DB IDs to images
    # We assume folder structure: known_faces/student_db_id/image.jpg
    # Or simplified: known_faces/student_db_id_name.jpg (easier for single image)
    # Better: standard dataset approach.
    
    # Let's iterate through students in DB
    with current_app.app_context():
        students = Student.query.all()
        
        has_faces = False
        for student in students:
            student_dir = os.path.join(FACES_DIR, str(student.id))
            if not os.path.exists(student_dir):
                continue
                
            for filename in os.listdir(student_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(student_dir, filename)
                    try:
                        img = cv2.imread(path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        for (x, y, w, h) in faces_rect:
                            faces.append(gray[y:y+h, x:x+w])
                            ids.append(student.id)
                            has_faces = True
                    except Exception as e:
                        print(f"Error processing {path}: {e}")

        if has_faces:
            recognizer.train(faces, np.array(ids))
            recognizer.save(TRAINER_PATH)
            print("Model trained and saved.")
        else:
            print("No faces found to train.")

def load_model():
    if os.path.exists(TRAINER_PATH):
        try:
            recognizer.read(TRAINER_PATH)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    return False

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/monitor')
def monitor():
    return render_template('monitor.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        student_id = request.form.get('student_id')
        file = request.files.get('file')

        if not name or not student_id or not file:
            flash('Please provide name, ID and a photo.')
            return redirect(url_for('main.register'))

        try:
            # Check if student exists or create new
            student = Student.query.filter_by(student_id=student_id).first()
            if not student:
                student = Student(name=name, student_id=student_id)
                db.session.add(student)
                db.session.commit()
            
            # Save Image
            student_dir = os.path.join(FACES_DIR, str(student.id))
            os.makedirs(student_dir, exist_ok=True)
            
            # Save file
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            save_path = os.path.join(student_dir, filename)
            file.save(save_path)
            
            # Trigger Training
            train_model()
            
            flash(f'Student {name} registered and model updated!')
            return redirect(url_for('main.index'))
            
        except Exception as e:
            flash(f'Error registering: {str(e)}')
            
    return render_template('register.html')

@main.route('/attendance')
def attendance():
    records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
    
    # Calculate Summary Stats
    total_registered = Student.query.count()
    # optimized: count distinct student_ids in attendance table
    attended_student_ids = db.session.query(Attendance.student_id).distinct().all()
    total_attended = len(attended_student_ids)
    
    return render_template('attendance_view.html', records=records, 
                           total_registered=total_registered, 
                           total_attended=total_attended)

import base64

@main.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return {'error': 'No image data'}, 400

    # Decode image
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    model_loaded = load_model()
    faces_data = []

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Using the same parameters as before
        faces_rect = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Load names (caching this would be better but keeping simple for now)
        names = {}
        with current_app.app_context():
            for s in Student.query.all():
                names[s.id] = s.name

        last_attendance_time = {} # We lose this state each request with stateless HTTP :(. 
        # Ideally, we should store this in a global var or DB/Cache.
        # For this simple app, let's use a global var in this module carefully.
        global global_last_attendance
        if 'global_last_attendance' not in globals():
            global_last_attendance = {}

        for (x, y, w, h) in faces_rect:
            name = "Unknown"
            display_label = "Unknown"
            color = "#FF0000" # Red

            if model_loaded:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    id_, confidence = recognizer.predict(roi_gray)
                    
                    if confidence < 80:
                        student_name = names.get(id_, "Unknown")
                        display_label = f"{student_name} (ID:{id_})"
                        color = "#00FF00" # Green
                        
                        # Mark Attendance
                        now = datetime.utcnow()
                        if id_ not in global_last_attendance or (now - global_last_attendance[id_]) > timedelta(minutes=1):
                            global_last_attendance[id_] = now
                            try:
                                with current_app.app_context():
                                    att = Attendance(student_id=id_, status='Present')
                                    db.session.add(att)
                                    db.session.commit()
                                    print(f"Attendance marked for {display_label}")
                            except Exception as e:
                                print(f"Error DB: {e}")
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            faces_data.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'name': display_label,
                'color': color
            })
            
    except Exception as e:
        print(f"Error processing frame: {e}")

    return {'faces': faces_data}

# Standard routes
