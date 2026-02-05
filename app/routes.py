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
    return render_template('attendance_view.html', records=records)

@main.route('/video_feed')
def video_feed():
    from flask import current_app
    app = current_app._get_current_object()
    return Response(gen_frames(app), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(app):
    # Try using DirectShow backend for Windows (fixes MSMF errors)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        # Return a black frame with text
        blank_image = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank_image, "Camera Not Found", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', blank_image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    model_loaded = load_model()
    
    last_attendance_time = {} 
    
    # Load names for display
    names = {}
    with app.app_context():
        for s in Student.query.all():
            names[s.id] = s.name

    while True:
        success, frame = camera.read()
        if not success:
             break
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Relaxed parameters: scaleFactor 1.1 (more detailed), minNeighbors 4
            faces_rect = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces_rect:
                name = "Unknown"
                display_label = "Unknown"
                confidence_text = ""
                
                if model_loaded:
                    roi_gray = gray[y:y+h, x:x+w]
                    try:
                        id_, confidence = recognizer.predict(roi_gray)
                        
                        if confidence < 80: # Relaxed confidence slightly
                            student_name = names.get(id_, "Unknown")
                            display_label = f"{student_name} (ID:{id_})"
                            confidence_text = f" {round(100 - confidence)}%"
                            
                            now = datetime.utcnow()
                            if id_ not in last_attendance_time or (now - last_attendance_time[id_]) > timedelta(minutes=1):
                                last_attendance_time[id_] = now
                                try:
                                    with app.app_context():
                                        att = Attendance(student_id=id_, status='Present')
                                        db.session.add(att)
                                        db.session.commit()
                                        print(f"Attendance marked for {name}")
                                except Exception as e:
                                    print(f"Error DB: {e}")
                        else:
                            display_label = "Unknown"
                    except Exception as e:
                        print(f"Prediction error: {e}")

                # Draw rect and name inside the loop
                color = (0, 255, 0) if display_label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_label + confidence_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error in video loop: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()
