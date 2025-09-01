from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import datetime
import time
import base64
from PIL import Image
import io
import threading
import logging
import pytz
from database import AttendanceDatabase

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "./TrainingImageLabel/Trainner.yml"
trainimage_path = "./TrainingImage"

# Global variables for face recognition
recognizer = None
face_cascade = None
db = None  # Database instance
training_in_progress = False

# State for temporal smoothing
last_detection_status = {"status": "no_face", "message": "No face detected"}
last_detection_time = 0
DETECTION_HOLD_SECONDS = 2.0  # Hold status for 2 seconds


# Detection configuration (tuned for more consistent results based on reference code)
DETECTION_SCALE_FACTOR = 1.1  # More sensitive detection like reference
DETECTION_MIN_NEIGHBORS = 3   # More sensitive like reference
DETECTION_MIN_SIZE = (50, 50)



def filter_and_select_face(faces, frame_shape):
    """Select the largest face (simplified approach like reference code).
    """
    if len(faces) == 0:
        return None
    
    # Simply return the largest face by area (like reference code)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    return largest_face

def preprocess_frame(bgr_image):
    """Apply preprocessing to improve detection stability (simplified based on reference).
    """
    try:
        # Simple grayscale conversion like reference code
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # Optional: light histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        return gray
    except Exception:
        # Fallback: simple grayscale
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

def adaptive_min_neighbors(gray_img):
    """Use consistent detection parameters like reference code."""
    # Use consistent parameters for better stability
    return DETECTION_MIN_NEIGHBORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_recognition_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_recognition_api")

def initialize_recognizer():
    """Initialize the face recognizer and load student data"""
    global recognizer, face_cascade, db
    
    try:
        # Initialize database
        db = AttendanceDatabase()
        logger.info("Database initialized successfully")
        
        # Initialize face cascade first
        face_cascade = cv2.CascadeClassifier(haarcasecade_path)
        if face_cascade.empty():
            logger.error(f"Failed to load Haar cascade from {haarcasecade_path}")
            face_cascade = None
            return False
        else:
            logger.info("Face cascade loaded successfully")
        
        # Load trained model if it exists
        if os.path.exists(trainimagelabel_path):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(trainimagelabel_path)
            logger.info("Face recognizer loaded successfully")
        else:
            logger.warning(f"Trained model not found at {trainimagelabel_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing recognizer: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if recognizer is not None else "not_loaded"
    students_count = len(db.get_all_students()) if db is not None else 0
    
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.datetime.now().isoformat(),
        "model_status": model_status,
        "students_count": students_count,
        "version": "1.0.0"
    })

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Enhanced face recognition endpoint with session-based entry/exit logic"""
    try:
        if recognizer is None:
            return jsonify({"status": "error", "message": "Model not loaded. Please train the model first."}), 400
        
        if face_cascade is None or face_cascade.empty():
            return jsonify({"status": "error", "message": "Face detection model not loaded. Please check Haar cascade file."}), 400
        
        # Always-on mode: proceed even if there's no active session
        active_session = db.get_active_session()
        always_on_mode = active_session is None
        if always_on_mode:
            logger.debug("Recognize: running in always-on mode (no active session)")
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert & preprocess
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = preprocess_frame(opencv_image)

        # Adaptive neighbors based on brightness
        min_neighbors = adaptive_min_neighbors(gray)

        # Detect faces with debug logging
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=min_neighbors,
            minSize=DETECTION_MIN_SIZE
        )
        
        logger.info(f"Recognition: Detected {len(faces)} raw faces")
        
        if len(faces) == 0:
            return jsonify({"status": "no_face", "message": "No face detected"})

        # Apply filtering & temporal smoothing (single dominant face scenario)
        selected = filter_and_select_face(faces, opencv_image.shape)
        if selected is not None:
            faces = [selected]
            logger.info(f"Recognition: Selected face at {selected}")
        else:
            return jsonify({"status": "no_face", "message": "Detections filtered out"})
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Predict
            id_pred, conf = recognizer.predict(face_roi)
            logger.debug(f"Recognition confidence: {conf} for ID: {id_pred}")
            
            # Threshold for recognition - lower value means better match
            if conf < 55:
                # Find student in database
                student = db.get_student(str(id_pred))
                
                if student:
                    name = student["name"]
                    student_id = str(id_pred)
                    ist = pytz.timezone('Asia/Kolkata')
                    current_time = datetime.datetime.now(ist)

                    if always_on_mode:
                        # Simple recognition without attendance logic
                        action = 'RECOGNIZE'
                        status = 'OUT_OF_SESSION'
                        message = 'Recognized outside any active session'
                        tts_message = f"Hello {name}."
                        try:
                            from improved_tts import speak_message
                            speak_message(tts_message)
                        except Exception:
                            pass
                        logger.info(f"Recognized (always-on) student: {name} (ID: {student_id})")
                        results.append({
                            "status": "recognized",
                            "action": action,
                            "attendance_status": status,
                            "student": {
                                "name": name,
                                "id": student_id,
                                "confidence": float(conf)
                            },
                            "session": None,
                            "message": message,
                            "timestamp": current_time.isoformat(),
                            "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                        })
                    else:
                        session_id = active_session['id']
                        # Parse session times - handle both HH:MM and full datetime formats
                        try:
                            if len(active_session['start_time']) <= 5:  # HH:MM format
                                today = current_time.date()
                                start_time = datetime.datetime.strptime(active_session['start_time'], '%H:%M').time()
                                end_time = datetime.datetime.strptime(active_session['end_time'], '%H:%M').time()
                                session_start = datetime.datetime.combine(today, start_time).replace(tzinfo=ist)
                                session_end = datetime.datetime.combine(today, end_time).replace(tzinfo=ist)
                            else:  # Full datetime format
                                session_start = datetime.datetime.fromisoformat(active_session['start_time']).replace(tzinfo=ist)
                                session_end = datetime.datetime.fromisoformat(active_session['end_time']).replace(tzinfo=ist)
                        except Exception as e:
                            logger.error(f"Error parsing session times: {e}")
                            continue

                        # Check previous entries
                        with db.get_connection() as conn:
                            last_log = conn.execute("""
                                SELECT action, status, timestamp, message FROM attendance_logs 
                                WHERE student_id = ? AND session_id = ? 
                                ORDER BY timestamp DESC LIMIT 1
                            """, (student_id, session_id)).fetchone()

                        action = 'ENTRY'
                        status = 'PRESENT'
                        message = ""
                        tts_message = ""

                        # Time windows based on your requirements:
                        # - Present: 15 min before start to 10 min after start
                        # - Absent: more than 10 min after start
                        # - No exit during class (blocked)
                        # - Exit only after class ends
                        
                        early_window_start = session_start - datetime.timedelta(minutes=15)
                        late_cutoff = session_start + datetime.timedelta(minutes=10)
                        
                        if current_time < early_window_start:
                            # Too early - before 15 min window
                            status = 'BLOCKED'
                            message = f"Too early - Please come back after {early_window_start.strftime('%H:%M')}"
                            tts_message = f"{name}, you're too early. Please come back after {early_window_start.strftime('%H:%M')}."
                            
                        elif early_window_start <= current_time <= late_cutoff:
                            # Present window: 15 min before to 10 min after start
                            if not last_log:
                                status = 'PRESENT'
                                if current_time < session_start:
                                    message = f"Present - Early arrival (Class starts at {session_start.strftime('%H:%M')})"
                                    tts_message = f"Good morning {name}! Attendance marked as present. Class starts at {session_start.strftime('%H:%M')}."
                                else:
                                    message = "Present - On time"
                                    tts_message = f"Welcome {name}! Attendance marked as present."
                            else:
                                status = 'BLOCKED'
                                message = "Already marked present for this class."
                                tts_message = f"{name}, you're already marked present for this class."
                                
                        elif late_cutoff < current_time <= session_end:
                            # Late arrival during class - marked as absent
                            if not last_log:
                                status = 'ABSENT'
                                message = f"Absent - Late arrival (more than 10 minutes late)"
                                tts_message = f"{name}, you are more than 10 minutes late. Unfortunately, you are marked as absent for this class."
                            else:
                                # Already has entry, trying to leave during class
                                if last_log['status'] in ['PRESENT']:
                                    status = 'BLOCKED'
                                    message = "Cannot leave during class hours. Please wait until class ends."
                                    tts_message = f"{name}, you cannot leave during class hours. Please wait until the class ends."
                                else:
                                    status = 'BLOCKED'
                                    message = "Already processed for this class."
                                    tts_message = f"{name}, you have already been processed for this class."
                                    
                        else:
                            # After class ends
                            if last_log and last_log['action'] == 'ENTRY' and last_log['status'] in ['PRESENT']:
                                action = 'EXIT'
                                status = 'EXIT'
                                message = "Class completed - Exit registered"
                                tts_message = f"Goodbye {name}! Have a great day. Your exit has been recorded."
                            elif last_log and last_log['action'] == 'EXIT':
                                status = 'BLOCKED'
                                message = "Exit already recorded for this class."
                                tts_message = f"{name}, your exit has already been recorded."
                            elif last_log and last_log['status'] == 'ABSENT':
                                status = 'BLOCKED'
                                message = "Already marked absent - no exit needed"
                                tts_message = f"{name}, you were marked absent for this class."
                            elif last_log and last_log['status'] == 'BLOCKED':
                                status = 'BLOCKED'
                                message = "Previous entry was blocked - no valid attendance recorded"
                                tts_message = f"{name}, your previous entry was blocked, so no exit is needed."
                            elif not last_log:
                                status = 'BLOCKED'
                                message = "No entry found for this class."
                                tts_message = f"{name}, no entry was recorded for this class. Please contact the teacher."
                            else:
                                # Debug: log what the last entry actually was
                                last_action = last_log['action'] if last_log else 'None'
                                last_status = last_log['status'] if last_log else 'None'
                                logger.info(f"Post-class recognition for {name}: last_action={last_action}, last_status={last_status}")
                                status = 'BLOCKED'
                                message = f"Invalid attendance state (Last: {last_action}/{last_status})"
                                tts_message = f"{name}, there's an issue with your attendance record. Please contact the teacher."

                        # Log every recognition attempt
                        db.log_attendance_action(student_id, session_id, action, conf, current_time, status, message)
                        # Only persist summary rows for meaningful attendance states
                        if status in ['PRESENT', 'ABSENT', 'EXIT']:
                            db.update_attendance_summary(student_id, session_id, action, current_time, active_session, status)
                        try:
                            from improved_tts import speak_message
                            speak_message(tts_message)
                        except Exception:
                            pass
                        logger.info(f"Recognized student: {name} (ID: {student_id}) - {status}: {message}")
                        results.append({
                            "status": "recognized",
                            "action": action,
                            "attendance_status": status,
                            "student": {
                                "name": name,
                                "id": student_id,
                                "confidence": float(conf)
                            },
                            "session": {
                                "id": session_id,
                                "name": active_session['name'],
                                "type": active_session['type']
                            },
                            "message": message,
                            "timestamp": current_time.isoformat(),
                            "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                        })
                        
                else:
                    # If the predicted ID doesn't exist in database, treat as unknown
                    display_conf = min(conf, 100.0)
                    results.append({
                        "status": "unknown",
                        "message": "Unknown person detected",
                        "confidence": float(display_conf),
                        "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
            else:
                # The confidence score can be a very large number for a clear non-match.
                display_conf = min(conf, 100.0)
                results.append({
                    "status": "unknown",
                    "message": "Unknown person detected",
                    "confidence": float(display_conf),
                    "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error in recognition: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/detect_face', methods=['POST'])
def detect_face_in_frame():
    """Detects a face in a single frame for real-time feedback"""
    try:
        if face_cascade is None or face_cascade.empty():
            return jsonify({"status": "error", "message": "Face detection model not loaded."}), 500

        data = request.get_json()
        if 'image' not in data:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert & preprocess
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = preprocess_frame(opencv_image)

        # Adaptive neighbors
        min_neighbors = adaptive_min_neighbors(gray)

        # Detect faces with more debugging
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=min_neighbors,
            minSize=DETECTION_MIN_SIZE
        )
        
        logger.info(f"Detected {len(faces)} raw faces: {faces.tolist() if len(faces) > 0 else 'None'}")

        if len(faces) > 0:
            selected = filter_and_select_face(faces, opencv_image.shape)
            if selected is None:
                resp = {"status": "no_face"}
                if request.args.get('debug') == '1':
                    resp["debug"] = {"faces_found": len(faces), "filtered_out": True}
                return jsonify(resp)
            x, y, w, h = selected
            logger.info(f"Selected face: x={x}, y={y}, w={w}, h={h}")
            response = {
                "status": "face_detected",
                "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            }
            # Optional debug info
            if request.args.get('debug') == '1':
                response["debug"] = {
                    "faces_found": len(faces),
                    "min_neighbors_used": min_neighbors,
                    "mean_brightness": float(np.mean(gray)),
                    "selected_face": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                }
            return jsonify(response)
        else:
            resp = {"status": "no_face"}
            if request.args.get('debug') == '1':
                resp["debug"] = {"min_neighbors_used": min_neighbors, "mean_brightness": float(np.mean(gray))}
            return jsonify(resp)

    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register new user endpoint"""
    try:
        data = request.get_json()
        enrollment_id = data.get('id')
        name = data.get('name')
        images = data.get('images', [])
        
        logger.info(f"Registration request: ID={enrollment_id}, Name={name}, Images={len(images)}")
        
        if not enrollment_id or not name:
            return jsonify({"status": "error", "message": "ID and name are required"}), 400
        
        if len(images) < 5:
            return jsonify({"status": "error", "message": f"At least 5 images are required, received {len(images)}"}), 400
        
        # Create directory for user images
        user_dir = os.path.join(trainimage_path, f"{enrollment_id}_{name}")
        os.makedirs(user_dir, exist_ok=True)
        
        saved_images = 0
        
        # Save images
        for i, img_data in enumerate(images):
            try:
                # Decode base64 image
                image_data = img_data.split(',')[1] if ',' in img_data else img_data
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert & preprocess
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = preprocess_frame(opencv_image)

                # Adaptive neighbors
                min_neighbors = adaptive_min_neighbors(gray) + 1  # Slightly stricter for saved dataset quality

                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=DETECTION_SCALE_FACTOR,
                    minNeighbors=min_neighbors,
                    minSize=DETECTION_MIN_SIZE
                )
                
                if len(faces) > 0:
                    # Take the largest face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Extract face region and resize
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))
                    
                    # Save the face image
                    filename = f"{name}_{enrollment_id}_{i+1}.jpg"
                    filepath = os.path.join(user_dir, filename)
                    cv2.imwrite(filepath, face_img)
                    saved_images += 1
                    
                    logger.debug(f"Saved face image {i+1}: {filename}")
                else:
                    logger.warning(f"No face detected in image {i+1}, skipping")
                    
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                continue  # Skip this image but continue with others
        
        if saved_images == 0:
            return jsonify({"status": "error", "message": "No valid face images could be saved"}), 400
        
        # Add student to database
        if not db.add_student(enrollment_id, name):
            return jsonify({"status": "error", "message": "User with this ID already exists"}), 400
        
        return jsonify({
            "status": "success", 
            "message": f"User {name} registered successfully with {saved_images} face images"
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model endpoint"""
    global training_in_progress
    
    if training_in_progress:
        return jsonify({"status": "error", "message": "Training already in progress"}), 400
    
    def train_async():
        global training_in_progress, recognizer
        training_in_progress = True
        try:
            logger.info("Starting model training...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(trainimagelabel_path), exist_ok=True)
            
            # Initialize LBPH Face Recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8,
                threshold=55  # Lower threshold for better discrimination
            )
            
            # Get all training images
            faces = []
            ids = []
            
            # Check if training directory exists
            if not os.path.exists(trainimage_path):
                logger.error(f"Training directory {trainimage_path} not found")
                training_in_progress = False
                return
            
            # Iterate through student folders
            for student_dir in os.listdir(trainimage_path):
                student_path = os.path.join(trainimage_path, student_dir)
                if not os.path.isdir(student_path):
                    continue
                
                # Extract ID from directory name
                try:
                    student_id = student_dir.split('_')[0]
                except:
                    logger.warning(f"Invalid directory name format: {student_dir}, skipping")
                    continue
                
                # Process all images in the student directory
                image_count = 0
                for img_file in os.listdir(student_path):
                    if not (img_file.endswith('.jpg') or img_file.endswith('.png')):
                        continue
                        
                    try:
                        img_path = os.path.join(student_path, img_file)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        # Add to training data
                        faces.append(face_img)
                        ids.append(int(student_id))
                        image_count += 1
                    except Exception as e:
                        logger.error(f"Error processing image {img_file}: {e}")
                
                logger.info(f"Added {image_count} training images for student {student_id}")
            
            # Train the model
            if len(faces) == 0:
                logger.error("No training images found")
                training_in_progress = False
                return
                
            logger.info(f"Training model with {len(faces)} images from {len(set(ids))} students")
            recognizer.train(faces, np.array(ids))
            recognizer.save(trainimagelabel_path)
            logger.info(f"Model trained and saved to {trainimagelabel_path}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
        finally:
            training_in_progress = False
    
    # Start training in background
    threading.Thread(target=train_async, daemon=True).start()
    
    return jsonify({
        "status": "training_started", 
        "message": "Model training has begun. This may take a few minutes."
    })

@app.route('/api/training-status', methods=['GET'])
def training_status():
    """Get training status"""
    model_exists = os.path.exists(trainimagelabel_path)
    model_size = os.path.getsize(trainimagelabel_path) if model_exists else 0
    model_date = datetime.datetime.fromtimestamp(
        os.path.getmtime(trainimagelabel_path)
    ).isoformat() if model_exists else None
    
    return jsonify({
        "training_in_progress": training_in_progress,
        "model_exists": model_exists,
        "model_size_kb": round(model_size / 1024, 2) if model_exists else 0,
        "model_last_modified": model_date,
        "students_registered": len(db.get_all_students()) if db is not None else 0
    })

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    """Get attendance records"""
    try:
        date_str = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
        target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        show_all = request.args.get('show_all', '0') == '1'
        
        attendance_records = db.get_attendance_summary(target_date)
        
        # Format for frontend compatibility
        formatted_records = []
        for record in attendance_records:
            first_entry = record["first_entry"]
            last_exit = record["last_exit"]
            # Normalize timestamp strings (can be 'YYYY-MM-DD HH:MM:SS[.ffffff]' or ISO with 'T')
            def extract_time(ts):
                if not ts:
                    return None
                ts_str = str(ts)
                # Replace ' ' with 'T' to standardize then split
                ts_norm = ts_str.replace(' ', 'T')
                try:
                    time_part = ts_norm.split('T')[1][:8]
                    return time_part
                except Exception:
                    return None
            entry_time = extract_time(first_entry)
            exit_time = extract_time(last_exit)
            if show_all or entry_time or exit_time:
                formatted_records.append({
                    "enrollment": record["enrollment_id"],
                    "name": record["name"],
                    "date": date_str,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "status": record["status"],
                    "duration_minutes": record["total_duration_minutes"]
                })
        
        return jsonify({"attendance": formatted_records})
        
    except Exception as e:
        logger.error(f"Error retrieving attendance: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/attendance/events', methods=['GET'])
def get_attendance_events():
    """Debug endpoint: raw events for a date (optional student filter)"""
    try:
        date_str = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
        target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        student_id = request.args.get('student_id')
        events = db.get_attendance_events(target_date, student_id)
        return jsonify({"events": events, "count": len(events)})
    except Exception as e:
        logger.error(f"Error retrieving attendance events: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/attendance/live', methods=['GET'])
def get_live_attendance():
    """Return attendance focused on the currently active session (if any).

    Response shape:
    {
      "mode": "session" | "daily",
      "session": { id, name, type, start_time, end_time } | null,
      "attendance": [
          { "student_id": str, "name": str, "entry_time": iso|None, "exit_time": iso|None, "status": str }
      ]
    }
    Falls back to daily summary if no active session.
    """
    try:
        active_session = db.get_active_session()
        if active_session:
            # Session-scoped attendance
            session_att = db.get_session_attendance(active_session['id'])
            records = []
            for r in session_att:
                records.append({
                    "student_id": r.get('student_id'),
                    "name": r.get('student_name'),
                    "entry_time": r.get('entry_time'),
                    "exit_time": r.get('exit_time'),
                    "status": r.get('status'),
                    "is_late": bool(r.get('is_late')),
                    "total_duration": r.get('total_duration')
                })
            return jsonify({
                "mode": "session",
                "session": {
                    "id": active_session['id'],
                    "name": active_session['name'],
                    "type": active_session['type'],
                    "room": active_session.get('room'),
                    "start_time": active_session['start_time'],
                    "end_time": active_session['end_time']
                },
                "attendance": records
            })
        else:
            # Fallback to daily summary used by legacy component
            date_str = request.args.get('date', datetime.date.today().strftime("%Y-%m-%d"))
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            attendance_records = db.get_attendance_summary(target_date)
            records = []
            for rec in attendance_records:
                first_entry = rec.get('first_entry')
                last_exit = rec.get('last_exit')
                records.append({
                    "student_id": rec.get('enrollment_id'),
                    "name": rec.get('name'),
                    "entry_time": first_entry,
                    "exit_time": last_exit,
                    "status": rec.get('status')
                })
            return jsonify({
                "mode": "daily",
                "session": None,
                "attendance": records
            })
    except Exception as e:
        logger.error(f"Error retrieving live attendance: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get all registered students with detailed information"""
    try:
        students = db.get_all_students() if db is not None else []
        # Format for frontend compatibility with detailed info
        formatted_students = []
        for student in students:
            # Count training images for this student.
            # Original implementation only looked for folder named exactly the enrollment_id,
            # but registration stores images in a folder named "<id>_<name>" (e.g. 123_john doe).
            # We now search for any directory in TrainingImage starting with '<id>_' OR a legacy exact-id directory.
            image_count = 0
            try:
                if os.path.exists(trainimage_path):
                    target_prefix = f"{student['enrollment_id']}_"
                    legacy_dir = os.path.join(trainimage_path, str(student['enrollment_id']))
                    matched_dir = None
                    # Prefer prefixed directory match
                    for d in os.listdir(trainimage_path):
                        d_path = os.path.join(trainimage_path, d)
                        if not os.path.isdir(d_path):
                            continue
                        if d.startswith(target_prefix):
                            matched_dir = d_path
                            break
                    # Fallback to legacy directory name (just the ID)
                    if matched_dir is None and os.path.isdir(legacy_dir):
                        matched_dir = legacy_dir
                    if matched_dir:
                        image_count = len([
                            f for f in os.listdir(matched_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))
                        ])
            except Exception as e:
                logger.warning(f"Error counting images for student {student.get('enrollment_id')}: {e}")
            
            formatted_students.append({
                "id": student["enrollment_id"],
                "enrollment_id": student["enrollment_id"],
                "name": student["name"],
                "created_at": student.get("created_at"),
                "image_count": image_count,
                # Legacy format for backward compatibility
                "Enrollment": student["enrollment_id"],
                "Name": student["name"]
            })
        return jsonify({"students": formatted_students})
    except Exception as e:
        logger.error(f"Error retrieving students: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/students/<enrollment_id>', methods=['DELETE'])
def delete_student(enrollment_id):
    """Delete a student (and training images)"""
    try:
        if db is None:
            return jsonify({"status": "error", "message": "Database not initialized"}), 500
        removed_images, ok = _delete_student_internal(enrollment_id)
        if not ok:
            return jsonify({"status": "error", "message": "Student not found"}), 404
        # Indicate model retrain advisable
        return jsonify({
            "status": "success",
            "message": f"Student {enrollment_id} deleted. {removed_images} training images removed. Retrain model to apply changes."})
    except Exception as e:
        logger.error(f"Error deleting student {enrollment_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def _delete_student_internal(enrollment_id: str):
    """Helper to delete student images + DB rows. Returns (removed_images, success)."""
    removed_images = 0
    try:
        if os.path.exists(trainimage_path):
            for d in os.listdir(trainimage_path):
                if d.startswith(f"{enrollment_id}_"):
                    full = os.path.join(trainimage_path, d)
                    for root, _, files in os.walk(full, topdown=False):
                        for f in files:
                            try:
                                os.remove(os.path.join(root, f))
                                removed_images += 1
                            except Exception:
                                pass
                        try:
                            os.rmdir(root)
                        except Exception:
                            pass
                    break
        ok = db.delete_student(enrollment_id)
        return removed_images, ok
    except Exception:
        return removed_images, False

@app.route('/api/students/bulk-delete', methods=['POST'])
def bulk_delete_students():
    """Bulk delete students: expects JSON {"ids": ["id1","id2",...]}."""
    try:
        if db is None:
            return jsonify({"status": "error", "message": "Database not initialized"}), 500
        data = request.get_json(force=True)
        ids = data.get('ids', []) if isinstance(data, dict) else []
        if not ids:
            return jsonify({"status": "error", "message": "No ids provided"}), 400
        summary = []
        total_images = 0
        deleted = 0
        for sid in ids:
            removed, ok = _delete_student_internal(str(sid))
            total_images += removed
            if ok:
                deleted += 1
            summary.append({"id": sid, "deleted": ok, "images_removed": removed})
        return jsonify({
            "status": "success",
            "requested": len(ids),
            "deleted": deleted,
            "total_images_removed": total_images,
            "details": summary,
            "message": f"Deleted {deleted}/{len(ids)} students. Retrain model to apply changes."})
    except Exception as e:
        logger.error(f"Bulk delete error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics - great for hackathon demo"""
    try:
        stats = db.get_statistics() if db is not None else {
            "total_students": 0,
            "present_today": 0,
            "events_today": 0,
            "attendance_rate": 0
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Session Management Endpoints
@app.route('/api/sessions', methods=['GET', 'POST'])
def sessions():
    """Manage sessions (classes/labs)"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            session_id = db.create_session(
                name=data['name'],
                session_type=data['type'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                room=data.get('room', ''),
                grace_period=data.get('grace_period', 15)
            )
            
            if session_id:
                logger.info(f"Created session: {data['name']} (ID: {session_id})")
                return jsonify({'status': 'success', 'session_id': session_id})
            else:
                return jsonify({'status': 'error', 'message': 'Failed to create session'}), 500
                
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # GET sessions
    try:
        sessions_list = db.get_all_sessions()
        return jsonify(sessions_list)
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/activate', methods=['POST'])
def activate_session(session_id):
    """Activate a specific session"""
    try:
        success = db.activate_session(session_id)
        if success:
            logger.info(f"Activated session ID: {session_id}")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to activate session'}), 500
    except Exception as e:
        logger.error(f"Error activating session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/deactivate', methods=['POST'])
def deactivate_session(session_id):
    """Deactivate a specific session"""
    try:
        with db.get_connection() as conn:
            conn.execute("UPDATE sessions SET is_active = 0 WHERE id = ?", (session_id,))
            conn.commit()
        logger.info(f"Deactivated session ID: {session_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error deactivating session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    try:
        success = db.delete_session(session_id)
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete session'}), 500
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<int:session_id>', methods=['PUT'])
def update_session(session_id):
    """Update a session"""
    try:
        data = request.get_json()
        success = db.update_session(
            session_id,
            name=data['name'],
            session_type=data['type'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            room=data.get('room', ''),
            grace_period=data.get('grace_period', 15)
        )
        
        if success:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update session'}), 500
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/next', methods=['GET'])
def get_next_session():
    """Get the next upcoming session"""
    try:
        next_session = db.get_next_session()
        return jsonify(next_session)
    except Exception as e:
        logger.error(f"Error retrieving next session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/active', methods=['GET'])
def get_active_session():
    """Get the currently active session"""
    try:
        active_session = db.get_active_session()
        return jsonify(active_session)
    except Exception as e:
        logger.error(f"Error retrieving active session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/attendance', methods=['GET'])
def get_session_attendance(session_id):
    """Get attendance for a specific session"""
    try:
        attendance = db.get_session_attendance(session_id)
        
        # Calculate statistics
        total_students = len(attendance)
        present_students = len([a for a in attendance if a['status'] in ['present', 'late']])
        late_students = len([a for a in attendance if a['is_late']])
        
        stats = {
            'total_students': total_students,
            'present_students': present_students,
            'attendance_rate': (present_students / total_students * 100) if total_students > 0 else 0,
            'late_students': late_students,
            'punctuality_rate': ((present_students - late_students) / total_students * 100) if total_students > 0 else 0
        }
        
        return jsonify({
            'attendance': attendance,
            'statistics': stats
        })
    except Exception as e:
        logger.error(f"Error retrieving session attendance: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/auto-detect-session', methods=['GET'])
def auto_detect_session():
    """Automatically detect and activate current session based on time"""
    try:
        from datetime import datetime
        current_time = datetime.now()
        
        # Get all sessions that should be active now
        sessions_list = db.get_all_sessions()
        current_sessions = []
        
        for session in sessions_list:
            start_time = datetime.fromisoformat(session['start_time'])
            end_time = datetime.fromisoformat(session['end_time'])
            
            if start_time <= current_time <= end_time:
                current_sessions.append(session)
        
        if current_sessions:
            # Activate the first matching session
            session = current_sessions[0]
            success = db.activate_session(session['id'])
            
            if success:
                logger.info(f"Auto-activated session: {session['name']}")
                return jsonify({
                    'status': 'auto_detected',
                    'session': session,
                    'message': f"Auto-activated: {session['name']}"
                })
        
        return jsonify({'status': 'no_session_found', 'message': 'No active sessions found for current time'})
        
    except Exception as e:
        logger.error(f"Error auto-detecting session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Initialize on startup
    logger.info("Initializing face recognition system...")
    initialize_recognizer()
    
    # Ensure directories exist
    os.makedirs(trainimage_path, exist_ok=True)
    os.makedirs(os.path.dirname(trainimagelabel_path), exist_ok=True)
    
    logger.info("Starting Flask API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
