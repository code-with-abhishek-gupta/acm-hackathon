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
    """Face recognition endpoint"""
    try:
        if recognizer is None:
            return jsonify({"status": "error", "message": "Model not loaded. Please train the model first."}), 400
        
        if face_cascade is None or face_cascade.empty():
            return jsonify({"status": "error", "message": "Face detection model not loaded. Please check Haar cascade file."}), 400
        
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
            
            # Threshold for recognition - lower value means better match (adjusted to be more lenient)
            if conf < 65:  # This threshold value is critical for recognition accuracy
                # Find student in database
                student = db.get_student(str(id_pred))
                
                if student:
                    name = student["name"]
                    logger.info(f"Recognized student: {name} (ID: {id_pred}) with confidence: {conf}")
                    
                    # Handle attendance logic (entry/exit)
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    
                    # Check last event for this student today
                    last_event = db.get_student_last_event(str(id_pred))
                    
                    if not last_event or last_event["event_type"] == "EXIT":
                        # First time today or last was exit - mark entry
                        db.record_attendance_event(str(id_pred), "ENTRY", conf)
                        message = f"Welcome, {name}! Entry marked at {time_str}"
                        status = "entry_marked"
                    else:
                        # Last was entry - mark exit
                        db.record_attendance_event(str(id_pred), "EXIT", conf)
                        message = f"Goodbye, {name}! Exit marked at {time_str}"
                        status = "exit_marked"
                    
                    results.append({
                        "status": status,
                        "name": name,
                        "id": str(id_pred),
                        "confidence": float(conf),
                        "message": message,
                        "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
                else:
                    results.append({
                        "status": "unknown_id",
                        "message": f"ID {id_pred} not found in database",
                        "confidence": float(conf),
                        "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                    })
            else:
                # The confidence score can be a very large number for a clear non-match.
                # We cap it at 100.0 for a cleaner and more user-friendly display.
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

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get all registered students"""
    try:
        students = db.get_all_students() if db is not None else []
        # Format for frontend compatibility
        formatted_students = []
        for student in students:
            formatted_students.append({
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

if __name__ == '__main__':
    # Initialize on startup
    logger.info("Initializing face recognition system...")
    initialize_recognizer()
    
    # Ensure directories exist
    os.makedirs(trainimage_path, exist_ok=True)
    os.makedirs(os.path.dirname(trainimagelabel_path), exist_ok=True)
    
    logger.info("Starting Flask API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
