# Face Recognition Attendance System (Web API Version)

A modern attendance management system using face recognition technology. This version is designed to work with a web frontend only.

## Features

- Face detection and recognition using OpenCV and LBPH algorithm
- REST API for frontend integration
- Registration of new students with face capture
- Automatic attendance marking with entry/exit tracking
- Real-time recognition with confidence score display

## System Requirements

- Python 3.6+
- OpenCV with contrib modules
- Flask for API
- React-based frontend
- Webcam for face capture

## Installation

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Setup project structure:

```
├── api_web_only.py         # Main API server
├── haarcascade_frontalface_default.xml  # Face detection model
├── requirements.txt        # Python dependencies
├── TrainingImage/          # Stores training images (auto-created)
├── TrainingImageLabel/     # Stores trained model (auto-created)
├── StudentDetails/         # Stores student records (auto-created)
├── Attendance/             # Stores attendance records (auto-created)
└── frontend/               # React frontend
```

3. Start the backend server:

```bash
python api_web_only.py
```

4. Start the frontend (from the frontend directory):

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `GET /api/health` - Check API status
- `POST /api/register` - Register new student with face images
- `POST /api/recognize` - Recognize faces in an image
- `POST /api/train` - Train the recognition model
- `GET /api/training-status` - Check training status
- `GET /api/attendance` - Get attendance records
- `GET /api/students` - Get registered students

## Usage

1. **Register New Users**: 
   - Navigate to the registration page
   - Enter ID and name
   - Allow the system to capture 50 face images from different angles
   - Submit registration

2. **Train the Model**:
   - After registering users, train the model
   - Wait for training to complete

3. **Take Attendance**:
   - Navigate to the attendance page
   - The system will automatically recognize faces
   - Entry/exit times are recorded automatically

## Technical Notes

- Face recognition uses LBPH (Local Binary Pattern Histograms) algorithm
- Recognition threshold is set to 55 (adjust as needed)
- Lower confidence values indicate better face matches
- For best results, ensure good lighting and clear face visibility

## Files You Can Remove

The following files from the original project are not needed for the web-only version:

- `automaticAttedance.py`
- `automaticAttedance_fixed.py`
- `attendance.py`
- `takeImage.py`
- `takeImage_fixed.py`
- `takemanually.py`
- `attendance_fixed_new.py`
- `trainImage.py` (functionality integrated into api_web_only.py)
- `show_attendance.py`
- `test.py`
- `test_face_detection.py`
- `AMS.ico`

## Troubleshooting

- **Recognition Issues**: Adjust confidence threshold in the API (currently set to 55)
- **Registration Problems**: Ensure good lighting and clear face visibility
- **Training Errors**: Make sure each student has multiple face images from different angles

## License

MIT License
