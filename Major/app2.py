from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import numpy as np
import threading
import time
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import dropbox
from io import BytesIO
import requests
app = Flask(__name__)

# Dropbox Configuration
DROPBOX_APP_KEY = "sh6pythii5dauob"
DROPBOX_APP_SECRET = "cb6t4hj28vrxcqb"
DROPBOX_REFRESH_TOKEN = "iuzx56ItXBcAAAAAAAAAAYgRBuNOvdUF2_WR_QmlMXmk0CPqiImsOzx5F424KW7A"
DROPBOX_ATTENDANCE_PATH = "/attendance/Attendance.xlsx"
DROPBOX_IMAGE_FOLDER = "/attendance/ImageData/"

# Global Variables
tracking_active = False
cap_entry = None
cap_exit = None
attendance_status = {}



def get_dropbox_access_token():
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
    }
    auth = (DROPBOX_APP_KEY, DROPBOX_APP_SECRET)
    
    response = requests.post(url, data=data, auth=auth)
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to refresh token: {response.json()}")

# Get a new access token
DROPBOX_ACCESS_TOKEN = get_dropbox_access_token()

# Initialize Dropbox with new access token
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


#####################################
# Ensure Temp Directory Exists
TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

#####################################
# Function to download images from Dropbox
def download_images_from_dropbox():
    try:
        entries = dbx.files_list_folder(DROPBOX_IMAGE_FOLDER).entries
        os.makedirs("ImageData", exist_ok=True)
        for entry in entries:
            file_path = os.path.join("ImageData", entry.name)
            metadata, res = dbx.files_download(DROPBOX_IMAGE_FOLDER + entry.name)
            with open(file_path, "wb") as f:
                f.write(res.content)
        print("✅ Images downloaded from Dropbox")
    except Exception as e:
        print("❌ Dropbox Image Download Error:", e)

download_images_from_dropbox()

#####################################
# Function to mark attendance
def mark_attendance(name, status):
    global attendance_status
    print(f"📌 DEBUG: attendance_status before: {attendance_status}")  # Debugging
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    if status == 'Entry' and attendance_status.get(name) != 'Inside':
        attendance_status[name] = 'Inside'
        log_attendance(name, 'Entry', timestamp)
    elif status == 'Exit' and attendance_status.get(name) == 'Inside':
        attendance_status[name] = 'Outside'
        log_attendance(name, 'Exit', timestamp)
    print(f"📌 DEBUG: attendance_status after: {attendance_status}")

#####################################
# Function to log attendance in Excel
def log_attendance(name, status, timestamp):
    try:
        metadata, res = dbx.files_download(DROPBOX_ATTENDANCE_PATH)
        df = pd.read_excel(BytesIO(res.content))
    except:
        df = pd.DataFrame(columns=['Name', 'Status', 'Timestamp'])
    
    new_record = {'Name': name, 'Status': status, 'Timestamp': timestamp}
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    dbx.files_upload(output.read(), DROPBOX_ATTENDANCE_PATH, mode=dropbox.files.WriteMode("overwrite"))
    print("✅ Attendance file uploaded to Dropbox")

#####################################
# Function to run face recognition and track attendance
def run_tracking():
    global tracking_active, cap_entry, cap_exit, attendance_status
    cap_entry = cv2.VideoCapture(0)
    cap_exit = cv2.VideoCapture(1)
    while tracking_active:
        for cam, cam_name in [(cap_entry, 'Entry'), (cap_exit, 'Exit')]:
            success, frame = cam.read()
            if success:
                result = DeepFace.find(frame, db_path="ImageData", enforce_detection=False)
                if result and not result[0].empty:
                    name = os.path.splitext(os.path.basename(result[0]['identity'][0]))[0].upper()
                    mark_attendance(name, cam_name)
        time.sleep(3)
    cap_entry.release()
    cap_exit.release()
    cv2.destroyAllWindows()

#####################################
# Start Tracking Route
@app.route('/start_tracking')
def start_tracking():
    global tracking_active, attendance_status
    print(f"📌 DEBUG: attendance_status at start: {attendance_status}")  # Debugging
    if not tracking_active:
        tracking_active = True
        threading.Thread(target=run_tracking).start()
    return jsonify({"status": "Tracking Started"})

#####################################
# Stop Tracking Route
@app.route('/stop_tracking')
def stop_tracking():
    global tracking_active
    tracking_active = False
    return jsonify({"status": "Tracking Stopped"})

#####################################
# Stream Camera Feed
def generate_frames(camera):
    while tracking_active:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    if camera:
        camera.release()

#####################################
# Video Feed Routes
@app.route('/video_feed/<camera>')
def video_feed(camera):
    global cap_entry, cap_exit
    if camera == "entry":
        return Response(generate_frames(cap_entry), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif camera == "exit":
        return Response(generate_frames(cap_exit), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not found", 404

#####################################
# Add Face Data
@app.route("/add_face", methods=["POST"])
def add_face():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files["file"]
        name = request.form.get("name")
        if not name or file.filename == "":
            return jsonify({"success": False, "error": "Missing name or file"}), 400
        filename = f"{name}.jpg"
        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)
        print(f"📸 Image saved locally at {temp_path}")
        dropbox_path = f"{DROPBOX_IMAGE_FOLDER}{filename}"
        with open(temp_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
        os.remove(temp_path)
        return jsonify({"success": True, "message": "Face added successfully"})
    except Exception as e:
        print(f"⚠️ Error in /add_face: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

#####################################
# Home Page
@app.route('/')
def index():
    return render_template('index.html')

#####################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

