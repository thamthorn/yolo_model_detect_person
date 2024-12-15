import cv2
import time
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# Google Sheets and Drive setup
def setup_google_sheet_and_drive(sheet_id, worksheet_name):
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('global-harmony-442806-r3-3ca7eeaa89e3.json', scope)
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    drive_service = build('drive', 'v3', credentials=credentials)
    return worksheet, drive_service


# Initialize Google Sheet and Drive
SHEET_ID = '1YRwQumypfdpou0Zm4k70Xy7jJ1BmT-tHclBRXyaVBXQ'
WORKSHEET_NAME = 'log'
worksheet, drive_service = setup_google_sheet_and_drive(SHEET_ID, WORKSHEET_NAME)

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize DeepSort tracker
tracker = DeepSort(max_age=5)

# Use webcam or ESP32 as the video source
esp32_ip = 'http://172.20.10.2/mjpeg/1'
cap = cv2.VideoCapture(esp32_ip)

if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

unique_ids = set()
processed_ids = set()
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)
    detections = []
    for result in results[0].boxes:
        if int(result.cls[0]) == 0:  # Class 0 is 'person'
            x1, y1, x2, y2 = map(float, result.xyxy[0])
            conf = float(result.conf[0])
            width, height = x2 - x1, y2 - y1
            detections.append([x1, y1, width, height, conf])

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

        if track_id not in processed_ids:
            processed_ids.add(track_id)

            image_path = f'detected_person_{track_id}.jpg'
            cv2.imwrite(image_path, frame)

            file_metadata = {'name': os.path.basename(image_path)}
            media = MediaFileUpload(image_path, mimetype='image/jpeg')
            file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            drive_service.permissions().create(
                fileId=file.get('id'),
                body={'type': 'anyone', 'role': 'reader'}
            ).execute()

            file = drive_service.files().get(fileId=file.get('id'), fields='webContentLink').execute()
            image_link = file.get('webContentLink')

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([timestamp, image_link])

            os.remove(image_path)

        bbox = track.to_ltwh()
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        unique_ids.add(track_id)

    cv2.imshow('YOLOv8 + DeepSort', frame)

    if time.time() - start_time > 60:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

number_of_person = len(unique_ids)
print(f'Number of unique people detected: {number_of_person}')

worksheet.append_row([time.strftime("%Y-%m-%d %H:%M:%S"), f'Total People: {number_of_person}'])

cap.release()
cv2.destroyAllWindows()
