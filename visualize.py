import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

try:
    results = pd.read_csv('./test_interpolated.csv')
    print("CSV file loaded successfully")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

video_path = './sample.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    car_data = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': car_data['license_number'].iloc[0]
    }

    frame_number = car_data['frame_nmr'].iloc[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame number: {frame_number} for car_id: {car_id}")
        continue

    try:
        x1, y1, x2, y2 = ast.literal_eval(car_data['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    except Exception as e:
        print(f"Error parsing bounding box for car_id {car_id}: {e}")
        continue

    try:
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        license_plate[car_id]['license_crop'] = license_crop
    except Exception as e:
        print(f"Error cropping/resizing license plate for car_id {car_id}: {e}")

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    frame_nmr += 1
    if not ret:
        break

    df_ = results[results['frame_nmr'] == frame_nmr]
    for _, row in df_.iterrows():
        try:
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25, line_length_x=200, line_length_y=200)

            x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            car_id = row['car_id']
            license_crop = license_plate[car_id]['license_crop']
            if license_crop is None:
                print(f"No license crop for car_id {car_id} at frame {frame_nmr}")
                continue
            H, W, _ = license_crop.shape
            license_number = license_plate[car_id]['license_plate_number']

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
                frame[int(car_y1) - H - 400:int(car_y1) - H - 100, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(license_number, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                cv2.putText(frame, license_number, (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))), cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
            except Exception as e:
                print(f"Error overlaying license plate and text for car_id {car_id} at frame {frame_nmr}: {e}")
        except Exception as e:
            print(f"Error processing car_id {row['car_id']} at frame {frame_nmr}: {e}")

    out.write(frame)
    frame_resized = cv2.resize(frame, (1280, 720))

    cv2.imshow('frame', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
