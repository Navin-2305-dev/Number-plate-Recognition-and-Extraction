import string
import easyocr
import csv

reader = easyocr.Reader(['en'], gpu=False)

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
                         'license_plate_bbox_score', 'license_number', 'license_number_score'])

        for frame_nmr, cars in results.items():
            for car_id, data in cars.items():
                if 'car' in data and 'license_plate' in data:
                    car_bbox = data['car']['bbox']
                    lp_bbox = data['license_plate']['bbox']
                    lp_text = data['license_plate']['text']
                    lp_bbox_score = data['license_plate']['bbox_score']
                    lp_text_score = data['license_plate']['text_score']
                    
                    writer.writerow([frame_nmr, car_id, f'[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]',
                                     f'[{lp_bbox[0]} {lp_bbox[1]} {lp_bbox[2]} {lp_bbox[3]}]', lp_bbox_score,
                                     lp_text, lp_text_score])

def license_complies_format(text):
    if len(text) != 7:
        return False

    if all([(text[i] in string.ascii_uppercase or text[i] in dict_int_to_char) if i in [0, 1, 4, 5, 6] else
            (text[i] in '0123456789' or text[i] in dict_char_to_int) for i in range(7)]):
        return True
    return False

def format_license(text):
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}

    formatted_text = ''.join([mapping[i].get(text[i], text[i]) for i in range(7)])
    return formatted_text

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for bbox, text, score in detections:
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1