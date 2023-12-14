from flask import Flask, render_template, request, send_file
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
from google.cloud import vision
from scipy.stats import mode
import base64

app = Flask(__name__, static_url_path='/static', static_folder='static')
id = 1

def generate_unique_id():
    id += 1
    return id

def detect_color(image_content, image_cv2, id):

    image = vision.Image(content=image_content)

    # Create a Cloud Vision API client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client-file.json'
    client = vision.ImageAnnotatorClient()

    # Detect the dominant colors
    response = client.image_properties(image=image)
    image_properties = response.image_properties_annotation
    colors = image_properties.dominant_colors.colors[:10]

    # Extract RGB values from dominant colors
    rgb_values = [
        [
            int(color.color.red * 255),
            int(color.color.green * 255),
            int(color.color.blue * 255)
        ]
        for color in colors
    ]

    # Convert RGB values to OpenCV format (BGR)
    bgr_values = np.array([[color[2], color[1], color[0]] for color in rgb_values], dtype=np.float32)

    # Perform KMeans clustering to group similar colors
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(bgr_values, 5, None, kmeans_criteria, 10, cv2.KMEANS_PP_CENTERS)

    kmeans_flags = cv2.KMEANS_PP_CENTERS
    kmeans_attempts = 10
    kmeans = cv2.kmeans(bgr_values, 5, None, criteria=kmeans_criteria, attempts=kmeans_attempts, flags=kmeans_flags)
    centers = kmeans[2]

    image_width, image_height, _ = image_cv2.shape
    pil_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    image_area = image_width * image_height

    # Draw bounding boxes around clusters
    mode_colors = {}
    bounding_boxes_info = {}
    for center in centers:
        center_rgb = [int(center[2]), int(center[1]), int(center[0])]
        lower_bound = np.array([max(0, c - 20) for c in center_rgb], dtype=np.uint8)
        upper_bound = np.array([min(255, c + 20) for c in center_rgb], dtype=np.uint8)
        mask = cv2.inRange(image_cv2, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            # Exclude the largest contour
            if area < .90*image_area and area > .1*image_area:
                id += 1
                box_id = id
                x, y, w, h = cv2.boundingRect(contour)

                bounding_boxes_info[box_id] = [x,y,w,h]

                draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=5)

                box_pixels = image_cv2[y:y+h, x:x+w]
                mode_color = np.median(box_pixels, axis=(0, 1)).astype(int)
                mode_color = mode_color.tolist()
                mode_colors[box_id] = mode_color

    return pil_image, bounding_boxes_info, mode_colors, id


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/annotate', methods=['POST'])
def annotate():
    global id
    if 'image' in request.files:
        image = request.files['image'].read()
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        pil_image, bounding_boxes_info, mode_colors, id = detect_color(image,image_cv2, id)

        annotated_image = io.BytesIO()
        pil_image.save(annotated_image, format='PNG')
        annotated_image = annotated_image.getvalue()

        encoded_image = base64.b64encode(annotated_image).decode('utf-8')

        return {'image': encoded_image, 'bounding_boxes':bounding_boxes_info, 'mode_colors':mode_colors}

    return 'Invalid request', 400

if __name__ == '__main__':
    app.run(debug=True)