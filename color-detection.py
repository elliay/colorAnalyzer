import io
import os
import cv2
import numpy as np
from google.cloud import vision
from PIL import Image, ImageDraw
from scipy.stats import mode

# Load the image into memory
image_path = 'images/test.jpg'
with io.open(image_path, 'rb') as image_file:
    image_content = image_file.read()

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
# bgr_values = np.array(rgb_values, dtype=np.uint8)
bgr_values = np.array([[color[2], color[1], color[0]] for color in rgb_values], dtype=np.float32)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("bgr_values:", bgr_values)
print("bgr_values shape:", bgr_values.shape)
print("bgr_values dtype:", bgr_values.dtype)

# Perform KMeans clustering to group similar colors
kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(bgr_values, 5, None, kmeans_criteria, 10, cv2.KMEANS_PP_CENTERS)
print("kmeans_criteria:", kmeans_criteria)

kmeans_flags = cv2.KMEANS_PP_CENTERS
print("kmeans_flags:", kmeans_flags)

kmeans_attempts = 10
print("kmeans_attempts:", kmeans_attempts)

kmeans = cv2.kmeans(bgr_values, 5, None, criteria=kmeans_criteria, attempts=kmeans_attempts, flags=kmeans_flags)


# Perform KMeans clustering to group similar colors
# kmeans = cv2.kmeans(bgr_values, 5, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

# Get cluster centers
centers = kmeans[2]

# Create an annotated image
image_cv2 = cv2.imread(image_path)
image_width, image_height, _ = image_cv2.shape

# Create a PIL ImageDraw object
pil_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_image)

# Draw bounding boxes around clusters
image_area = image_width * image_height

# Draw bounding boxes around clusters
for center in centers:
    center_rgb = [int(center[2]), int(center[1]), int(center[0])]  # Convert BGR to RGB
    lower_bound = np.array([max(0, c - 20) for c in center_rgb], dtype=np.uint8)
    upper_bound = np.array([min(255, c + 20) for c in center_rgb], dtype=np.uint8)
    mask = cv2.inRange(image_cv2, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    for contour in contours:
        area = cv2.contourArea(contour)


        # Exclude the largest contour
        if area < .90*image_area and area > .1*image_area:
            print("area of box is: " + str(area))
            print("image area is: " + str(image_area))
            x, y, w, h = cv2.boundingRect(contour)
            draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=5)

            box_pixels = image_cv2[y:y+h, x:x+w]
            mode_color = np.median(box_pixels, axis=(0, 1)).astype(int)
            print("Mode color RGB values:", mode_color)

# Display the annotated image
# pil_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
pil_image.show()
