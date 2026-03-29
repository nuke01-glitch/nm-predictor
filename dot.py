# --- RUN THIS SCRIPT LOCALLY ONCE ---
import numpy as np
import json
from PIL import Image

# 1. Load your preprocessed Dot Matrix image (B&W)
# It should be small, like 64x64 or 128x128 pixels.
img_path = r"C:\Users\acer\Documents\nanomaterial project final\dot_image.jpeg" # Update this
img = Image.open(img_path).convert('L') # Convert to Grayscale
img_data = np.array(img)

# 2. Extract Dot coordinates (X, Y)
dots = []
width, height = img.size
threshold = 128 # Define what counts as a 'dot' (black or white)

for y in range(height):
    for x in range(width):
        # Assuming black dots on a white background (dots are near 0)
        if img_data[y, x] < threshold:
            # Shift coordinates so (0,0) is in the center
            # and flip Y (because image coords are top-down)
            real_x = (x / width - 0.5) * 4 # Adjust multiplier (4) for size
            real_y = (0.5 - y / height) * 4
            dots.append({'x': real_x, 'y': real_y, 'z': 0}) # Initial flat profile

# 3. Add a small random depth (Z) to make it feel 3D
for dot in dots:
    dot['z'] = (np.random.random() - 0.5) * 0.5

# 4. Save the data to a JSON file
with open('my_dots_data.json', 'w') as f:
    json.dump(dots, f)

print(f"✅ Extracted {len(dots)} dot coordinates. Save 'my_dots_data.json' in your project folder.")