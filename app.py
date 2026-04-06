import streamlit as st   # For creating web app UI
import cv2              # OpenCV for image processing
import numpy as np      # For handling arrays (images)
from PIL import Image   # For opening uploaded images

# Set page title and layout
st.set_page_config(page_title="Photo Editor", layout="wide")

# Title on web page
st.title("🖼️ Photo Editor using OpenCV & Streamlit")

# Upload image from user
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# If user uploads image
if uploaded_file is not None:

    # Open image using PIL
    image = Image.open(uploaded_file)

    # Convert image to numpy array and RGB → BGR (OpenCV format)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Show original image
    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Reset button (reload app)
    if st.sidebar.button("🔄 Reset All"):
        st.experimental_rerun()

    # ---------------- Resize ----------------
    st.sidebar.header("Resize")

    # Slider to change width and height
    width = st.sidebar.slider("Width", 100, 1500, img.shape[1])
    height = st.sidebar.slider("Height", 100, 1500, img.shape[0])

    # Resize image
    edited = cv2.resize(img, (width, height))

    # ---------------- Brightness & Contrast ----------------
    st.sidebar.header("Brightness & Contrast")

    # Slider for brightness (-100 to 100)
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)

    # Slider for contrast (0.5 to 3.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

    # Apply brightness and contrast
    edited = cv2.convertScaleAbs(edited, alpha=contrast, beta=brightness)

    # ---------------- Filters ----------------
    st.sidebar.header("Filters")

    # Dropdown to select filter
    filter_option = st.sidebar.selectbox(
        "Choose Filter",
        ["None", "Grayscale", "Blur", "Sharpen", "Warm"]
    )

    # Convert to grayscale
    if filter_option == "Grayscale":
        edited = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    # Apply blur effect
    elif filter_option == "Blur":
        blur_value = st.sidebar.slider("Blur Strength", 1, 25, 5)

        # Kernel must be odd number
        if blur_value % 2 == 0:
            blur_value += 1

        edited = cv2.GaussianBlur(edited, (blur_value, blur_value), 0)

    # Apply sharpen effect
    elif filter_option == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])   # Sharpen kernel
        edited = cv2.filter2D(edited, -1, kernel)

    # Apply warm filter (increase red tones)
    elif filter_option == "Warm":

        # Create lookup table for red increase
        increase_lookup = np.interp(
            np.arange(256),
            [0, 64, 128, 256],
            [0, 80, 160, 256]
        ).astype('uint8')

        # Split channels
        b, g, r = cv2.split(edited)

        # Apply lookup to red channel
        r = cv2.LUT(r, increase_lookup)

        # Merge channels back
        edited = cv2.merge((b, g, r))

    # ---------------- Portrait Blur ----------------
    st.sidebar.header("Portrait Effect")

    # Checkbox to enable portrait blur
    if st.sidebar.checkbox("Enable Portrait Blur"):

        # Load face detection model
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Convert to grayscale for detection
        gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Create black mask (same size as image)
        mask = np.zeros(edited.shape[:2], dtype=np.uint8)

        # Mark face area as white
        for (x, y, w, h) in faces:
            mask[y:y+h, x:x+w] = 255

        # Convert mask to 3 channels
        mask_3ch = cv2.merge([mask, mask, mask])

        # Blur full image
        blur_bg = cv2.GaussianBlur(edited, (35, 35), 0)

        # Keep face clear, blur background
        edited = np.where(mask_3ch == 255, edited, blur_bg)

    # ---------------- Optional Effects ----------------
    st.sidebar.header("Optional Effects")

    # Dropdown for extra effects
    option = st.sidebar.selectbox(
        "Extra Effect",
        ["None", "Edge Detection", "Cartoon", "Rotate"]
    )

    # Edge detection
    if option == "Edge Detection":
        edited = cv2.Canny(edited, 100, 200)

    # Cartoon effect
    elif option == "Cartoon":
        gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        edges = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )

        color = cv2.bilateralFilter(edited, 9, 250, 250)

        edited = cv2.bitwise_and(color, color, mask=edges)

    # Rotate image
    elif option == "Rotate":
        angle = st.sidebar.slider("Angle", 0, 360, 0)

        (h, w) = edited.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        edited = cv2.warpAffine(edited, matrix, (w, h))

    # ---------------- Display Output ----------------
    st.subheader("Edited Image")

    # If grayscale image
    if len(edited.shape) == 2:
        st.image(edited, clamp=True)
    else:
        # Convert BGR → RGB for display
        st.image(cv2.cvtColor(edited, cv2.COLOR_BGR2RGB))

    # ---------------- Download ----------------
    st.subheader("Download")

    # Convert image before saving
    if len(edited.shape) == 2:
        final_img = edited
    else:
        final_img = cv2.cvtColor(edited, cv2.COLOR_BGR2RGB)

    # Encode image to PNG
    _, buffer = cv2.imencode(".png", final_img)

    # Download button
    st.download_button(
        label="📥 Download Image",
        data=buffer.tobytes(),
        file_name="edited_image.png",
        mime="image/png"
    )

# If no image uploaded
else:
    st.info("👆 Please upload an image to start editing")