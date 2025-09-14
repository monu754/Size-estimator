import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Helper Functions ---

def detect_objects(pil_image):
    """
    Detects contours in the image using OpenCV.
    Returns the image with numbered contours drawn on it and a map of contours.
    """
    # Convert PIL Image to an OpenCV image
    image = np.array(pil_image.convert('RGB'))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Pre-processing for contour detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    dilated = cv2.dilate(edged, None, iterations=1)
    eroded = cv2.erode(dilated, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small, noisy contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]

    # Sort contours from left-to-right to maintain a consistent order
    valid_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    
    output_image = image_bgr.copy()
    contour_map = {}

    # Draw and number the valid contours
    for i, c in enumerate(valid_contours):
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box).astype("int")
        cv2.drawContours(output_image, [box], -1, (0, 255, 0), 2)
        
        # Get the top-left corner of the bounding box to place the number
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.putText(output_image, str(i + 1), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        contour_map[i + 1] = c
        
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), contour_map

# --- Streamlit UI ---

st.set_page_config(page_title="Object Size Detector", layout="wide")
st.title("📏 Object Size Detector")
st.write("---")

# Initialize session state for multi-step workflow
if 'stage' not in st.session_state:
    st.session_state.stage = 'capture'
    st.session_state.image = None
    st.session_state.contours = {}

# --- Sidebar for controls and instructions ---
with st.sidebar:
    st.header("Instructions & Controls")
    
    if st.session_state.stage == 'capture':
        st.info("Use the main panel to either take a live picture or upload an image. Place objects against a plain background for best results.")
    
    elif st.session_state.stage == 'select':
        st.info("The app has detected objects. Now, provide details about your reference object and select which ones to measure.")
        
        # User specifies if they are providing width or height
        ref_dimension_type = st.radio(
            "1. Known dimension of your reference object:",
            ('Width (shorter side)', 'Height (longer side)'),
            key='ref_dim_type',
            horizontal=True
        )

        # User enters the known dimension value
        ref_dimension_value = st.number_input(
            f"2. Enter Reference Object's {ref_dimension_type.split(' ')[0]} (cm)",
            min_value=0.1, max_value=1000.0, value=2.5, step=0.1
        )

        # Dropdown to select the reference object
        ref_options = list(st.session_state.contours.keys())
        ref_selection = st.selectbox("3. Select the Reference Object Number", options=ref_options)

        # Multiselect for target objects
        obj_options = [k for k in ref_options if k != ref_selection]
        obj_selections = st.multiselect("4. Select Target Object Numbers (Max 2)", options=obj_options, max_selections=2)

        if st.button("Calculate Dimensions"):
            st.session_state.ref_dimension_type = ref_dimension_type
            st.session_state.ref_dimension_value = ref_dimension_value
            st.session_state.ref_contour = st.session_state.contours[ref_selection]
            st.session_state.obj_contours = [st.session_state.contours[s] for s in obj_selections]
            st.session_state.stage = 'results'
            st.rerun()

    if st.button("Start Over"):
        # Reset the state completely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Main Content Panel ---
if st.session_state.stage == 'capture':
    st.header("Step 1: Provide an Image")
    
    tab1, tab2 = st.tabs(["📷 Take a Photo", "📂 Upload an Image"])
    img_file_buffer = None
    
    with tab1:
        img_file_buffer = st.camera_input("Take a picture", key="camera_input")

    with tab2:
        img_file_buffer = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    if img_file_buffer:
        image = Image.open(img_file_buffer)
        st.session_state.image = image
        
        # Immediately run detection and store contours before switching stage
        _, contours = detect_objects(image)
        st.session_state.contours = contours
        
        st.session_state.stage = 'select'
        st.rerun()

elif st.session_state.stage == 'select':
    st.header("Step 2: Select Your Objects")
    if st.session_state.image:
        # We already ran detection, so now we just draw the numbered image
        numbered_img, _ = detect_objects(st.session_state.image)
        
        if not st.session_state.contours:
            st.error("No objects were detected. Please try another picture with better lighting or a simpler background.")
            st.session_state.stage = 'capture'
        else:
            st.image(numbered_img, caption="Detected Objects - Choose your reference object from the sidebar", use_column_width=True)

elif st.session_state.stage == 'results':
    st.header("Step 3: Measurement Results")
    
    image = np.array(st.session_state.image.convert('RGB'))
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    pixels_per_metric = None
    
    # --- Process Reference Object ---
    ref_contour = st.session_state.ref_contour
    ref_box = cv2.minAreaRect(ref_contour)
    ref_box_pts = cv2.boxPoints(ref_box).astype("int")

    # CORRECTED LOGIC: Get width and height from the rotated rectangle
    (ref_w_px, ref_h_px) = ref_box[1]
    
    # To be consistent, let's define width as the smaller side and height as the larger side
    ref_pixel_width = min(ref_w_px, ref_h_px)
    ref_pixel_height = max(ref_w_px, ref_h_px)

    # Determine the pixels-per-metric scale based on user's choice
    if st.session_state.ref_dimension_type == 'Width (shorter side)':
        if ref_pixel_width > 0: pixels_per_metric = ref_pixel_width / st.session_state.ref_dimension_value
    else: # 'Height (longer side)'
        if ref_pixel_height > 0: pixels_per_metric = ref_pixel_height / st.session_state.ref_dimension_value
    
    # Draw reference box
    (x, y, w, h) = cv2.boundingRect(ref_contour) # Still use this for text placement
    cv2.drawContours(output_image, [ref_box_pts], -1, (255, 165, 0), 2)
    cv2.putText(output_image, "Reference", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # --- Process Target Objects ---
    if pixels_per_metric and st.session_state.obj_contours:
        for contour in st.session_state.obj_contours:
            obj_box = cv2.minAreaRect(contour)
            obj_box_pts = cv2.boxPoints(obj_box).astype("int")
            
            # CORRECTED LOGIC: Get width and height from the rotated rectangle
            (obj_w_px, obj_h_px) = obj_box[1]
            
            # Calculate real-world dimensions consistently
            dim_w = min(obj_w_px, obj_h_px) / pixels_per_metric
            dim_h = max(obj_w_px, obj_h_px) / pixels_per_metric
            
            # Draw target box and dimensions
            (x, y, w, h) = cv2.boundingRect(contour) # Still use this for text placement
            cv2.drawContours(output_image, [obj_box_pts], -1, (0, 255, 0), 2)
            cv2.putText(output_image, f"W: {dim_w:.2f} cm", (x, y - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_image, f"H: {dim_h:.2f} cm", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Final Measurements", use_column_width=True)

