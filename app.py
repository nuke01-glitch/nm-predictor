import streamlit as st
import base64

st.set_page_config(layout="wide")

# Function to load your local GLB file
def get_glb_html(file_path):
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    
    # This uses the Google Model Viewer web component
    return f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>
    <model-viewer src="data:model/gltf+binary;base64,{data}" 
                  alt="ZnO Structure" 
                  auto-rotate 
                  camera-controls 
                  style="width: 100%; height: 500px; background-color: #0e1117;">
    </model-viewer>
    """

st.title("Nano-Material Predictor")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Parameters")
    # Your prediction logic here...
    formula = st.text_input("Formula", "ZnO")
    # ... rest of your form

with col2:
    st.subheader("Wurtzite Structure")
    # Render your local GLB
    st.components.v1.html(get_glb_html("assets/wurtzite_zno.glb"), height=500)