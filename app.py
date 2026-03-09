import streamlit as st
import base64
import pandas as pd
import numpy as np
import re
import os
from catboost import CatBoostRegressor

# 1. ELEMENTAL DATA & FEATURE EXTRACTION (Matches your original logic)
elements_data = {'H': [1.00, 2.20], 'Li': [6.94, 0.98], 'C': [12.01, 2.55], 'O': [16.00, 3.44], 
                 'Zn': [65.38, 1.65], 'Ti': [47.87, 1.54], 'Si': [28.09, 1.90]} # Add remaining as needed

def extract_features(formula):
    parts = re.findall(r'([A-Z][a-z]*)(\d*)', str(formula))
    w, ens = [], []
    for el, c in parts:
        c = int(c) if c else 1
        if el in elements_data:
            w.extend([elements_data[el][0]] * c)
            ens.extend([elements_data[el][1]] * c)
    if not ens: return 50.0, 2.0, 0.0, 0.0
    return np.mean(w), np.mean(ens), np.max(ens) - np.min(ens), np.std(ens)

@st.cache_resource
def load_models():
    models = [CatBoostRegressor() for _ in range(4)]
    for i in range(4):
        models[i].load_model(f"model_v3_{i}.cbm")
    return models

models = load_models()

# 2. GLB HTML FUNCTION
def get_glb_html(file_path):
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>
    <model-viewer src="data:model/gltf+binary;base64,{data}" auto-rotate camera-controls 
                  style="width: 100%; height: 500px; background-color: #0e1117;">
    </model-viewer>
    """

st.set_page_config(layout="wide")
# Inject custom CSS for a professional animated background
page_bg_style = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* Optional: Style the cards to be semi-transparent so they pop */
[data-testid="stVerticalBlock"] {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)
st.title("Nano-Material Predictor")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Parameters")
    formula = st.text_input("Formula", "ZnO")
    size_nm = st.number_input("Size (nm)", 30.0)
    crystal_structure = st.selectbox("Structure", ["Hexagonal", "Rutile", "Monoclinic", "Cubic"])
    shape = st.selectbox("Shape", ["Powder", "Ellipsoidal", "Sphere", "Rod"])
    material_class = st.text_input("Material Class", "metal oxide")

    if st.button("Get Prediction"):
        w, avg_en, en_diff, en_std = extract_features(formula)
        input_dict = {
            'avg_w': w, 'avg_en': avg_en, 'en_diff': en_diff, 'en_std': en_std,
            'crystal_structure': crystal_structure, 'material_class': material_class,
            'size_nm': size_nm, 'inv_size': 1.0 / (size_nm + 1e-5), 'shape': shape
        }
        X = pd.DataFrame([input_dict])
        preds = [model.predict(X)[0] for model in models]
        
        st.subheader("Results")
        res1, res2 = st.columns(2)
        res1.metric("Bandgap", f"{preds[0]:.2f} eV")
        res1.metric("Density", f"{preds[1]:.2f} g/cm³")
        res2.metric("Formation Energy", f"{preds[2]:.2f} eV/atom")
        res2.metric("Specific Heat", f"{preds[3]:.4f} J/gK")

with col2:
    st.subheader("Wurtzite Structure")
    st.components.v1.html(get_glb_html("assets/wurtzite_zno.glb"), height=500)
