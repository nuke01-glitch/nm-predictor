import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from stmol import showmol
import py3Dmol


# --- 1. MODEL LOADING ---
@st.cache_resource
def load_models():
    # These MUST match the filenames in your /models/ folder
    model_files = [
        "model_bandgap_eV.cbm",
        "model_density_g_cm3.cbm",
        "model_formation_energy_eV.cbm",
        "model_specific_heat_J_gK.cbm"
    ]
    
    loaded_models = []
    for f_name in model_files:
        model = CatBoostRegressor()
        # Look inside the 'models' folder
        path = os.path.join("models", f_name)
        if os.path.exists(path):
            model.load_model(path)
            loaded_models.append(model)
        else:
            st.error(f"⚠️ Missing model file: {path}")
    return loaded_models

# Initialize Models
models = load_models()

# --- 2. THE PHYSICS ENGINE (Must Match Training DNA) ---
def add_quantum_features(df):
    """
    Synchronizes the input data with the training features:
    Index 5: inv_size_sq, Index 6: inv_size, Index 7: log_size
    """
    df['size_nm'] = df['size_nm'].astype(float)
    # 1/L^2: Particle in a Box (The 0.677eV shift driver)
    df['inv_size_sq'] = 1 / (df['size_nm']**2 + 1e-6)
    # 1/L: Surface area scaling
    df['inv_size'] = 1 / (df['size_nm'] + 1e-6)
    # log(L): Non-linear scaling
    df['log_size'] = np.log1p(df['size_nm'])
    return df

# --- 3. 3D LATTICE GENERATOR ---
def render_lattice(structure_type):
    view = py3Dmol.view(width=400, height=400)
    if structure_type == "Cubic":
        for x in [0, 2]:
            for y in [0, 2]:
                for z in [0, 2]:
                    view.addSphere({'center':{'x':x,'y':y,'z':z}, 'radius':0.6, 'color':'#00d4ff'})
        view.addBox({'center':{'x':1,'y':1,'z':1}, 'dimensions': {'w':2,'h':2,'d':2}, 'color':'white', 'opacity':0.2})
    elif structure_type == "Hexagonal":
        centers = [[0,0,0], [1,1.73,0], [-1,1.73,0], [2,0,0], [1,-1.73,0]]
        for c in centers:
            view.addSphere({'center':{'x':c[0],'y':c[1],'z':0}, 'radius':0.7, 'color':'purple'})
            view.addSphere({'center':{'x':c[0],'y':c[1],'z':1.2}, 'radius':0.4, 'color':'yellow'})
            view.addSphere({'center':{'x':c[0],'y':c[1],'z':-1.2}, 'radius':0.4, 'color':'yellow'})
    else:
        view.addSphere({'center':{'x':0,'y':0,'z':0}, 'radius':1.0, 'color':'red'})
        for i in range(6):
            view.addSphere({'center':{'x':np.cos(i)*2,'y':np.sin(i)*2,'z':0}, 'radius':0.5, 'color':'silver'})
    
    view.zoomTo()
    view.spin(True)
    return showmol(view, height=400, width=500)

# --- 4. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .physics-card { background: rgba(0, 150, 255, 0.1); padding: 20px; border-radius: 15px; border-left: 5px solid #00d4ff; margin-top: 20px;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: rgba(255,255,255,0.05); border-radius: 5px; padding: 10px 20px; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Nano-Material Predictive AI Lab")
st.write("First-Year B.Tech Project | Cluster Innovation Centre (CIC)")

# --- 5. TABS INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction Dashboard", "🧊 3D Structural Lab", "📜 Project Abstract"])

with tab1:
    col_input, col_graph = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.header("⚙️ Parameters")
        formula = st.text_input("Chemical Formula", "ZnO")
        size_nm = st.slider("Particle Size (nm)", 2.0, 120.0, 5.0)
        
        c1, c2 = st.columns(2)
        with c1:
            structure = st.selectbox("Crystal System", ["Hexagonal", "Cubic", "Monoclinic", "Rutile"])
        with c2:
            m_class = st.selectbox("Material Class", ["semiconductor", "metal oxide", "perovskite", "carbon-based"])
        
        shape = st.selectbox("Shape", ["Powder", "Ellipsoidal", "Sphere", "Rod"])

        # --- THE CRITICAL SYNC BLOCK ---
        # 1. Map Inputs to Dictionary
        input_dict = {
            'formula': str(formula),
            'crystal_structure': str(structure),
            'material_class': str(m_class),
            'shape': str(shape),
            'size_nm': float(size_nm)
        }
        
        # 2. Convert to DataFrame and Apply Physics
        input_df = pd.DataFrame([input_dict])
        input_df = add_quantum_features(input_df)
        
        # 3. FORCE THE GOLDEN COLUMN ORDER (Indices 0-7)
        feature_cols = [
            'formula', 'crystal_structure', 'material_class', 'shape', 
            'size_nm', 'inv_size_sq', 'inv_size', 'log_size'
        ]
        
        # FINAL REORDERING BEFORE PREDICTION
        input_df = input_df[feature_cols]

        try:
            # 4. Generate Predictions
            preds = [model.predict(input_df)[0] for model in models]
            
            st.markdown("---")
            st.subheader("🎯 Model Output")
            r1, r2 = st.columns(2)
            r1.metric("Bandgap", f"{preds[0]:.4f} eV")
            r1.metric("Density", f"{preds[1]:.2f} g/cm³")
            r2.metric("Formation Energy", f"{preds[2]:.4f} eV/at")
            r2.metric("Specific Heat", f"{preds[3]:.4f} J/gK")
            
        except Exception as e:
            st.error(f"❌ Alignment Error: {e}")
            st.info("Ensure you have uploaded the V5 .cbm models to your models/ folder.")

    with col_graph:
        st.header("📈 Scaling Curve")
        
        # Generate dynamic curve based on predicted bandgap & 1/L^2 physics
        sizes = np.linspace(2, 120, 100)
        # Shift curve relative to the current prediction
        shift_constant = 2.0 
        curve_vals = (preds[0] - (shift_constant / (size_nm**2))) + (shift_constant / (sizes**2))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=curve_vals, name="Quantum Confinement Trend", line=dict(color='#00d4ff', width=3)))
        fig.add_trace(go.Scatter(x=[size_nm], y=[preds[0]], mode='markers', marker=dict(size=15, color='orange', symbol='diamond'), name="Current Prediction"))
        
        fig.update_layout(
            template="plotly_dark", 
            margin=dict(l=0, r=0, t=20, b=0), 
            xaxis_title="Size (nm)", 
            yaxis_title="Bandgap (eV)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_view=True)
        
        insight = "✨ **Strong Quantum Confinement:** Significant Bandgap Widening." if size_nm < 15 else \
                  "⚡ **Surface Dominance:** Scaling laws in effect." if size_nm < 50 else \
                  "🏢 **Bulk-like Behavior:** Properties stabilizing."
        
        st.markdown(f"<div class='physics-card'><h4>🧠 Physics Insight</h4>{insight}</div>", unsafe_allow_html=True)

with tab2:
    st.header("🧊 Atomic Visualization")
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.write(f"**Chemical Identity:** {formula}")
        st.write(f"**Lattice Symmetry:** {structure}")
        st.info("Interactive unit cell model. Drag to rotate, scroll to zoom.")
    with c_right:
        render_lattice(structure)

with tab3:
    st.header("📜 Project Abstract")
    st.info("Research Mentor: Prof. Mahima Kaushik (CIC)")
    st.markdown(f"""
    **Project Title:** AI-Driven Predictive Modeling of Nanomaterial Properties.
    
    **Overview:** This project explores the intersection of Material Science and Machine Learning. By utilizing Gradient Boosted Trees (CatBoost), the platform predicts properties that are traditionally expensive to measure experimentally.
    
    **The Physics Edge:** Traditional ML models treat nanomaterials like bulk materials. This version (V5) incorporates **Quantum Confinement Descriptors** ($1/L^2$ and $1/L$), allowing the model to capture the non-linear electronic shifts inherent in nanostructures.
    
    **Academic Context:** Part of the foundational research for the IIT Madras BS (Data Science) and Cluster Innovation Centre (University of Delhi) curriculum.
    """)