import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from stmol import showmol
import py3Dmol
import streamlit.components.v1 as components
import json

# --- 1. CONSTANTS & BENCHMARKS ---
BULK_BENCHMARKS = {
    "ZnO": 3.37, "TiO2": 3.20, "CdSe": 1.74, "GaN": 3.40, 
    "Si": 1.12, "CuO": 1.20, "MAPbI3": 1.55, "GaAs": 1.42,
    "CdS": 2.42, "ZnS": 3.60, "MoS2": 1.20, "InP": 1.35
}

DATASET_MATERIALS = ["ZnO", "TiO2", "Fe3O4", "CdSe", "Au", "Ag", "SiO2", "CuO", "Al2O3", 
                     "MoS2", "Al-ZnO", "Fe-TiO2", "MAPbI3", "C", "CsPbBr3", "FAPbI3", 
                     "Ti3C2", "WS2", "GaN", "CNT", "Pt", "BN", "SiC", "NiO", "MgO", 
                     "Si", "Ge", "GaAs", "InP", "CdS", "ZnS", "WSe2", "MoSe2", "ZrO2", 
                     "SnO2", "In2O3", "MAPbBr3", "CsPbI3", "C60"]

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_models():
    model_files = ["model_bandgap_eV.cbm", "model_density_g_cm3.cbm", 
                   "model_formation_energy_eV.cbm", "model_specific_heat_J_gK.cbm"]
    loaded_models = []
    for f_name in model_files:
        path = os.path.join("models", f_name)
        if os.path.exists(path):
            model = CatBoostRegressor()
            model.load_model(path)
            loaded_models.append(model)
        else:
            st.error(f"⚠️ Missing model file: {path}")
    return loaded_models

models = load_models()

# --- 3. THE PHYSICS & PREDICTION ENGINE ---
def add_quantum_features(df):
    df['size_nm'] = df['size_nm'].astype(float)
    df['inv_size_sq'] = 1 / (df['size_nm']**2 + 1e-6)
    df['inv_size'] = 1 / (df['size_nm'] + 1e-6)
    df['log_size'] = np.log1p(df['size_nm'])
    return df

def predict_material(models_list, formula, size, structure, m_class, shape):
    input_dict = {
        'formula': str(formula),
        'crystal_structure': str(structure),
        'material_class': str(m_class),
        'shape': str(shape),
        'size_nm': float(size)
    }
    df = pd.DataFrame([input_dict])
    df = add_quantum_features(df)
    feature_cols = ['formula', 'crystal_structure', 'material_class', 'shape', 
                    'size_nm', 'inv_size_sq', 'inv_size', 'log_size']
    return [m.predict(df[feature_cols])[0] for m in models_list]

# --- 4. 3D LATTICE GENERATOR ---
def render_lattice(structure_type):
    view = py3Dmol.view(width=400, height=400)
    if structure_type == "Cubic":
        for x in [0, 2]:
            for y in [0, 2]:
                for z in [0, 2]:
                    view.addSphere({'center':{'x':x,'y':y,'z':z}, 'radius':0.6, 'color':'#00d4ff'})
        view.addBox({'center':{'x':1,'y':1,'z':1}, 'dimensions': {'w':2,'h':2,'d':2}, 'color':'white', 'opacity':0.1})
    elif structure_type == "Hexagonal":
        centers = [[0,0,0], [1,1.73,0], [-1,1.73,0], [2,0,0], [0,1.15,1.5], [1,2.88,1.5]]
        for c in centers:
            view.addSphere({'center':{'x':c[0],'y':c[1],'z':c[2] if len(c)>2 else 0}, 'radius':0.7, 'color':'purple'})
    elif structure_type == "Rutile":
        for x in [0, 2]:
            for y in [0, 2]:
                for z in [0, 3.5]:
                    view.addSphere({'center':{'x':x,'y':y,'z':z}, 'radius':0.5, 'color':'#ff4b4b'})
        view.addBox({'center':{'x':1,'y':1,'z':1.75}, 'dimensions': {'w':2,'h':2,'d':3.5}, 'color':'white', 'opacity':0.1})
    elif structure_type == "Perovskite":
        view.addSphere({'center':{'x':1,'y':1,'z':1}, 'radius':0.8, 'color':'#ff00ff'}) 
        for x in [0, 2]:
            for y in [0, 2]:
                for z in [0, 2]:
                    view.addSphere({'center':{'x':x,'y':y,'z':z}, 'radius':0.6, 'color':'#00d4ff'})
    else:
        view.addSphere({'center':{'x':0,'y':0,'z':0}, 'radius':1.0, 'color':'red'})
    view.zoomTo(); view.spin(True)
    return showmol(view, height=400, width=500)


# --- 5. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")

st.markdown("""
<style>
    /* 1. Base Terminal Aesthetic */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&display=swap');

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 10%, #1e2a4a 0%, #0f0c29 50%, #050505 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* 2. Glassmorphism Containers */
    div[data-testid="stMetric"], .physics-card, div.stTable, .stExpander {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* 3. Metric Enhancements */
    [data-testid="stMetricValue"] {
        font-family: 'Fira Code', monospace;
        color: #00d4ff !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }

    /* 4. Glowing Title Effect */
    .main-title {
        font-size: 3rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #ffffff, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* 5. Modern Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(0,0,0,0.2);
        padding: 5px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 212, 255, 0.15) !important;
        border: 1px solid #00d4ff !important;
    }

    /* 6. Sidebar 'Console' Look */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }

    /* 7. Button Glow */
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #005f73);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #00d4ff;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='color:#00d4ff;'>🔬 NanoPredict v5</h1>", unsafe_allow_html=True)
    app_mode = st.radio("System Mode", ["Public Dashboard", "Advanced Research Hub"])
    
    st.markdown("---")
    st.markdown("### 🛰️ System Status")
    st.markdown("""
    <div style="font-size: 0.8rem; line-height: 1.6;">
        <span style="color: #00ff00;">●</span> AI Core: Operational<br>
        <span style="color: #00d4ff;">●</span> Physics Engine: Loaded<br>
        <span style="color: #ffaa00;">●</span> Latency: 12ms
    </div>
    """, unsafe_allow_html=True)

if app_mode == "Public Dashboard":
    st.markdown('<h1 class="main-title">Nano-Material Predictive AI Lab</h1>', unsafe_allow_html=True)
    


        # --- INSERT THIS HERO SECTION HERE ---
    # We'll generate the base coordinate logic inside the JS to keep Python clean.
    hero_component = """
    <div id="container" style="width: 100%; height: 550px; background: #000; border-radius: 20px; position: relative; overflow: hidden; border: 1px solid #333;">
        <div id="ui-overlay" style="position: absolute; top: 20px; left: 20px; color: #00d4ff; font-family: 'Fira Code', monospace; pointer-events: none;">
            <div id="shape-name" style="font-size: 1.5rem; text-transform: uppercase; letter-spacing: 2px;">STATUS: STABLE CLOUD</div>
        </div>
        <button id="transmute-btn" style="position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); 
            padding: 15px 40px; background: transparent; border: 2px solid #00d4ff; 
            border-radius: 5px; color: #00d4ff; font-weight: bold; cursor: pointer; z-index: 100;
            transition: 0.3s; font-family: 'Fira Code', monospace;">
            TRANSMUTE MATTER
        </button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.9.1/gsap.min.js"></script>

    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 550, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, 550);
        document.getElementById('container').appendChild(renderer.domElement);

        const count = 6000; // Total particles
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        
        // Initial State: Random Cloud
        for(let i=0; i < count*3; i++) {
            positions[i] = (Math.random() - 0.5) * 10;
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const material = new THREE.PointsMaterial({ 
            color: 0x00d4ff, 
            size: 0.035, 
            transparent: true, 
            opacity: 0.8,
            blending: THREE.AdditiveBlending 
        });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        camera.position.z = 5;

        // --- SHAPE GENERATORS ---
        function getNanoparticle() {
            const arr = new Float32Array(count * 3);
            for(let i=0; i<count; i++) {
                const phi = Math.acos(-1 + (2 * i) / count);
                const theta = Math.sqrt(count * Math.PI) * phi;
                arr[i*3] = Math.cos(theta) * Math.sin(phi) * 1.5;
                arr[i*3+1] = Math.sin(theta) * Math.sin(phi) * 1.5;
                arr[i*3+2] = Math.cos(phi) * 1.5;
            }
            return arr;
        }

        function getAnimal(type) {
    const arr = new Float32Array(count * 3);
    let i = 0;
    while (i < count) {
        // 1. Pick a random 3D coordinate in a workspace
        let x = (Math.random() - 0.5) * 4;
        let y = (Math.random() - 0.5) * 4;
        let z = (Math.random() - 0.5) * 2;
        let keep = false;

        // 2. Define Animal "Zones"
        if (type === 'dog') {
            // Body + Head + Floppy Ears
            if ((Math.abs(x) < 0.8 && Math.abs(y) < 0.5 && Math.abs(z) < 0.4) || // Body
                (Math.abs(x-0.9) < 0.4 && Math.abs(y-0.4) < 0.4 && Math.abs(z) < 0.3)) { // Head
                keep = true;
            }
        } else if (type === 'cat') {
            // Slender body + Pointy Triangle Ears
            if ((Math.abs(x) < 0.6 && Math.abs(y) < 0.4 && Math.abs(z) < 0.3) || // Body
                (Math.abs(x-0.7) < 0.3 && Math.abs(y-0.4) < 0.3 && Math.abs(z) < 0.2)) { // Head
                // Simple pointy ear check
                if (y > 0.6 && Math.abs(x-0.7) < (0.8 - y)) keep = true; 
                else keep = true;
            }
        } else if (type === 'snake') {
            // A 3D Sine Wave "Tube"
            let tubeX = x;
            let targetY = Math.sin(x * 2) * 0.8;
            let targetZ = Math.cos(x * 2) * 0.2;
            if (Math.sqrt((y-targetY)**2 + (z-targetZ)**2) < 0.15) keep = true;
        } else if (type === 'hornbill') {
            // Small body + Massive curved Beak
            if ((Math.abs(x) < 0.4 && Math.abs(y) < 0.4) || // Body
                (x > 0.3 && x < 1.5 && y > 0 && y < 0.5 - (x-0.3)*0.3)) { // Beak
                keep = true;
            }
        } else if (type === 'rhino') {
            // Huge heavy block + a Spike (Horn)
            if ((Math.abs(x) < 1.2 && Math.abs(y) < 0.7 && Math.abs(z) < 0.8) || // Massive body
                (x > 1.0 && y > 0.5 && Math.abs(z) < 0.1 && y < 1.5 - (x-1.0)*2)) { // Horn
                keep = true;
            }
        }

        if (keep) {
            arr[i*3] = x; arr[i*3+1] = y; arr[i*3+2] = z;
            i++;
        }
    }
    return arr;
}

        const shapes = [
            { name: "Nanoparticle", data: getNanoparticle(), color: 0x00d4ff },
            { name: "Canine (Dog)", data: getAnimal('dog'), color: 0xffaa00 },
            { name: "Feline (Cat)", data: getAnimal('cat'), color: 0xff00ff },
            { name: "Serpent (Snake)", data: getAnimal('snake'), color: 0x00ff88 },
            { name: "Hornbill (Bird)", data: getAnimal('hornbill'), color: 0xffff00 },
            { name: "Rhinoceros", data: getAnimal('rhino'), color: 0xff4400 }
        ];

        let currentIdx = 0;
        const btn = document.getElementById('transmute-btn');
        const label = document.getElementById('shape-name');

        btn.addEventListener('click', () => {
            const target = shapes[currentIdx];
            label.innerText = "TRANSMUTING TO: " + target.name;
            
            // Morph Animation
            gsap.to(points.geometry.attributes.position.array, {
                duration: 2,
                endArray: target.data,
                ease: "power3.inOut",
                onUpdate: () => points.geometry.attributes.position.needsUpdate = true
            });

            // Color Transition
            gsap.to(material.color, {
                duration: 1.5,
                r: new THREE.Color(target.color).r,
                g: new THREE.Color(target.color).g,
                b: new THREE.Color(target.color).b
            });

            currentIdx = (currentIdx + 1) % shapes.length;
        });

        function animate() {
            requestAnimationFrame(animate);
            points.rotation.y += 0.005;
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / 550;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, 550);
        });
    </script>
    """

    components.html(hero_component, height=560)
        # --- END OF HERO SECTION ---
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Dashboard", "🧊 3D Structural Lab", "🧪 Virtual Experiment", "📜 Project Abstract"])

    # TAB 1: DASHBOARD
    with tab1:
        col_input, col_graph = st.columns([1, 1.2], gap="large")
        with col_input:
            st.header("⚙️ Parameters")
            formula = st.selectbox("Chemical Formula", DATASET_MATERIALS, key="t1_f")
            size_nm = st.slider("Particle Size (nm)", 2.0, 120.0, 5.0, key="t1_s")
            c1, c2 = st.columns(2)
            with c1:
                structure = st.selectbox("Crystal System", ["Hexagonal", "Cubic", "Monoclinic", "Rutile", "Perovskite"], key="t1_st")
            with c2:
                m_class = st.selectbox("Material Class", ["semiconductor", "metal oxide", "perovskite", "carbon-based"], key="t1_cl")
            shape = st.selectbox("Shape", ["Powder", "Ellipsoidal", "Sphere", "Rod"], key="t1_sh")

            try:
                preds = predict_material(models, formula, size_nm, structure, m_class, shape)
                st.markdown("---")
                st.subheader("🎯 Model Output")
                r1, r2 = st.columns(2)
                r1.metric("Bandgap", f"{preds[0]:.4f} eV")
                r1.metric("Density", f"{preds[1]:.2f} g/cm³")
                r2.metric("Formation Energy", f"{preds[2]:.4f} eV/at")
                r2.metric("Specific Heat", f"{preds[3]:.4f} J/gK")
            except Exception as e:
                st.error(f"❌ Alignment Error: {e}")

        with col_graph:
            st.header("📈 Scaling Curve")
            sizes = np.linspace(2, 120, 100)
            shift_constant = 2.0 
            curve_vals = (preds[0] - (shift_constant / (size_nm**2))) + (shift_constant / (sizes**2))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sizes, y=curve_vals, name="Quantum Confinement Trend", line=dict(color='#00d4ff', width=3)))
            fig.add_trace(go.Scatter(x=[size_nm], y=[preds[0]], mode='markers', marker=dict(size=15, color='orange', symbol='diamond'), name="Current Prediction"))
            fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Size (nm)", yaxis_title="Bandgap (eV)")
            st.plotly_chart(fig, use_container_view=True)
            
            insight = "✨ **Strong Quantum Confinement:** Significant Bandgap Widening." if size_nm < 15 else \
                      "⚡ **Surface Dominance:** Scaling laws in effect." if size_nm < 50 else \
                      "🏢 **Bulk-like Behavior:** Properties stabilizing."
            st.markdown(f"<div class='physics-card'><h4>🧠 Physics Insight</h4>{insight}</div>", unsafe_allow_html=True)

    # TAB 2: VISUALIZATION
    with tab2:
        st.header("🧊 Atomic Visualization")
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.write(f"**Chemical Identity:** {formula}")
            st.write(f"**Lattice Symmetry:** {structure}")
            st.info("Interactive unit cell model. Drag to rotate, scroll to zoom.")
            st.markdown("---")
            st.write("This 3D render represents the unit cell configuration for the selected crystal system.")
        with c_right:
            render_lattice(structure)

    # TAB 3: VIRTUAL EXPERIMENT
    with tab3:
        st.header("🧪 Virtual Experiment: Multi-Material Comparison")
        show_bulk = st.checkbox("🔍 Show Experimental Bulk Benchmarks", value=True)
        col_input, col_graph = st.columns([1, 1.2], gap="large")
        
        # Initialize compare_mode OUTSIDE the expander to prevent crashes
        compare_mode = False
        
        with col_input:
            st.subheader("⚙️ Experiment Setup")
            st.markdown("#### **Material A (Control)**")
            formula_a = st.selectbox("Material A", DATASET_MATERIALS, index=0, key="v_fa")
            size_a = st.slider("Size A (nm)", 2.0, 120.0, 10.0, key="v_sa")
            preds_a = predict_material(models, formula_a, size_a, "Cubic", "metal oxide", "Sphere")
            st.markdown("---")
            
            with st.expander("🧪 Add Challenger (Material B)", expanded=True):
                formula_b = st.selectbox("Material B", DATASET_MATERIALS, index=1, key="v_fb")
                size_b = st.slider("Size B (nm)", 2.0, 120.0, 15.0, key="v_sb")
                preds_b = predict_material(models, formula_b, size_b, "Cubic", "metal oxide", "Sphere")
                compare_mode = True

        with col_graph:
            st.subheader("📈 Scaling Comparison")
            sizes = np.linspace(2, 120, 100)
            fig_comp = go.Figure()
            
            # Curve A
            curve_a = (preds_a[0] - (2.0 / (size_a**2))) + (2.0 / (sizes**2))
            fig_comp.add_trace(go.Scatter(x=sizes, y=curve_a, name=f"A: {formula_a}", line=dict(color='#00d4ff', width=3)))
            
            # Curve B (Only if compare_mode is True)
            if compare_mode:
                curve_b = (preds_b[0] - (2.0 / (size_b**2))) + (2.0 / (sizes**2))
                fig_comp.add_trace(go.Scatter(x=sizes, y=curve_b, name=f"B: {formula_b}", line=dict(color='#ff4b4b', width=3, dash='dot')))
            
            # Bulk Benchmarks Overlay
            if show_bulk:
                if formula_a in BULK_BENCHMARKS:
                    fig_comp.add_hline(y=BULK_BENCHMARKS[formula_a], line_dash="dash", line_color="#00d4ff", opacity=0.4, annotation_text=f"Bulk {formula_a}")
                if compare_mode and (formula_b in BULK_BENCHMARKS):
                    fig_comp.add_hline(y=BULK_BENCHMARKS[formula_b], line_dash="dash", line_color="#ff4b4b", opacity=0.4, annotation_text=f"Bulk {formula_b}")
            
            fig_comp.update_layout(template="plotly_dark", xaxis_title="Size (nm)", yaxis_title="Bandgap (eV)")
            st.plotly_chart(fig_comp, use_container_view=True)
            
            if compare_mode:
                st.subheader("🏁 Result Comparison")
                comp_df = pd.DataFrame({
                    "Property": ["Bandgap (eV)", "Formation Energy (eV)", "Density (g/cm³)"],
                    f"{formula_a} (A)": [f"{preds_a[0]:.3f}", f"{preds_a[2]:.3f}", f"{preds_a[1]:.2f}"],
                    f"{formula_b} (B)": [f"{preds_b[0]:.3f}", f"{preds_b[2]:.3f}", f"{preds_b[1]:.2f}"]
                })
                st.table(comp_df)

    # TAB 4: ABSTRACT
    with tab4:
        st.header("📜 Project Abstract")
        # EVERYTHING BELOW MUST BE INDENTED (4 SPACES)
        st.info("Research Mentor: Prof. Mahima Kaushik (CIC)")
        st.markdown(f"""
        **Project Title:** AI-Driven Predictive Modeling of Nanomaterial Properties.
        
        **Overview:** This project explores the intersection of Material Science and Machine Learning. By utilizing Gradient Boosted Trees (CatBoost), the platform predicts properties that are traditionally expensive to measure experimentally.
        
        **The Physics Edge:** Traditional ML models treat nanomaterials like bulk materials. This version (V5) incorporates **Quantum Confinement Descriptors** ($1/L^2$ and $1/L$), allowing the model to capture the non-linear electronic shifts inherent in nanostructures.
        
        **Academic Context:** Part of the foundational research for the IIT Madras BS (Data Science) and Cluster Innovation Centre (University of Delhi) curriculum.
        """)

else: # --- RESEARCH HUB MODE ---
    st.title("🔬 Advanced Research Hub")
    st.write("Specialized tools for high-throughput screening.")

    # FEATURE 1: BATCH PROCESSING
    st.header("📂 Batch CSV Analysis")
    uploaded = st.file_uploader("Upload candidates (CSV)", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        if st.button("Run Bulk Prediction"):
            if 'formula' in df_batch.columns and 'size_nm' in df_batch.columns:
                # Corrected function call to predict_material
                preds_list = [predict_material(models, r['formula'], r['size_nm'], "Cubic", "metal oxide", "Sphere")[0] for _, r in df_batch.iterrows()]
                df_batch['Predicted_Bandgap_eV'] = preds_list
                st.dataframe(df_batch)
                st.download_button("Download Report", df_batch.to_csv(index=False), "research_results.csv")
            else:
                st.error("CSV must have 'formula' and 'size_nm' columns.")

    # FEATURE 2: GLOBAL FEATURE IMPORTANCE
    st.markdown("---")
    st.header("🧠 Model Interpretability")
    if st.button("Show Decision Drivers"):
        importance = models[0].get_feature_importance()
        features = ['Formula', 'Structure', 'Class', 'Shape', 'Size', '1/L²', '1/L', 'log(L)']
        fig_imp = go.Figure(go.Bar(x=importance, y=features, orientation='h', marker_color='#00d4ff'))
        fig_imp.update_layout(template="plotly_dark", title="Key Feature Drivers for Bandgap Prediction")
        st.plotly_chart(fig_imp)