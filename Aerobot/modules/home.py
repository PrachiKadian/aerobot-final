import streamlit as st
import os
import base64

# --- HELPER: IMAGE ENCODER ---
@st.cache_data
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None

def show_home(nav_callback):
    
    # --- CSS ANIMATIONS & STYLING ---
    st.markdown("""
    <style>
        /* 1. KEYFRAME DEFINITIONS */
        @keyframes slideInLeft {
            0% { transform: translateX(-50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideInRight {
            0% { transform: translateX(50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideInUp {
            0% { transform: translateY(50px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }

        /* 2. APPLY ANIMATIONS */
        .animate-left {
            animation: slideInLeft 0.8s ease-out forwards;
        }
        .animate-up {
            animation: slideInUp 0.8s ease-out forwards;
            opacity: 0; /* Start hidden to prevent flash */
            animation-delay: 0.3s; /* Wait for text to start */
        }
        .animate-right-1 {
            animation: slideInRight 0.6s ease-out forwards;
            opacity: 0;
            animation-delay: 0.2s;
        }
        .animate-right-2 {
            animation: slideInRight 0.6s ease-out forwards;
            opacity: 0;
            animation-delay: 0.4s;
        }
        .animate-right-3 {
            animation: slideInRight 0.6s ease-out forwards;
            opacity: 0;
            animation-delay: 0.6s;
        }

        /* 3. COMPONENT STYLING */
        .hero-title {
            font-size: 3.8rem;
            font-weight: 800;
            color: #00BFFF;
            line-height: 1.1;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            opacity: 0.8;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .feature-box {
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 10px;
            transition: transform 0.2s;
        }
        .feature-box:hover {
            transform: translateX(5px);
        }
        .box-title {
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .box-desc {
            font-size: 0.85rem;
            margin-top: 5px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Split Layout
    col_left, col_space, col_right = st.columns([5, 0.5, 4])

    # --- LEFT SIDE (Text & Image) ---
    with col_left:
        # Title Container (Left to Right)
        st.markdown("""
        <div class="animate-left">
            <div class="hero-title">Aviation<br>Intelligence<br>Redefined.</div>
            <div class="hero-subtitle">
                <b>Aerobot</b> centralizes your airline's entire operation into one intelligent dashboard. 
                From predictive maintenance to real-time revenue tracking, we turn complex data into 
                clear decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Image Logic
        local_img_path = "assets/dashboard_cover3.png"
        img_src = ""
        if not os.path.exists(local_img_path):
            local_img_path = "assets/dashboard_cover3.jpg"

        if os.path.exists(local_img_path):
            base64_img = get_base64_image(local_img_path)
            if base64_img:
                mime_type = "image/png" if local_img_path.endswith(".png") else "image/jpeg"
                img_src = f"data:{mime_type};base64,{base64_img}"
            else:
                img_src = "https://img.freepik.com/free-vector/airplane-flight-route-map-background_1017-33475.jpg?w=1380"
        else:
            img_src = "https://img.freepik.com/free-vector/airplane-flight-route-map-background_1017-33475.jpg?w=1380"

        # Image Container (Bottom to Top)
        st.markdown(f"""
            <div class="animate-up" style="
                width: 100%; 
                height: 250px; 
                border-radius: 12px; 
                overflow: hidden; 
                border: 1px solid #374151;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: 20px;
            ">
                <img src="{img_src}" style="
                    width: 100%; 
                    height: 100%; 
                    object-fit: cover; 
                    object-position: center;
                ">
            </div>
        """, unsafe_allow_html=True)

    # --- RIGHT SIDE (Cards sliding in from Right) ---
    with col_right:
        # CARD 1
        st.markdown("""
        <div class="animate-right-1 feature-box">
            <div class="box-title">📊 Flight Analytics</div>
            <div class="box-desc">
                Comprehensive dashboard for revenue, routes, and operational efficiency metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Launch Dashboard", on_click=nav_callback, args=("Dashboard",), use_container_width=True, key="btn_dash")
            
        st.markdown("<br>", unsafe_allow_html=True)

        # CARD 2
        st.markdown("""
        <div class="animate-right-2 feature-box">
            <div class="box-title">🤖 Aero Copilot</div>
            <div class="box-desc">
                Your AI-powered strategic partner. Ask questions, get summaries, and forecast trends.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open AI Copilot", on_click=nav_callback, args=("Chatbot",), use_container_width=True, key="btn_ai")
            
        st.markdown("<br>", unsafe_allow_html=True)

        # CARD 3
        st.markdown("""
        <div class="animate-right-3 feature-box">
            <div class="box-title">🧠 Neural Engine</div>
            <div class="box-desc">
                Experimental LLM architecture built from scratch. Explore tokenization layers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button("Enter Neural Lab", on_click=nav_callback, args=("LLM",), use_container_width=True, key="btn_llm")