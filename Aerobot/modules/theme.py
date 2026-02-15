import streamlit as st

def apply_theme(is_dark_mode):
    """
    Injects CSS to override the default theme.
    Fixed: Sidebar Radio Buttons are now Centered Cards.
    """
    
    # --- CSS CONSTANTS ---
    # We use these to style the radio buttons purely via CSS
    card_css = """
        /* Hide the default radio circle */
        [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
            display: none;
        }
        
        /* THE CARD CONTAINER */
        [data-testid="stRadio"] div[role="radiogroup"] > label {
            border-radius: 8px;
            padding: 0px 10px; /* Horizontal padding */
            margin-bottom: 10px;
            width: 100%;
            height: 50px; /* Fixed Height */
            display: flex;
            justify-content: center; /* Horizontal Center */
            align-items: center;     /* Vertical Center */
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }
        
        /* Centering the Text inside the Card */
        [data-testid="stRadio"] div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] {
            width: 100%;
            text-align: center;
            line-height: 1.2;
        }
        
        /* Hover Effect */
        [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            transform: scale(1.02);
        }
    """

    # --- GLOBAL STYLES ---
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important; 
            padding-bottom: 1rem !important;
        }
        
        .stButton > button { border-radius: 8px; }
        .stTextInput > div > div { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    if is_dark_mode:
        # --- DARK MODE CSS ---
        st.markdown(f"""
        <style>
            /* Main Backgrounds */
            [data-testid="stAppViewContainer"] {{ background-color: #0E1117; color: #E0E0E0; }}
            [data-testid="stSidebar"] {{ background-color: #1E1E1E; border-right: 1px solid #2D2D2D; }}
            
            /* Header */
            header, [data-testid="stHeader"] {{ 
                background-color: #0E1117 !important;
            }}
            [data-testid="stHeader"] button {{
                color: #E0E0E0 !important;
            }}
            
            /* --- SIDEBAR CARD STYLES (DARK) --- */
            {card_css}
            
            /* Default State (Unselected) */
            [data-testid="stRadio"] div[role="radiogroup"] > label {{
                background-color: #1F2937;
                border: 1px solid #374151;
                color: #E0E0E0;
            }}
            [data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
                border-color: #00BFFF;
            }}
            
            /* SELECTED STATE (The Blue Card) */
            [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
                background-color: #00BFFF !important;
                border-color: #00BFFF !important;
                color: #FFFFFF !important;
                font-weight: bold;
                box-shadow: 0 0 10px rgba(0, 191, 255, 0.3);
            }}
            /* Ensure text inside selected card is white */
            [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {{
                color: #FFFFFF !important;
            }}

            /* --- UPLOAD BOX --- */
            section[data-testid="stFileUploaderDropzone"] {{
                background-color: #1F2937 !important;
                border: 1px dashed #4B5563 !important;
            }}
            section[data-testid="stFileUploaderDropzone"] * {{
                color: #E0E0E0 !important;
            }}
            section[data-testid="stFileUploaderDropzone"] button {{
                border-color: #00BFFF !important; 
                color: #00BFFF !important;
            }}
            [data-testid="stFileUploaderDropzone"] svg {{
                fill: #00BFFF !important;
            }}

            /* --- CHAT INPUT --- */
            [data-testid="stBottom"] {{
                background-color: #0E1117 !important;
                border-top: 1px solid #1E1E1E; 
            }}
            [data-testid="stBottom"] > div {{
                background-color: #0E1117 !important;
            }}
            div[data-testid="stChatInput"] > div {{
                background-color: #1F2937 !important; 
                border: 1px solid #374151 !important;
                color: #E0E0E0 !important;
            }}
            div[data-testid="stChatInput"] [data-baseweb="input"] {{
                background-color: #1F2937 !important; 
                border: none !important;
            }}
            div[data-testid="stChatInput"] [data-baseweb="base-input"] {{
                background-color: #1F2937 !important;
            }}
            textarea[data-testid="stChatInputTextArea"] {{
                background-color: transparent !important; 
                color: #E0E0E0 !important;
                caret-color: #00BFFF !important;
            }}
            textarea[data-testid="stChatInputTextArea"]::placeholder {{
                color: #888888 !important;
            }}
            button[data-testid="stChatInputSubmitButton"] {{
                color: #00BFFF !important;
                background-color: transparent !important;
            }}

            /* --- REST --- */
            h1, h2, h3, p, li, .stMarkdown {{ color: #E0E0E0 !important; }}
            .hero-title {{ color: #00BFFF !important; }} 
            .hero-subtitle {{ color: #B0B0B0 !important; }}
            
            /* Buttons */
            .stButton > button {{
                background-color: #1F2937 !important; 
                color: #00BFFF !important;           
                border: 1px solid #00BFFF !important;
                transition: all 0.3s ease;
            }}
            .stButton > button:hover {{
                background-color: #00BFFF !important; 
                color: #FFFFFF !important;           
                box-shadow: 0 0 10px rgba(0, 191, 255, 0.4);
            }}
            
            /* Inputs */
            .stTextInput > div > div {{
                background-color: #1F2937; 
                color: white;
                border: 1px solid #4B5563;
            }}

            /* Cards */
            .feature-box {{ 
                background-color: #1F2937 !important; 
                border-left: 5px solid #00BFFF !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
            }}
            .box-title {{ color: #E0E0E0 !important; }}
            .box-desc {{ color: #A0A0A0 !important; }}
            
            div[data-testid="stMetric"] {{ background-color: #1F2937; border: 1px solid #374151; }}
            div[data-testid="stMetric"] label {{ color: #B0B0B0 !important; }}
            div[data-testid="stMetric"] div {{ color: #00BFFF !important; }}
            .streamlit-expanderHeader {{ background-color: #1F2937; color: #E0E0E0; }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # --- LIGHT MODE CSS ---
        st.markdown(f"""
        <style>
            [data-testid="stAppViewContainer"] {{ background-color: #FFFFFF; color: #2D2D2D; }}
            [data-testid="stSidebar"] {{ background-color: #F8F9FA; border-right: 1px solid #E5E7EB; }}
            [data-testid="stBottom"] {{ background-color: #FFFFFF !important; border-top: 1px solid #F0F0F0; }}
            [data-testid="stHeader"] {{ background-color: #FFFFFF !important; }}
            
            /* --- SIDEBAR CARD STYLES (LIGHT) --- */
            {card_css}
            
            /* Default State (Unselected) */
            [data-testid="stRadio"] div[role="radiogroup"] > label {{
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                color: #2D2D2D;
            }}
            [data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
                border-color: #00BFFF;
                background-color: #F0F9FF;
            }}
            
            /* SELECTED STATE */
            [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
                background-color: #00BFFF !important;
                border-color: #00BFFF !important;
                color: #FFFFFF !important;
                font-weight: bold;
                box-shadow: 0 2px 5px rgba(0, 191, 255, 0.2);
            }}
            [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {{
                color: #FFFFFF !important;
            }}

            /* Upload Box */
            section[data-testid="stFileUploaderDropzone"] {{
                background-color: #F8F9FA !important;
                border: 1px dashed #E5E7EB !important;
            }}
            section[data-testid="stFileUploaderDropzone"] * {{
                color: #2D2D2D !important;
            }}
            section[data-testid="stFileUploaderDropzone"] button {{
                color: #00BFFF !important;
            }}

            .stButton > button {{
                background-color: #FFFFFF !important;
                color: #00BFFF !important;
                border: 1px solid #00BFFF !important;
                transition: all 0.3s ease;
            }}
            .stButton > button:hover {{
                background-color: #00BFFF !important;
                color: #FFFFFF !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}

            .feature-box {{ 
                background-color: #F8F9FA !important;
                border-left: 5px solid #00BFFF !important;
            }}
            .box-title {{ color: #2D2D2D !important; }}
            .box-desc {{ color: #666666 !important; }}
        </style>
        """, unsafe_allow_html=True)