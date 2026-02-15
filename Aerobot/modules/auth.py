import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import requests
import json

# --- CONFIGURATION ---
if "FIREBASE_WEB_API_KEY" in st.secrets:
    WEB_API_KEY = st.secrets["FIREBASE_WEB_API_KEY"]
else:
    st.error("Missing FIREBASE_WEB_API_KEY in secrets.toml")
    st.stop()

def initialize_firebase():
    """Initializes the Firebase Admin SDK safely."""
    try:
        if not firebase_admin._apps:
            cred_dict = dict(st.secrets["firebase"])
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")

# --- AUTH FUNCTIONS ---

def sign_up_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return True, user.uid
    except Exception as e:
        return False, str(e)

def sign_in_user(email, password):
    request_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={WEB_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"email": email, "password": password, "returnSecureToken": True}
    
    try:
        response = requests.post(request_url, headers=headers, data=json.dumps(payload))
        data = response.json()
        
        if "error" in data:
            return False, data["error"]["message"]
        
        return True, {"email": data["email"], "localId": data["localId"], "idToken": data["idToken"]}
        
    except Exception as e:
        return False, str(e)

# --- UI COMPONENT ---

def login_page():
    initialize_firebase()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("✈️ Aerobot Access")
        
        tab1, tab2 = st.tabs(["Login", "Create Account"])
        
        # --- LOGIN TAB ---
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Secure Login", use_container_width=True)
                
                if submit:
                    if not email or not password:
                        st.warning("Please fill in all fields.")
                    else:
                        success, result = sign_in_user(email, password)
                        if success:
                            st.success("Login Successful!")
                            
                            # 1. SET SESSION STATE
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = result['email'].split('@')[0]
                            
                            # 2. SET URL PERSISTENCE TOKEN (The Fix)
                            st.query_params["session_id"] = result['localId']
                            st.query_params["user_email"] = result['email']
                            
                            st.rerun()
                        else:
                            st.error(f"Login Failed: {result}")

        # --- SIGN UP TAB ---
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("New Email")
                new_pass = st.text_input("New Password (Min 6 chars)", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                create_btn = st.form_submit_button("Create Account", use_container_width=True)
                
                if create_btn:
                    if new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    elif len(new_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        success, result = sign_up_user(new_email, new_pass)
                        if success:
                            st.success("Account Created! Please go to the Login tab.")
                            st.balloons()
                        else:
                            st.error(f"Error: {result}")

        st.markdown("---")
        st.caption("Protected by Google Firebase Security")