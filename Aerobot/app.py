import streamlit as st
import modules.auth as auth
import modules.dashboard as dashboard
import modules.chatbot as chatbot
import modules.llm_scratch as llm_scratch
import modules.data_utils as data_utils
import modules.home as home
import modules.theme as theme

# --- Page Config ---
st.set_page_config(
    page_title="Aerobot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

# --- AUTO-LOGIN LOGIC (The Fix) ---
# Check if the URL has a session token (meaning user refreshed the page)
if not st.session_state['logged_in']:
    params = st.query_params
    if "session_id" in params and "user_email" in params:
        # Restore the session automatically
        st.session_state['logged_in'] = True
        st.session_state['username'] = params["user_email"].split('@')[0]
        # (Optional: You could verify the token with Firebase here for extra security, 
        # but for this project, checking presence is sufficient)

# Define Page Mapping
PAGES = {
    "Home": "🏠 Home Base",
    "Dashboard": "📊 Flight Analytics",
    "Chatbot": "🤖 Aero Copilot",
    "LLM": "🧠 Neural Engine"
}

def change_page(page_key):
    st.session_state['current_page'] = PAGES[page_key]

# --- Main Logic ---
def main():
    if not st.session_state['logged_in']:
        auth.login_page()
    else:
        # --- SIDEBAR ---
        with st.sidebar:
            st.title("✈️ Aerobot")
            st.caption("v3.4 Enterprise")
            
            st.markdown("---")
            
            # Module Header
            st.markdown("""
                <div style="
                    font-size: 1.1rem; 
                    font-weight: 700; 
                    color: #00BFFF; 
                    margin-bottom: 15px; 
                    text-transform: uppercase; 
                    letter-spacing: 1px;">
                    System Modules
                </div>
            """, unsafe_allow_html=True)
            
            if 'current_page' not in st.session_state:
                st.session_state['current_page'] = PAGES["Home"]
            
            selected_page = st.radio(
                "System Modules", 
                list(PAGES.values()),
                key="current_page",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            is_dark = st.toggle("🌙 Dark Mode", value=False, key="is_dark")
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.container(border=True):
                st.write(f"User: **{st.session_state['username']}**")
                # LOGOUT LOGIC
                if st.button("Logout", use_container_width=True):
                    st.session_state['logged_in'] = False
                    # CLEAR URL TOKENS ON LOGOUT
                    st.query_params.clear()
                    st.rerun()

        # --- APPLY THEME ---
        theme.apply_theme(is_dark)

        # --- ROUTING ---
        if selected_page == PAGES["Home"]:
            home.show_home(change_page)
            
        elif selected_page == PAGES["Dashboard"]:
            dashboard.show_dashboard()
            if data_utils.load_data() is None:
                st.divider()
                if st.button("Generate Sample Data"):
                    data_utils.generate_dummy_data_file()
                    st.rerun()

        elif selected_page == PAGES["Chatbot"]:
            chatbot.show_chatbot()
            
        elif selected_page == PAGES["LLM"]:
            llm_scratch.show_llm_scratch_pad()

if __name__ == "__main__":
    main()