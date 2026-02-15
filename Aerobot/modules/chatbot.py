import streamlit as st
import google.generativeai as genai
import os
import toml
import time
import modules.data_utils as data_utils # Import the data handler

def load_gemini_key():
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            with open(".streamlit/secrets.toml", "r") as f:
                config = toml.load(f)
                return config.get("GOOGLE_API_KEY")
        return st.secrets["GOOGLE_API_KEY"]
    except:
        return None

def get_dataset_summary(df):
    """
    Creates a condensed summary of the dataframe to feed to the AI.
    This gives the AI 'vision' of your data without uploading the whole file.
    """
    summary = []
    summary.append(f"Dataset Overview: {df.shape[0]} flight records, {df.shape[1]} columns.")
    summary.append(f"Columns: {', '.join(df.columns)}")
    
    # Add Key Financials if they exist
    if 'Revenue' in df.columns:
        total_rev = df['Revenue'].sum()
        summary.append(f"Total Revenue: ${total_rev:,.2f}")
    if 'Profit' in df.columns:
        total_prof = df['Profit'].sum()
        summary.append(f"Total Profit: ${total_prof:,.2f}")
        
    # Add Date Range if exists
    if 'Date' in df.columns:
        start = df['Date'].min()
        end = df['Date'].max()
        summary.append(f"Date Range: {start} to {end}")
        
    # Add a sample (First 3 rows) so AI sees the format
    summary.append("\nSample Data (First 3 rows):")
    summary.append(df.head(3).to_string(index=False))
    
    return "\n".join(summary)

def show_chatbot():
    st.title("🤖 Aerobot AI Assistant")
    st.markdown("Ask questions about your uploaded airline data.")

    # 1. Load Data Context
    df = data_utils.load_data()
    data_context = ""
    
    if df is not None:
        data_context = get_dataset_summary(df)
        with st.expander("✅ Data Connection Active", expanded=False):
            st.text(data_context) # Show user what the AI sees
            st.success(f"Connected to {len(df)} rows of airline data.")
    else:
        st.warning("⚠️ No data loaded. Please upload a CSV in the Dashboard.")

    # 2. Configure AI
    api_key = load_gemini_key()
    if not api_key:
        st.error("Missing API Key.")
        st.stop()

    genai.configure(api_key=api_key)

    # 3. Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I have analyzed your dashboard data. Ask me about revenue, top routes, or efficiency."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Handle User Input
    if prompt := st.chat_input("Ex: Which route is the most profitable?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Analyzing data...")
            
            # --- CONTEXT INJECTION ---
            # We combine the Data Summary + User Question
            if data_context:
                augmented_prompt = f"""
                You are Aerobot, an expert Airline Business Analyst.
                Here is the summary of the current dataset loaded in the dashboard:
                
                --- DATA SUMMARY START ---
                {data_context}
                --- DATA SUMMARY END ---
                
                User Question: {prompt}
                
                Instructions:
                1. Use the data summary above to answer.
                2. If the answer isn't in the summary, make a reasonable estimate based on general airline knowledge but mention it's an estimate.
                3. Keep answers professional and concise.
                """
            else:
                augmented_prompt = prompt

            # --- MODEL GENERATION (With Fallback) ---
            models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
            success = False
            
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(augmented_prompt)
                    
                    placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    success = True
                    break
                except Exception:
                    time.sleep(1)
                    continue
            
            if not success:
                placeholder.error("Could not connect to AI. Please try again.")