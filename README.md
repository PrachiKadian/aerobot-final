# **✈️ Aerobot: AI-Powered Aviation Intelligence Suite**

**Aerobot** is a comprehensive Business Intelligence (BI) dashboard designed for the aviation industry. It integrates real-time data visualization, AI-driven insights, and deep learning simulations into a single "Command Center" interface.

## **🌟 Key Features**

### **1\. 📊 Operational Dashboard**

* **Real-time Analytics:** Visualizes Revenue, Passenger Load, and Fuel Efficiency.  
* **Interactive Maps:** Live flight routes plotted on a global map using Plotly.  
* **Data Engineering:** Automatically calculates missing financial metrics (Profit Margins, Ticket Costs) from raw operational logs.

### **2\. 🤖 Aero Copilot (AI Assistant)**

* **Context-Aware:** The chatbot "reads" the uploaded dataset and answers specific questions like *"Which route is the most profitable?"*.  
* **Powered by Gemini 2.5:** Uses Google's latest LLM for high-accuracy reasoning.

### **3\. 🧠 Neural Engine (Educational Module)**

* **LLM from Scratch:** A PyTorch implementation of a Transformer model.  
* **Visualized Architecture:** Demonstrates "First Principles" of AI, including Tokenization, Embeddings, and Self-Attention mechanisms.

## **🛠️ Tech Stack**

* **Frontend:** Streamlit (Python)  
* **Data Processing:** Pandas, NumPy  
* **Visualization:** Plotly Express & Graph Objects  
* **AI/ML:** PyTorch, Google Gemini API  
* **Database/Auth:** Firebase Authentication

## **🚀 How to Run Locally**

1. **Clone the repository:**  
   git clone \[https://github.com/your-username/aerobot.git\](https://github.com/your-username/aerobot.git)

2. **Install dependencies:**  
   pip install \-r requirements.txt

3. **Run the application:**  
   streamlit run app.py

## **📂 Project Structure**

* app.py: Main entry point and navigation logic.  
* modules/: Contains separate logic for Dashboard, Chatbot, and Auth.  
* data/: Stores airline datasets (Indigo, Generic).  
* assets/: UI assets and dashboard banners.

*Built for the Deep Learning Engineering Project (Feb 2026).*
