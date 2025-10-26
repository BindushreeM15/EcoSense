import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- PAGE SETUP -----------------
st.set_page_config(
    page_title="EcoSense — Sustainability Profiler 🌱",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- LOAD MODEL & DATA -----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/sustainable_Dataset_15000.csv")
    except:
        df = pd.read_csv("sustainable_Dataset_15000.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/sustainability_model.pkl")
    return model

df = load_data()
model = load_model()

# ----------------- CUSTOM STYLES -----------------
st.markdown("""
    <style>
        .title {
            font-size: 38px;
            font-weight: 700;
            color: #2e7d32;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
        }
        .result-box {
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
        }
        .high {background-color: rgba(56, 142, 60, 0.15);}
        .medium {background-color: rgba(255, 235, 59, 0.15);}
        .low {background-color: rgba(211, 47, 47, 0.15);}
    </style>
""", unsafe_allow_html=True)

# ----------------- APP HEADER -----------------
st.markdown("<p class='title'>🌿 EcoSense — Product Sustainability Profiler</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze, Predict & Suggest Sustainable Alternatives</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- SIDEBAR -----------------
st.sidebar.header("🔧 Settings & Information")
st.sidebar.info("""
EcoSense uses ML & NLP techniques to:
- Predict the **Sustainability Level** (High / Medium / Low)
- Provide **eco-friendly alternatives**
- Suggest **product improvements**
""")

st.sidebar.markdown("**Project Objectives:**")
st.sidebar.markdown("""
1️⃣ Analyze product materials & properties  
2️⃣ Validate user input for product relevance  
3️⃣ Predict sustainability level  
4️⃣ Suggest alternatives or improvements  
""")

st.sidebar.markdown("**Team:** EcoVision | SDG Aligned 🌏")

# ----------------- MAIN SECTION -----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Enter Product Details or Description")
    user_input = st.text_area(
        "Type a product name, short description, or material details:",
        placeholder="Example: Plastic water bottle with aluminum cap",
        height=100
    )

    if st.button("Analyze Sustainability"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid product description.")
        else:
            # Basic simulation: simple text-based lookup
            matches = df[df["Product Name"].str.contains(user_input.split()[0], case=False, na=False)].head(3)
            
            if len(matches) == 0:
                st.error("No matching products found. Try with a broader or simpler keyword.")
            else:
                st.success(f"Found {len(matches)} similar products!")

                for _, row in matches.iterrows():
                    score = row["Sustainability_Score"]
                    level = (
                        "High" if score >= 75 else
                        "Medium" if score >= 40 else
                        "Low"
                    )
                    color_class = "high" if level == "High" else "medium" if level == "Medium" else "low"

                    st.markdown(f"""
                    <div class='result-box {color_class}'>
                        <b>🧾 Product:</b> {row['Product Name']}<br>
                        <b>📦 Category:</b> {row['Category']}<br>
                        <b>🌱 Sustainability Score:</b> {score:.2f} / 100<br>
                        <b>🟢 Level:</b> {level}<br>
                        <b>♻️ Alternative Suggestion:</b> {row.get('Sustainable_Alternative', 'Not available')}
                    </div>
                    """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 Quick Insights")

    avg_score = df["Sustainability_Score"].mean()
    st.metric("Average Sustainability Score", f"{avg_score:.2f} / 100")

    high_count = len(df[df["Sustainability_Score"] >= 75])
    med_count = len(df[(df["Sustainability_Score"] < 75) & (df["Sustainability_Score"] >= 40)])
    low_count = len(df[df["Sustainability_Score"] < 40])

    st.write("### Distribution of Products")
    st.bar_chart(pd.DataFrame({
        "High": [high_count],
        "Medium": [med_count],
        "Low": [low_count]
    }).T.rename(columns={0: "Count"}))

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:14px; color:#888;'>
Developed by <b>EcoVision Team</b> 🌱 | Powered by <b>Streamlit & Scikit-learn</b>  
</div>
""", unsafe_allow_html=True)
