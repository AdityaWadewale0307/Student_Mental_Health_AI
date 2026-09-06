import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Mental Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - MOBILE FRIENDLY + COLORFUL
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 25%),
        radial-gradient(circle at 90% 10%, rgba(236,72,153,0.12), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(20,184,166,0.10), transparent 30%),
        #f8fafc;
}

/* Main container */

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Header */

.hero {
    padding: 30px;
    border-radius: 28px;
    background:
        linear-gradient(135deg,
        #4f46e5 0%,
        #7c3aed 45%,
        #ec4899 100%);
    color: white;
    box-shadow: 0 15px 40px rgba(79,70,229,0.25);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity: 0.92;
}

/* Cards */

.metric-card {
    background: rgba(255,255,255,0.90);
    border-radius: 20px;
    padding: 22px;
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 8px 25px rgba(15,23,42,0.07);
    min-height: 120px;
}

.metric-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-icon {
    font-size: 28px;
}

/* Prediction */

.prediction-card {
    padding: 35px;
    border-radius: 28px;
    background: linear-gradient(
        135deg,
        rgba(79,70,229,0.12),
        rgba(236,72,153,0.12)
    );
    border: 1px solid rgba(99,102,241,0.15);
    text-align: center;
}

.prediction-score {
    font-size: 65px;
    font-weight: 800;
    color: #4f46e5;
}

.prediction-label {
    font-size: 22px;
    font-weight: 700;
}

/* Section headings */

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 12px 18px;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(79,70,229,0.25);
}

/* Inputs */

.stSelectbox, .stNumberInput, .stSlider {
    margin-bottom: 8px;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 16px;
    font-weight: 700;
}

/* Info */

.info-box {
    padding: 18px;
    border-radius: 18px;
    background: #eef2ff;
    border-left: 5px solid #6366f1;
    margin: 15px 0;
}

.warning-box {
    padding: 18px;
    border-radius: 18px;
    background: #fff7ed;
    border-left: 5px solid #f97316;
    margin: 15px 0;
}

.success-box {
    padding: 18px;
    border-radius: 18px;
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    margin: 15px 0;
}

/* Mobile */

@media (max-width: 768px) {

    .block-container {
        padding: 0.7rem 0.7rem 2rem 0.7rem;
    }

    .hero {
        padding: 22px;
        border-radius: 20px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .hero p {
        font-size: 14px;
    }

    .metric-card {
        margin-bottom: 10px;
    }

    .prediction-score {
        font-size: 50px;
    }

    .section-title {
        font-size: 22px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 13px;
        padding: 8px 10px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "student_mental_health_model.joblib"

TARGET = "Mental_Health_Score"

FEATURES = [
    "Age",
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level"
]

# Values visible in your dataset
AGE_RANGE = (18, 24)
USAGE_RANGE = (1.0, 8.8)
UNLOCK_RANGE = (62, 273)
STUDY_RANGE = (0.3, 8.3)
ACTIVITY_RANGE = (-0.4, 4.1)
SLEEP_RANGE = (3.6, 9.9)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not Path(MODEL_PATH).exists():
        return None

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_default_dataset():

    possible_files = [
        "Student Social Media And Mental Health Impact.csv",
        "student_mental_health.csv",
        "mental_health.csv",
        "data.csv"
    ]

    for file in possible_files:

        if Path(file).exists():

            try:
                return pd.read_csv(file)

            except:
                pass

    return None


default_df = load_default_dataset()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🧠 Student Mental Health AI</h1>

<p>
AI-powered Student Mental Health Score Prediction &
Interactive Analytics Dashboard
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL STATUS
# ============================================================

if model is not None:

    st.markdown("""
    <div class="success-box">
    🟢 <b>AI Model Connected</b><br>
    Gradient Boosting Regression pipeline is ready for prediction.
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="warning-box">
    🔴 <b>Model file not found</b><br>
    Put <b>student_mental_health_model.joblib</b>
    in the same folder as app.py.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATA UPLOAD
# ============================================================

with st.expander("📂 Upload Dataset / Use Dataset"):

    uploaded_file = st.file_uploader(
        "Upload your Student Mental Health CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.success(
                f"Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns"
            )

        except Exception as e:

            st.error(f"Unable to read CSV: {e}")

    else:

        df = default_df

        if df is not None:
            st.info(
                f"Using local dataset: {df.shape[0]} rows × {df.shape[1]} columns"
            )
        else:
            df = None


# ============================================================
# TOP NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🧠 Prediction",
    "📊 Dashboard",
    "🤖 Model",
    "ℹ️ About"
])


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">📌 Project Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    if df is not None:

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">👨‍🎓</div>
                <div class="metric-title">Students</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📋</div>
                <div class="metric-title">Features</div>
                <div class="metric-value">{len(FEATURES)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-title">Best R²</div>
                <div class="metric-value">87.40%</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">🤖</div>
                <div class="metric-title">Best Model</div>
                <div class="metric-value" style="font-size:20px">
                Gradient Boosting
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.info("Upload your CSV to activate the analytics dashboard.")


    st.markdown(
        '<div class="section-title">✨ What this application does</div>',
        unsafe_allow_html=True
    )

    features = [
        ("🧠", "AI Prediction",
         "Predict a student's mental health score using the trained ML model."),

        ("📊", "Interactive Dashboard",
         "Explore student behavior, social media usage, sleep, stress and activity."),

        ("📈", "Model Evaluation",
         "Compare the six regression models used in the project."),

        ("🔍", "Feature Analysis",
         "Understand which features contribute most to the model."),

        ("💡", "Smart Insights",
         "Generate simple lifestyle-oriented observations from the entered profile."),

        ("📥", "Download Results",
         "Download prediction information as CSV.")
    ]

    cols = st.columns(2)

    for i, item in enumerate(features):

        icon, title, description = item

        with cols[i % 2]:

                        card_html = f"""
<div class="metric-card" style="margin-bottom:15px">

    <div style="font-size:30px">{icon}</div>

    <h3 style="margin-bottom:5px">
        {title}
    </h3>

    <p style="color:#64748b">
        {description}
    </p>

</div>
"""

        st.html(card_html)


# ============================================================
# TAB 2 - PREDICTION
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🧠 Predict Mental Health Score</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">

    Enter the student's information below and click
    <b>Predict Mental Health Score</b>.

    The prediction is generated by the trained regression pipeline.

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.markdown("### 👤 Personal Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=60,
            value=21,
            step=1
        )

    with c2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    with c3:

        country = st.selectbox(
            "Country",
            [
                "India",
                "USA",
                "Canada",
                "UK",
                "Australia",
                "Germany",
                "China",
                "Japan",
                "Other"
            ]
        )


    # --------------------------------------------------------
    # ACADEMIC
    # --------------------------------------------------------

    st.markdown("### 🎓 Academic & Social Profile")

    c1, c2, c3 = st.columns(3)

    with c1:

        academic_level = st.selectbox(
            "Academic Level",
            [
                "High School",
                "Undergraduate",
                "Graduate"
            ]
        )

    with c2:

        platform = st.selectbox(
            "Most Used Platform",
            [
                "Instagram",
                "YouTube",
                "Facebook",
                "TikTok",
                "Snapchat",
                "LinkedIn",
                "Twitter",
                "WeChat",
                "Other"
            ]
        )

    with c3:

        purpose = st.selectbox(
            "Purpose Of Use",
            [
                "Entertainment",
                "Education",
                "Networking",
                "Communication",
                "Other"
            ]
        )


    # --------------------------------------------------------
    # DAILY HABITS
    # --------------------------------------------------------

    st.markdown("### 📱 Daily Habits")

    c1, c2 = st.columns(2)

    with c1:

        usage_hours = st.slider(
            "Average Daily Social Media Usage (hours)",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.1
        )

        daily_unlocks = st.slider(
            "Daily Phone / App Unlocks",
            min_value=0,
            max_value=400,
            value=171,
            step=1
        )

        study_hours = st.slider(
            "Study Hours per Day",
            min_value=0.0,
            max_value=12.0,
            value=3.0,
            step=0.1
        )

    with c2:

        activity_hours = st.slider(
            "Physical Activity Hours",
            min_value=0.0,
            max_value=8.0,
            value=2.0,
            step=0.1
        )

        sleep_hours = st.slider(
            "Sleep Hours per Night",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.1
        )

        stress = st.selectbox(
            "Stress Level",
            [
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Mental Health Score",
        use_container_width=True
    )


    if predict_button:

        if model is None:

            st.error(
                "Model not found. Please place "
                "student_mental_health_model.joblib "
                "next to app.py."
            )

        else:

            input_data = pd.DataFrame([{

                "Age": age,

                "Gender": gender,

                "Country": country,

                "Academic_Level": academic_level,

                "Most_Used_Platform": platform,

                "Purpose_Of_Use": purpose,

                "Avg_Daily_Usage_Hours": usage_hours,

                "Daily_Unlocks": daily_unlocks,

                "Study_Hours": study_hours,

                "Physical_Activity_Hours": activity_hours,

                "Sleep_Hours_Per_Night": sleep_hours,

                "Stress_Level": stress

            }])

            try:

                prediction = float(
                    model.predict(input_data)[0]
                )

                st.session_state.last_prediction = prediction

                # ------------------------------------------------
                # HISTORY
                # ------------------------------------------------

                record = input_data.copy()

                record["Predicted_Mental_Health_Score"] = prediction

                st.session_state.history.append(record)

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">🎯 Prediction Result</div>',
                    unsafe_allow_html=True
                )

                if prediction >= 7:

                    level = "Higher Score Range"
                    emoji = "😊"

                elif prediction >= 5:

                    level = "Middle Score Range"
                    emoji = "🙂"

                else:

                    level = "Lower Score Range"
                    emoji = "⚠️"

                prediction_html = f"""
                <div class="prediction-card">

                    <div style="font-size:40px">
                        {emoji}
                    </div>

                    <div class="prediction-score">
                        {prediction:.2f}
                    </div>

                    <div class="prediction-label">
                        {level}
                    </div>

                    <p style="color:#64748b">
                        Predicted Mental Health Score
                    </p>

                </div>
                """

                st.html(prediction_html)

                # ------------------------------------------------
                # GAUGE
                # ------------------------------------------------

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        title={
                            "text": "Predicted Score"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 10]
                            },
                            "bar": {
                                "color": "#6366f1"
                            },
                            "steps": [
                                {
                                    "range": [0, 5],
                                    "color": "#fee2e2"
                                },
                                {
                                    "range": [5, 7],
                                    "color": "#fef3c7"
                                },
                                {
                                    "range": [7, 10],
                                    "color": "#dcfce7"
                                }
                            ]
                        }
                    )
                )

                fig.update_layout(
                    height=350,
                    margin=dict(
                        l=20,
                        r=20,
                        t=50,
                        b=20
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # ------------------------------------------------
                # PROFILE SUMMARY
                # ------------------------------------------------

                st.markdown("### 👤 Profile Summary")

                summary_cols = st.columns(4)

                profile = [
                    ("Age", age),
                    ("Social Usage", f"{usage_hours:.1f} hrs"),
                    ("Study", f"{study_hours:.1f} hrs"),
                    ("Sleep", f"{sleep_hours:.1f} hrs")
                ]

                for col, item in zip(summary_cols, profile):

                    with col:

                                                st.html(
                            f"""
                            <div class="metric-card">

                                <div class="metric-title">
                                    {item[0]}
                                </div>

                                <div class="metric-value">
                                    {item[1]}
                                </div>

                            </div>
                            """
                        )

                # ------------------------------------------------
                # SIMPLE INSIGHTS
                # ------------------------------------------------

                st.markdown("### 💡 Profile Insights")

                insights = []

                if usage_hours > 6:

                    insights.append(
                        "📱 Social media usage is relatively high compared with the dataset's typical usage."
                    )

                if usage_hours < 3:

                    insights.append(
                        "📱 Social media usage is relatively low."
                    )

                if sleep_hours < 6:

                    insights.append(
                        "😴 Sleep duration is below 6 hours."
                    )

                elif sleep_hours >= 7:

                    insights.append(
                        "😴 Sleep duration is 7 hours or more."
                    )

                if study_hours < 2:

                    insights.append(
                        "📚 Study time is relatively low."
                    )

                elif study_hours >= 5:

                    insights.append(
                        "📚 Study time is relatively high."
                    )

                if activity_hours < 1:

                    insights.append(
                        "🏃 Physical activity is relatively low."
                    )

                elif activity_hours >= 2:

                    insights.append(
                        "🏃 Physical activity is 2 hours or more."
                    )

                if stress in ["High", "Very High"]:

                    insights.append(
                        "😟 Reported stress level is high."
                    )

                else:

                    insights.append(
                        "🙂 Reported stress level is Low/Medium."
                    )

                for insight in insights:

                    st.markdown(
                        f"""
                        <div class="info-box">
                            {insight}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.caption(
                    "⚠️ These insights are educational observations based on "
                    "the entered variables. They are not a medical diagnosis."
                )

                # ------------------------------------------------
                # DOWNLOAD RESULT
                # ------------------------------------------------

                csv = input_data.copy()

                csv["Predicted_Mental_Health_Score"] = prediction

                csv_data = csv.to_csv(index=False)

                st.download_button(
                    "📥 Download Prediction",
                    data=csv_data,
                    file_name="mental_health_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

if len(st.session_state.history) > 0:

        st.markdown("---")

        st.markdown("### 📋 Prediction History")

        history_df = pd.concat(
            st.session_state.history,
            ignore_index=True
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        history_csv = history_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Prediction History",
            history_csv,
            "prediction_history.csv",
            "text/csv",
            use_container_width=True
        )


# ============================================================
# TAB 3 - DASHBOARD
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">📊 Interactive Student Dashboard</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.warning(
            "Please upload your CSV dataset to activate this dashboard."
        )

    else:

        # ----------------------------------------------------
        # DATA VALIDATION
        # ----------------------------------------------------

        missing_features = [
            col for col in FEATURES + [TARGET]
            if col not in df.columns
        ]

        if missing_features:

            st.error(
                "Missing columns: "
                + ", ".join(missing_features)
            )

        else:

            # ------------------------------------------------
            # FILTERS
            # ------------------------------------------------

            st.markdown("### 🎛️ Dashboard Filters")

            c1, c2, c3 = st.columns(3)

            with c1:

                selected_gender = st.multiselect(
                    "Gender",
                    sorted(
                        df["Gender"]
                        .dropna()
                        .unique()
                    ),
                    default=sorted(
                        df["Gender"]
                        .dropna()
                        .unique()
                    )
                )

            with c2:

                selected_stress = st.multiselect(
                    "Stress Level",
                    sorted(
                        df["Stress_Level"]
                        .dropna()
                        .unique()
                    ),
                    default=sorted(
                        df["Stress_Level"]
                        .dropna()
                        .unique()
                    )
                )

            with c3:

                selected_platform = st.multiselect(
                    "Platform",
                    sorted(
                        df["Most_Used_Platform"]
                        .dropna()
                        .unique()
                    ),
                    default=sorted(
                        df["Most_Used_Platform"]
                        .dropna()
                        .unique()
                    )
                )


            filtered_df = df[
                df["Gender"].isin(selected_gender)
                &
                df["Stress_Level"].isin(selected_stress)
                &
                df["Most_Used_Platform"].isin(selected_platform)
            ]


            # ------------------------------------------------
            # KPI CARDS
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "👨‍🎓 Students",
                    len(filtered_df)
                )

            with c2:

                st.metric(
                    "🧠 Avg Mental Health Score",
                    f"{filtered_df[TARGET].mean():.2f}"
                )

            with c3:

                st.metric(
                    "📱 Avg Usage",
                    f"{filtered_df['Avg_Daily_Usage_Hours'].mean():.2f} h"
                )

            with c4:

                st.metric(
                    "😴 Avg Sleep",
                    f"{filtered_df['Sleep_Hours_Per_Night'].mean():.2f} h"
                )


            # ------------------------------------------------
            # CHART 1
            # ------------------------------------------------

            st.markdown("### 📱 Social Media Usage vs Mental Health")

            fig1 = px.scatter(
                filtered_df,
                x="Avg_Daily_Usage_Hours",
                y="Mental_Health_Score",
                color="Stress_Level",
                hover_data=[
                    "Age",
                    "Gender",
                    "Most_Used_Platform",
                    "Study_Hours",
                    "Sleep_Hours_Per_Night"
                ],
                title="Daily Social Media Usage vs Mental Health Score"
            )

            fig1.update_layout(
                height=450
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )


            # ------------------------------------------------
            # CHART 2
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            with c1:

                platform_avg = (
                    filtered_df
                    .groupby("Most_Used_Platform")[TARGET]
                    .mean()
                    .sort_values()
                    .reset_index()
                )

                fig2 = px.bar(
                    platform_avg,
                    x=TARGET,
                    y="Most_Used_Platform",
                    orientation="h",
                    title="Average Mental Health Score by Platform"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )


            with c2:

                stress_avg = (
                    filtered_df
                    .groupby("Stress_Level")[TARGET]
                    .mean()
                    .reset_index()
                )

                fig3 = px.bar(
                    stress_avg,
                    x="Stress_Level",
                    y=TARGET,
                    color="Stress_Level",
                    title="Mental Health Score by Stress Level"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )


            # ------------------------------------------------
            # CHART 3
            # ------------------------------------------------

            st.markdown("### 😴 Sleep vs Mental Health")

            fig4 = px.scatter(
                filtered_df,
                x="Sleep_Hours_Per_Night",
                y=TARGET,
                size="Daily_Unlocks",
                color="Academic_Level",
                hover_data=[
                    "Age",
                    "Gender",
                    "Stress_Level"
                ],
                title="Sleep Hours vs Mental Health Score"
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )


            # ------------------------------------------------
            # CHART 4
            # ------------------------------------------------

            st.markdown("### 🎓 Academic Level Analysis")

            academic_avg = (
                filtered_df
                .groupby("Academic_Level")[TARGET]
                .mean()
                .reset_index()
            )

            fig5 = px.pie(
                academic_avg,
                names="Academic_Level",
                values=TARGET,
                hole=0.45,
                title="Mental Health Score Distribution by Academic Level"
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )


            # ------------------------------------------------
            # DATA TABLE
            # ------------------------------------------------

            with st.expander("📋 View Filtered Dataset"):

                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# TAB 4 - MODEL
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">🤖 Machine Learning Model</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    model_html = """
<div class="prediction-card">

    <div style="font-size:45px">
        🏆
    </div>

    <h2>
        Gradient Boosting Regressor
    </h2>

    <div class="prediction-score">
        87.40%
    </div>

    <div>
        R² Score
    </div>

</div>
"""

    st.html(model_html)


    st.markdown("### 📈 Model Comparison")


    model_results = pd.DataFrame({

        "Model": [
            "Gradient Boosting",
            "Extra Trees",
            "Random Forest",
            "Decision Tree",
            "Linear Regression",
            "Ridge Regression"
        ],

        "R² (%)": [
            87.40,
            87.23,
            86.82,
            82.40,
            79.34,
            79.31
        ],

        "MAE": [
            0.3597,
            0.3632,
            0.3572,
            0.4108,
            0.4692,
            0.4702
        ],

        "RMSE": [
            0.4675,
            0.4706,
            0.4781,
            0.5524,
            0.5986,
            0.5989
        ]

    })


    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MODEL CHART
    # --------------------------------------------------------

    fig_model = px.bar(
        model_results.sort_values("R² (%)"),
        x="R² (%)",
        y="Model",
        orientation="h",
        text="R² (%)",
        title="Regression Model Performance"
    )

    fig_model.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_model.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )


    # --------------------------------------------------------
    # METRICS EXPLANATION
    # --------------------------------------------------------

    st.markdown("### 📚 Evaluation Metrics")

    metric_data = pd.DataFrame({

        "Metric": [
            "R²",
            "MAE",
            "RMSE"
        ],

        "Meaning": [
            "Explains how much variance is captured by the model.",
            "Average absolute prediction error.",
            "Penalizes larger prediction errors more strongly."
        ],

        "Better": [
            "Higher",
            "Lower",
            "Lower"
        ]

    })


    st.dataframe(
        metric_data,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("### 🔍 Feature Importance")


    if model is not None:

        try:

            regressor = model.named_steps["regressor"]

            preprocessor = model.named_steps["preprocessor"]

            if hasattr(regressor, "feature_importances_"):

                importances = regressor.feature_importances_

                feature_names = preprocessor.get_feature_names_out()

                fi_df = pd.DataFrame({

                    "Feature": feature_names,

                    "Importance": importances

                }).sort_values(
                    "Importance",
                    ascending=False
                ).head(20)


                fig_fi = px.bar(
                    fi_df.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Feature Importances"
                )

                st.plotly_chart(
                    fig_fi,
                    use_container_width=True
                )

            else:

                st.info(
                    "Feature importance is not available for this model."
                )

        except Exception as e:

            st.info(
                f"Feature importance could not be displayed: {e}"
            )


# ============================================================
# TAB 5 - ABOUT
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    ## 🧠 Student Mental Health Score AI

    This project uses machine learning to estimate a student's
    **Mental Health Score** from demographic, academic,
    social-media and lifestyle-related variables.

    ### 🎯 Objective

    The objective is to demonstrate an end-to-end machine learning
    workflow:

    **Data → Cleaning → Preprocessing → Model Training →
    Evaluation → Prediction → Deployment**

    ### 📊 Dataset

    The project contains approximately **5,000 student records**
    and **13 columns**.

    The prediction target is:

    **Mental_Health_Score**

    ### 🤖 Models

    Six regression algorithms were evaluated:

    1. Linear Regression
    2. Ridge Regression
    3. Decision Tree
    4. Random Forest
    5. Gradient Boosting
    6. Extra Trees

    ### 🏆 Best Model

    Gradient Boosting achieved the strongest R² score in the
    notebook evaluation.

    ### 🚀 Deployment

    The trained model is stored as:

    `student_mental_health_model.joblib`

    The Streamlit application loads that pipeline and uses the
    same feature structure for prediction.

    """)


    st.markdown("""
    <div class="warning-box">

    ⚠️ <b>Important Disclaimer</b>

    <br><br>

    This application is an educational machine-learning project.
    It does not diagnose depression, anxiety, stress disorders,
    or any other medical condition.

    A machine-learning score should not replace professional
    mental-health assessment or medical advice.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:20px;
color:#64748b;
">

<b>🧠 Student Mental Health AI</b>

<br>

Built with Python • Scikit-learn • Streamlit • Plotly

<br><br>

Machine Learning • Data Science • AI

</div>
""", unsafe_allow_html=True)
