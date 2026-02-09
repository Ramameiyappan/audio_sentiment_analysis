import streamlit as st
import requests
import pandas as pd
import html

BACKEND_URL = "http://localhost:5000/analyze"

st.set_page_config(
    page_title="Emotion Analysis Dashboard",
    layout="wide"
)

if "df" not in st.session_state:
    st.session_state.df = None

st.markdown("""
<style>
body {
    background-color: #f8fafc;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f172a;
}

.subtitle {
    color: #475569;
    margin-bottom: 20px;
}

.timeline-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border-left: 6px solid #6366f1;
}

.time {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 600;
}

.text {
    font-size: 1rem;
    color: #1e293b;
    margin-top: 6px;
    margin-bottom: 12px;
    line-height: 1.6;
    max-width: 900px;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    color: white;
}

.happy { background: #22c55e; }
.sad { background: #3b82f6; }
.angry { background: #ef4444; }
.neutral { background: #64748b; }

.conf {
    font-size: 0.8rem;
    color: #475569;
    margin-left: 8px;
}

.insight-card {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    padding: 26px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 15px 30px rgba(34,197,94,0.35);
}

.insight-card.orange {
    background: linear-gradient(135deg, #f97316, #ea580c);
}

.insight-value {
    font-size: 2rem;
    font-weight: 800;
}

.insight-label {
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-title'>Emotion Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Audio Sentiment Analysis by using audio and text in that audio for confirmation</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload an audio file (wav / mp3)",
    type=["wav", "mp3"]
)

if uploaded:
    st.audio(uploaded)

    if st.button("Analyze Emotion", width="stretch"):
        with st.spinner("Backend can go to sleep so kindly wait for 3-4 min"):
            try:
                response = requests.post(
                    BACKEND_URL,
                    files={"audio": uploaded.getvalue()},
                    timeout=240
                )
                response.raise_for_status()
            
                data = response.json()["timeline"]
                df = pd.DataFrame(data)
                st.session_state.df = df
            except Exception as e:
                st.error(f"Backend Error: {e}")
                st.stop()

df = st.session_state.df
if df is not None and not df.empty:
    st.subheader("Key Insights")

    df["duration"] = df["end_time"] - df["start_time"]

    dominant_emotion = (
        df.groupby("emotion", as_index=False)["duration"]
        .sum()
        .sort_values("duration", ascending=False)
        .iloc[0]["emotion"]
    )

    avg_conf = round(df["confidence"].mean(), 3)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-value">{dominant_emotion.upper()}</div>
                <div class="insight-label">Dominant Emotion</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="insight-card orange">
                <div class="insight-value">{avg_conf}</div>
                <div class="insight-label">Average Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📜 Sentence-Level Emotion Timeline")

    for _, row in df.iterrows():
        safe_text = html.escape(row["text"])
    
        html_block = f"""
            <div class="timeline-card">
                <div class="time">⏱ {row['start_time']}s – {row['end_time']}s</div>
                <div class="text">“{safe_text}”</div>
                <span class="badge {row['emotion'].lower()}">{row['emotion'].upper()}</span>
                <span class="conf">Confidence: {row['confidence']:.2f}</span>
            </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)

elif uploaded:
    st.info("Click **Analyze Emotion** to see results.")
else:
    st.info("⬆️ Upload an audio file to begin.")
