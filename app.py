# ============================================================
#       Big Five Personality Predictor - Streamlit App
#
# ============================================================

import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Big Five Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Header ────────────────────────────────────────────────
st.markdown(
    """
    <h2 style='text-align: center; color: #1a73e8; margin-bottom: 8px;'>
        🧠 Big Five Personality Predictor
    </h2>
    <p style='text-align: center; color: #555; margin-top: 0;'>
        Describe yourself in a few sentences — discover your Big Five traits
    </p>
    """,
    unsafe_allow_html=True
)

# ─── Input Area ────────────────────────────────────────────
with st.container():
    st.markdown("<div style='max-width: 720px; margin: 0 auto; padding: 0 16px;'>", unsafe_allow_html=True)

    user_text = st.text_area(
        "Tell us about yourself (or paste from CV / bio)",
        height=160,
        placeholder="Example: I am quite organized, love planning things in advance, but sometimes I worry too much about small details...",
        key="user_input"
    )

    model_choice = st.selectbox(
        "Choose a model",
        [
            "Logistic_Regression",
            "Linear_SVM",
            "SVM",
            "Naive_Bayes",
            "Random_Forest",
            "Decision_Tree",
            "KNN"
        ],
        index=0,
        help="Logistic Regression and Linear SVM usually give the best balance of accuracy and speed in this project."
    )

    predict_btn = st.button("Analyze My Personality →", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── Model name mapping (matches exactly what exists in your Drive) ──
MODEL_FILES = {
    "Logistic_Regression": "Logistic_Regression_model.pkl",
    "Linear_SVM":          "Linear_SVM_model.pkl",
    "SVM":                 "SVM_model.pkl",
    "Naive_Bayes":         "Naive_Bayes_model.pkl",
    "Random_Forest":       "Random_Forest_model.pkl",
    "Decision_Tree":       "Decision_Tree_model.pkl",
    "KNN":                 "KNN_model.pkl"
}

# ─── Prediction Logic ──────────────────────────────────────
if predict_btn:
    if not user_text.strip():
        st.warning("Please write something about yourself first 😊")
    else:
        with st.spinner("Analyzing your text..."):
            try:
                base_path = Path("/content/drive/MyDrive/Thesis_2026/Results")

                # Debug (visible in colab logs / terminal)
                st.session_state.debug_msg = f"Loading model: {MODEL_FILES[model_choice]}"

                vectorizer = joblib.load(base_path / "tfidf_vectorizer.pkl")
                model_path = base_path / MODEL_FILES[model_choice]
                model = joblib.load(model_path)

                # Clean text — same way as in training
                cleaned = (
                    user_text.lower()
                    .replace(r'[^a-z\s]', '')           # remove everything except letters & space
                    .replace(r'\s+', ' ')               # normalize multiple spaces
                    .strip()
                )

                if not cleaned:
                    st.warning("After cleaning, no meaningful text remained. Try writing more.")
                else:
                    X_input = vectorizer.transform([cleaned])

                    pred = model.predict(X_input)[0]

                    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

                    # ─── Result Cards ──────────────────────────────────────
                    cols = st.columns(5, gap="small")
                    for i, trait in enumerate(traits):
                        has_trait = bool(pred[i])
                        emoji = "✅" if has_trait else "❌"
                        color = "#27ae60" if has_trait else "#c0392b"
                        bg_color = "#e8f5e9" if has_trait else "#ffebee"

                        with cols[i]:
                            st.markdown(f"""
                                <div style="
                                    text-align: center;
                                    padding: 14px 6px;
                                    border-radius: 10px;
                                    background: {bg_color};
                                    border: 2px solid {color};
                                    min-height: 110px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: center;
                                ">
                                    <div style="font-size: 1.05rem; font-weight: 600; color: #333;">
                                        {trait}
                                    </div>
                                    <div style="font-size: 2.1rem; margin-top: 8px; color: {color};">
                                        {emoji}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    st.success("Analysis complete!")

            except FileNotFoundError as e:
                st.error(f"File not found: {str(e)}")
                st.info("Make sure:\n• Drive is mounted\n• Model files exist in /Thesis_2026/Results/\n• Model name matches selection")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("Check colab logs or runtime. You may need to restart and remount Drive.")

# ─── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("Thesis Prototype • Md. Abu Sayed Islam • 2026")
