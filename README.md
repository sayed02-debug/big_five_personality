# Big Five Personality Predictor 

Predict someone's personality (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) just from a few lines of text.

I built this project because I was curious — can machines actually understand human personality the way we do? Turns out, they can get pretty close.

### The Problem
Most personality tests are long questionnaires that feel boring and biased. What if we could predict the Big Five traits just by analyzing how someone writes or speaks? That's exactly what this project does.

I used short behavioral statements (CV-style text) and trained both classic ML models and BERT to see which one performs better in real life.

### Dataset
- **Name**: `personality_dataset_10000.csv`
- **Size**: 10,000 samples
- **Content**: Short text statements + binary labels (0/1) for all 5 Big Five traits
- **Source**: Custom curated dataset (originally inspired by openpsychometrics + Kaggle-style behavioral data)

*(Dataset is stored in Google Drive for privacy. If you need access for research, just DM me.)*

### What I Built

- **6 Traditional ML Models** (classic but powerful)
- **BERT Fine-tuning** (transformer approach)
- **Interactive Streamlit Web App** — you type something and instantly get personality predictions
- Full comparison, visualizations, and saved models

### Models I Compared

| Model                | Macro F1 | Hamming Loss | Notes                     |
|----------------------|----------|--------------|---------------------------|
| Logistic Regression  | 0.5261   | 0.4990       | Surprisingly strong       |
| SVM                  | 0.5261   | 0.4990       | Very stable               |
| Random Forest        | 0.5261   | 0.4990       | Good all-rounder          |
| Decision Tree        | 0.5261   | 0.4990       | Fast but basic            |
| Naive Bayes          | 0.5050   | 0.4991       | Text-friendly             |
| KNN                  | 0.4034   | 0.4990       | Weakest performer         |
| **BERT**             | **0.533** | ~0.500      | Slight edge but slower    |

### Live Demo (Prototype)

Try it yourself!  
→ **[Open the Web App](https://authentication-maintains-assets-better.trycloudflare.com/)** *(temporary link — will update when stable)*

Just type a few sentences about yourself and pick any model. Super fun to play with.

### Tech Stack

- Python
- scikit-learn
- Hugging Face Transformers + PyTorch (for BERT)
- Streamlit (for the web interface)
- pandas, matplotlib, seaborn


### Quick Project Structure

```text
big_five_personality/
├── notebooks/                                         # All Jupyter notebooks (core experiments)
│   ├── 01_classical_ml_bigfive.ipynb                  # 6 traditional ML models + comparison & results
│   └── 02_bert_finetuning_bigfive.ipynb               # BERT fine-tuning, training & evaluation
├── app.py                                             # Streamlit web app (prototype — run with `streamlit run app.py`)
├── figures/                                           # All generated plots, confusion matrices, comparison bars
├── models/                                            # Saved model files (.pkl) — load these in app.py
└── results/                                           # CSV tables, metrics summaries, raw outputs
```

---

## How to Run (Step-by-Step)

### 1. Notebooks (Experiments)

Open in Google Colab (recommended) or Jupyter Notebook  
Mount Google Drive to access the dataset  
Run cells top to bottom — everything is self-contained  

### 2. Streamlit Web App (The Fun Part)

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Browser-এ http://localhost:8501 খুলবে  
Type a few sentences about yourself, pick a model → get instant personality prediction  
Works offline once models are loaded  

Live Demo (temporary — updates regularly):  
Click to try the app  
(If link expired, just ping me — I can spin up a new one in 30 seconds)

---

## Results at a Glance

Best traditional model: Logistic Regression / SVM / Random Forest (tied at Macro F1 0.5261)  
BERT edge: Macro F1 0.533 (tiny improvement, but more compute-heavy)  
Interesting observation: All models struggle with short text (65-105 chars) — recall high, precision low (bias toward positive class)

---

## Tech Stack (What Powers This)

- Python 3.10+
- scikit-learn (classic ML)
- Hugging Face Transformers + PyTorch (BERT)
- Streamlit (web UI — super fast prototyping)
- pandas, matplotlib, seaborn (data viz)
- joblib (model saving/loading)

---

## Lessons I Learned (ML Enthusiast Notes)

- TF-IDF + simple models can surprisingly hold their own against BERT on short text  
- Short statements (CV-style) make personality inference really hard — context is king  
- Binary labels (0/1) lead to high recall but low precision — next step: regression scores (0-1 scale)  
- Streamlit is insanely productive for ML demos — from idea to live app in <1 hour  

---

## Future Ideas (What I'm Thinking Next)

- Fine-tune RoBERTa or DistilBERT (faster & lighter)  
- Add SHAP/LIME explainability (which words influence which trait?)  
- Support longer inputs (social media posts, essays)  
- Multi-language (Bangla + English)  
- Deploy permanently on Hugging Face Spaces or Render  

---

## Made With

Curiosity, late-night debugging sessions, endless Colab runtime disconnects,  
and way too much coffee ☕

**Md. Abu Sayed Islam**  
ML Enthusiast | Final Year Student  
Jagannath University, 2026  

📧 Email: mdabusayedislam2@gmail.com  
🔗 LinkedIn: https://linkedin.com/in/sayed02  
💻 GitHub: https://github.com/sayed02-debug  

✨ Open to collaborations, feedback, or just geeking out over ML — feel free to reach out!
