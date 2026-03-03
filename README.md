# 🧠 Big Five Personality Predictor

Predict someone's personality (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from just a few lines of text.

I built this project out of pure curiosity —  
Can machines understand human personality the way we do?

Turns out… they can get surprisingly close.

---

## 🚀 Project Overview

Traditional personality tests are long, questionnaire-based, and often biased.  
This project explores a different idea:

> What if we could infer personality traits simply from how someone writes?

Using short behavioral statements (CV-style text), I trained:

- 6 Classical Machine Learning models  
- A fine-tuned BERT model  
- And built an interactive Streamlit web app to test everything live

---

## 📊 Dataset

- **File:** `personality_dataset_10000.csv`
- **Samples:** 10,000
- **Input:** Short text statements
- **Output:** Binary labels (0/1) for all 5 Big Five traits
- **Traits:**  
  - Openness  
  - Conscientiousness  
  - Extraversion  
  - Agreeableness  
  - Neuroticism  

*(Dataset stored privately in Google Drive. Available for academic collaboration.)*

---

## 🧪 Models Compared

| Model                | Macro F1 | Hamming Loss | Notes                     |
|----------------------|----------|--------------|---------------------------|
| Logistic Regression  | 0.5261   | 0.4990       | Strong baseline           |
| SVM                  | 0.5261   | 0.4990       | Very stable               |
| Random Forest        | 0.5261   | 0.4990       | Good all-rounder          |
| Decision Tree        | 0.5261   | 0.4990       | Fast but simple           |
| Naive Bayes          | 0.5050   | 0.4991       | Text-friendly             |
| KNN                  | 0.4034   | 0.4990       | Weakest performer         |
| **BERT (Fine-tuned)**| **0.533**| ~0.500       | Slight edge, slower       |

### 🔎 Key Observation

- Short text (65–105 characters) makes personality inference difficult.
- Most models show high recall but lower precision (bias toward positive class).
- Surprisingly, TF-IDF + Logistic Regression competes closely with BERT.

Sometimes simpler really is stronger.

---

## 🌐 Live Demo (Prototype)

👉 **[Try the Web App](https://authentication-maintains-assets-better.trycloudflare.com/)**  
*(Temporary link — will update with permanent deployment soon)*

Type a few sentences about yourself and choose a model.  
Instant predictions across all five traits.

---

## 🛠 Tech Stack

- Python 3.10+
- scikit-learn
- Hugging Face Transformers + PyTorch
- Streamlit
- pandas, matplotlib, seaborn
- joblib

---

## 📦 Installation

```bash
git clone https://github.com/sayed02-debug/big_five_personality.git
cd big_five_personality
pip install -r requirements.txt
```

---

## ▶️ Run the Web App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

Works fully offline once models are loaded.

---

## 📂 Project Structure

```
big_five_personality/
├── notebooks/
│   ├── 01_classical_ml_bigfive.ipynb
│   └── 02_bert_finetuning_bigfive.ipynb
├── app.py
├── figures/
├── models/
└── results/
```

---

## 📈 What This Project Demonstrates

- Multi-label classification
- Classical ML vs Transformer comparison
- Model evaluation using Macro F1 & Hamming Loss
- Confusion Matrix analysis per trait
- Deployment of ML models via Streamlit
- End-to-end ML workflow (data → training → evaluation → deployment)

---

## 🧠 Lessons Learned

- Simple TF-IDF + Linear models can be extremely competitive.
- Personality inference from short text lacks context depth.
- Binary labels inflate recall.
- Streamlit makes ML demo deployment incredibly fast.

---

## 🔮 Future Improvements

- RoBERTa / DistilBERT fine-tuning
- SHAP/LIME explainability
- Regression-based personality scoring (0–1 scale)
- Longer text support
- Multi-language (Bangla + English)
- Permanent cloud deployment (Hugging Face Spaces)

---

## ☕ Made With

Curiosity.  
Late-night debugging sessions.  
Endless Colab runtime disconnects.  
And way too much coffee.

---

### 👤 Md. Abu Sayed Islam

ML Enthusiast | Final Year Student  
Jagannath University, 2026  

📧 Email: mdabusayedislam2@gmail.com  
🔗 LinkedIn: https://linkedin.com/in/sayed02  
💻 GitHub: https://github.com/sayed02-debug  

---

✨ Open to collaborations, feedback, or just geeking out over ML — feel free to reach out!
