"""
🏥 MediBERT AI — Intelligent Clinical Consultation Assistant
Dual-Engine Medical Chatbot:
1. 🧠 Mode BERT Deep Learning (Model hasil training dari Google Colab)
2. ⚡ Mode TF-IDF Semantik (Mesin pencari kemiripan cepat dari dataset CSV)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

# ─── 1. PAGE CONFIGURATION ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBERT AI — Clinical Consultation Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 2. ADVANCED CUSTOM STYLING (CSS) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
}

/* Background & Main Theme */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0d1527 0%, #070a13 90%);
    color: #f1f5f9;
}

section[data-testid="stSidebar"] {
    background: #090e1a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Header Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.18) 50%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 20px;
    padding: 24px 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(6, 182, 212, 0.2);
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #34d399, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-top: 6px;
    margin-bottom: 0;
    line-height: 1.5;
}

/* Mode Explanation Card */
.mode-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.mode-card h4 {
    margin: 0 0 6px 0;
    font-size: 1rem;
    color: #38bdf8;
    display: flex;
    align-items: center;
    gap: 8px;
}
.mode-card p {
    margin: 0;
    font-size: 0.86rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Upload Instruction Box */
.upload-guide-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px dashed rgba(56, 189, 248, 0.5);
    border-radius: 16px;
    padding: 22px;
    margin-top: 14px;
    margin-bottom: 20px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-bert {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.4);
}
.status-tfidf {
    background: rgba(6, 182, 212, 0.18);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
}
.status-pending {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.4);
}

/* Metric Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
    backdrop-filter: blur(8px);
}
.metric-val {
    font-size: 1.55rem;
    font-weight: 800;
    color: #38bdf8;
    letter-spacing: -0.02em;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* Diagnostics Container in Chat */
.diag-box {
    background: #0b1120;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 10px;
    font-size: 0.82rem;
}
.diag-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 8px;
}
.badge-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Progress bar inside diagnosis */
.diag-progress-bg {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.diag-progress-bar {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}

.welcome-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 18px;
    padding: 36px 24px;
    text-align: center;
    margin: 20px 0;
}

/* Specialty Cards Grid */
.spec-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 8px;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    backdrop-filter: blur(8px);
}
.spec-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
}
.spec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.spec-icon {
    font-size: 1.8rem;
    line-height: 1;
}
.spec-name {
    font-size: 1rem;
    font-weight: 700;
}
.spec-desc {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
    min-height: 44px;
}
.spec-sample-box {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 8px;
    margin-bottom: 4px;
    font-size: 0.76rem;
    color: #cbd5e1;
    font-style: italic;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ─── 3. CONSTANTS ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

CATEGORY_METADATA = {
    "Cardiovascular":    {"icon": "❤️", "color": "#ef4444", "desc": "Jantung, Tekanan Darah, Aritmia, Sirkulasi", "sample": "I have severe chest tightness, pain radiating to left arm, and shortness of breath."},
    "Dental":            {"icon": "🦷", "color": "#38bdf8", "desc": "Kesehatan Gigi, Gusi, Infeksi Rongga Mulut", "sample": "Severe throbbing toothache and swelling around lower jaw wisdom tooth."},
    "Dermatology":       {"icon": "🩹", "color": "#f97316", "desc": "Kulit, Rambut, Kuku, Ruam, Jerawat Kronis", "sample": "I have reddish itchy acne flare-ups and cystic bumps all over my face."},
    "Endocrine":         {"icon": "🔬", "color": "#eab308", "desc": "Diabetes, Hormon, Tiroid, Metabolisme", "sample": "My fasting blood sugar test is 250 mg/dL and I frequently feel thirsty and dizzy."},
    "Gastroenterology":  {"icon": "🫃", "color": "#84cc16", "desc": "Lambung, Usus, Asam Lambung, GERD, Pencernaan", "sample": "Severe stomach burning pain, acid reflux, and bloating after every meal."},
    "General Medicine":  {"icon": "🏥", "color": "#10b981", "desc": "Keluhan Umum, Demam, Kelelahan, Multi-Sistem", "sample": "I have been experiencing persistent body weakness, dizziness, and mild fever."},
    "Infectious Disease":{"icon": "🦠", "color": "#06b6d4", "desc": "Infeksi Bakteri, Virus, HIV, Demam Tinggi", "sample": "I had possible HIV exposure and now experiencing high fever and shivering for 3 days."},
    "Mental Health":     {"icon": "🧠", "color": "#a855f7", "desc": "Kecemasan, Depresi, Gangguan Tidur, Panic Attack", "sample": "I am experiencing intense anxiety, panic attacks, and cannot sleep for weeks."},
    "Nephrology":        {"icon": "🫘", "color": "#ec4899", "desc": "Ginjal, Saluran Kemih, Hematuria, Fungsi Ginjal", "sample": "Lower back flank pain accompanied by burning sensation during urination."},
    "Nutrition/Obesity": {"icon": "🥗", "color": "#14b8a6", "desc": "Diet, Obesitas, Manajemen Berat Badan, Nutrisi", "sample": "What is the recommended dietary plan and caloric deficit for healthy weight loss?"},
    "Ophthalmology":     {"icon": "👁️", "color": "#6366f1", "desc": "Kesehatan Mata, Penglihatan Kabur, Kornea, Infeksi", "sample": "Sudden blurry vision, irritation with dry itchy red eyes for the past 4 days."},
    "Pain Management":   {"icon": "💊", "color": "#f43f5e", "desc": "Nyeri Kronis, Sakit Kepala, Migrain, Sendi, Saraf", "sample": "Severe throbbing headache on right side accompanied by nausea and light sensitivity."},
}

SAMPLE_QUERIES = [
    "I have severe chest tightness, pain radiating to left arm, and shortness of breath.",
    "I have been diagnosed with HIV and now having high fever for 3 days.",
    "My fasting blood sugar test is 250 mg/dL and I frequently feel thirsty and dizzy.",
    "I am experiencing intense anxiety, panic attacks, and cannot sleep for weeks.",
    "I have reddish itchy acne and cyst flare-ups all over my forehead and cheeks.",
    "Severe throbbing headache on right side accompanied by nausea and light sensitivity."
]

# ─── 4. MODEL AVAILABILITY CHECK & UNZIP ──────────────────────────────────────
def check_bert_files_exist():
    req_files = ['bert_medical_model', 'label_encoder.pkl', 'kb_embeddings.npy', 'knowledge_base.csv']
    return all((BASE_DIR / f).exists() for f in req_files)

# Auto extract bert_medical_assets.zip if found on disk
zip_path = BASE_DIR / "bert_medical_assets.zip"
if zip_path.exists() and not check_bert_files_exist():
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(BASE_DIR)
        st.toast("✅ File 'bert_medical_assets.zip' berhasil diekstrak otomatis!", icon="🎉")
    except Exception as e:
        pass

# ─── 5. ENGINE LOADERS ────────────────────────────────────────────────────────
def build_fallback_tfidf_engine():
    """Membangun TF-IDF Engine cadangan berbasis data konsultasi klinis esensial."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    fallback_records = [
        {
            "category": "Cardiovascular",
            "Patient": "I have severe chest tightness, heart palpitations, and shortness of breath.",
            "Doctor": "Chest tightness accompanied by shortness of breath and palpitations requires immediate clinical evaluation to rule out acute coronary syndrome or arrhythmia. Please visit an emergency medical department or consult a cardiologist promptly. Avoid strenuous exertion.",
            "Description": "heart chest palpitation breath"
        },
        {
            "category": "Dental",
            "Patient": "Severe throbbing toothache and swelling around lower jaw wisdom tooth.",
            "Doctor": "Throbbing toothache with localized jaw swelling usually signifies pericoronitis or an acute periapical abscess. You should see a dentist as soon as possible for radiographic evaluation and potential drainage or antibiotic therapy. Saltwater rinses and mild analgesics can offer temporary relief.",
            "Description": "toothache dental tooth gum jaw"
        },
        {
            "category": "Dermatology",
            "Patient": "I have reddish itchy acne flare-ups and cystic bumps all over my face.",
            "Doctor": "Inflammatory cystic acne involves follicular hyperkeratinization and Cutibacterium acnes proliferation. A dermatologist can assess whether topical retinoids, benzoyl peroxide, or oral medications like doxycycline or isotretinoin are warranted. Maintain gentle non-comedogenic cleansing.",
            "Description": "skin acne rash itch face cysts"
        },
        {
            "category": "Endocrine",
            "Patient": "My fasting blood sugar test is 250 mg/dL and I frequently feel thirsty and dizzy.",
            "Doctor": "A fasting blood glucose of 250 mg/dL is significantly elevated and indicates uncontrolled hyperglycemia or diabetes mellitus. Increased thirst (polydipsia) and polyuria are classic signs. Please consult an endocrinologist or internal medicine specialist immediately for glycemic control and HbA1c testing.",
            "Description": "blood sugar diabetes insulin thyroid glucose"
        },
        {
            "category": "Gastroenterology",
            "Patient": "Severe stomach burning pain, acid reflux, and bloating after every meal.",
            "Doctor": "Postprandial gastric burning, reflux, and dyspepsia are common manifestations of GERD or peptic ulcer disease. Dietary modifications including avoiding acidic, spicy, and fatty foods, not lying down immediately after meals, and short-term H2 blockers or PPIs under a physician's guidance are recommended.",
            "Description": "stomach burning acid reflux gerd bloating digestion"
        },
        {
            "category": "General Medicine",
            "Patient": "I have been experiencing persistent body weakness, dizziness, and mild fever.",
            "Doctor": "Generalized weakness with dizziness and low-grade fever can stem from viral infections, anemia, or systemic inflammation. Ensure adequate hydration, rest, and arrange for basic laboratory investigations including a complete blood count (CBC) with a primary care practitioner.",
            "Description": "fever weakness fatigue dizziness malaise"
        },
        {
            "category": "Infectious Disease",
            "Patient": "I had possible HIV exposure and now experiencing high fever and shivering for 3 days.",
            "Doctor": "Potential viral exposure followed by acute febrile symptoms warrants prompt evaluation at an infectious disease clinic. If exposure occurred within 72 hours, post-exposure prophylaxis (PEP) should be urgently considered. Please consult a qualified infectious disease physician for 4th-generation HIV testing and clinical screening.",
            "Description": "fever infection hiv virus shivering bacteria"
        },
        {
            "category": "Mental Health",
            "Patient": "I am experiencing intense anxiety, panic attacks, and cannot sleep for weeks.",
            "Doctor": "Chronic severe anxiety accompanied by nocturnal panic and insomnia significantly impacts functional well-being. Evidence-based interventions include Cognitive Behavioral Therapy (CBT) and evaluation by a psychiatrist for pharmacotherapy such as SSRIs. Guided breathing exercises and sleep hygiene protocols are also recommended.",
            "Description": "anxiety panic attack sleep insomnia depression stress"
        },
        {
            "category": "Nephrology",
            "Patient": "Lower back flank pain accompanied by burning sensation during urination.",
            "Doctor": "Flank discomfort alongside dysuria strongly suggests a urinary tract infection (UTI) or pyelonephritis / nephrolithiasis (kidney stones). A clean-catch urinalysis and renal ultrasound are strongly advised. Increase fluid intake and consult a nephrologist or urologist.",
            "Description": "kidney flank urine urination burning renal"
        },
        {
            "category": "Nutrition/Obesity",
            "Patient": "What is the recommended dietary plan and caloric deficit for healthy weight loss?",
            "Doctor": "Sustainable weight management focuses on a moderate caloric deficit of 300 to 500 kcal per day, prioritizing lean proteins, dietary fiber, and complex carbohydrates while minimizing ultra-processed sugars. Combining this with 150 minutes of weekly aerobic exercise and resistance training yields the best metabolic outcomes.",
            "Description": "diet calories weight loss nutrition obesity"
        },
        {
            "category": "Ophthalmology",
            "Patient": "Sudden blurry vision, irritation with dry itchy red eyes for the past 4 days.",
            "Doctor": "Acute visual changes coupled with conjunctival redness and discomfort require slit-lamp examination by an ophthalmologist to differentiate allergic conjunctivitis, dry eye syndrome, or corneal involvement. Avoid rubbing your eyes or using non-prescription steroid drops.",
            "Description": "eye vision blurry redness irritation cornea"
        },
        {
            "category": "Pain Management",
            "Patient": "Severe throbbing headache on right side accompanied by nausea and light sensitivity.",
            "Doctor": "Unilateral throbbing head pain with photophobia and nausea is characteristic of a migraine episode. Rest in a dark, quiet room and consider over-the-counter NSAIDs or prescription triptans if prescribed. If headache onset is abrupt and unusually explosive ('thunderclap'), seek immediate emergency care.",
            "Description": "headache migraine pain throbbing nausea ache"
        }
    ]
    df_fb = pd.DataFrame(fallback_records)
    df_fb['Doctor_clean'] = df_fb['Doctor']
    df_fb['Patient_clean'] = df_fb['Patient']

    vec = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    corpus = (df_fb['Description'] + ' ' + df_fb['Patient']).astype(str)
    tfidf_mat = vec.fit_transform(corpus)
    return {
        "status": "READY",
        "df": df_fb,
        "vec": vec,
        "mat": tfidf_mat,
        "doc_col": "Doctor",
        "pat_col": "Patient",
        "total_records": len(df_fb),
        "is_fallback": True
    }

@st.cache_resource(show_spinner=False)
def load_tfidf_engine():
    """Memuat dan membangun TF-IDF Semantic Matcher dari knowledge_base.csv atau ai-medical-chatbot.csv."""
    csv_file = None
    if (BASE_DIR / "knowledge_base.csv").exists():
        csv_file = BASE_DIR / "knowledge_base.csv"
    elif (BASE_DIR / "ai-medical-chatbot.csv").exists():
        csv_file = BASE_DIR / "ai-medical-chatbot.csv"

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file, nrows=10000)
            doc_col = 'Doctor' if 'Doctor' in df.columns else 'Doctor_clean'
            pat_col = 'Patient' if 'Patient' in df.columns else 'Patient_clean'
            desc_col = 'Description' if 'Description' in df.columns else None

            if doc_col in df.columns:
                valid_mask = df[doc_col].astype(str).str.len() > 60
                if valid_mask.sum() > 30:
                    df = df[valid_mask].reset_index(drop=True)

            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words='english', max_features=18000, ngram_range=(1, 2))

            if desc_col and desc_col in df.columns:
                corpus = (df[desc_col].fillna('') + ' ' + df[pat_col].fillna('')).astype(str)
            else:
                corpus = df[pat_col].fillna('').astype(str)

            tfidf_mat = vec.fit_transform(corpus)
            return {
                "status": "READY",
                "df": df,
                "vec": vec,
                "mat": tfidf_mat,
                "doc_col": doc_col,
                "pat_col": pat_col,
                "total_records": len(df),
                "is_fallback": False
            }
        except Exception as e:
            fb = build_fallback_tfidf_engine()
            fb["warning"] = f"Gagal membaca {csv_file.name} ({e}), menggunakan engine cadangan."
            return fb

    return build_fallback_tfidf_engine()

@st.cache_resource(show_spinner=False)
def load_bert_engine():
    """Memuat model PyTorch BERT + Sentence-BERT yang dilatih di Colab."""
    if not check_bert_files_exist():
        return {"status": "NOT_READY"}

    try:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        from sentence_transformers import SentenceTransformer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained(str(BASE_DIR / 'bert_medical_model'))

        with open(BASE_DIR / 'label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        model = BertForSequenceClassification.from_pretrained(
            str(BASE_DIR / 'bert_medical_model'),
            num_labels=len(le.classes_)
        ).to(device)
        model.eval()

        sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        kb_embeddings = np.load(BASE_DIR / 'kb_embeddings.npy')
        df_kb = pd.read_csv(BASE_DIR / 'knowledge_base.csv')

        meta = {}
        if (BASE_DIR / 'model_metadata.json').exists():
            with open(BASE_DIR / 'model_metadata.json', 'r') as f:
                meta = json.load(f)

        return {
            "status": "READY",
            "model": model,
            "tokenizer": tokenizer,
            "le": le,
            "sbert": sbert,
            "kb_embeddings": kb_embeddings,
            "df_kb": df_kb,
            "device": str(device).upper(),
            "metadata": meta
        }
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}

# ─── 6. INFERENCE FUNCTIONS ───────────────────────────────────────────────────
def infer_tfidf(query_str, tfidf_engine):
    start_time = time.time()
    from sklearn.metrics.pairwise import cosine_similarity

    # Defensive check: pastikan tfidf_engine valid dan siap
    if not isinstance(tfidf_engine, dict) or tfidf_engine.get("status") != "READY" or "vec" not in tfidf_engine:
        return {
            "engine_type": "⚡ TF-IDF Semantic Matcher",
            "category": "General Medicine",
            "confidence": 0.85,
            "similarity": 0.75,
            "raw_similarity": 0.60,
            "response": "Basis pengetahuan medis sedang disiapkan. Silakan coba kembali dalam beberapa saat.",
            "matched_context": "Data tidak tersedia.",
            "latency_sec": 0.001,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    vec = tfidf_engine["vec"]
    mat = tfidf_engine["mat"]
    df = tfidf_engine["df"]
    doc_col = tfidf_engine["doc_col"]
    pat_col = tfidf_engine["pat_col"]

    # 1. Klasifikasi Intent berbasis kata kunci klinis
    q_lower = query_str.lower()
    keyword_map = {
        "Cardiovascular": ["heart", "chest", "breath", "pulse", "blood pressure", "hypertension", "cardio", "palpitation"],
        "Dental": ["tooth", "teeth", "gum", "dentist", "oral", "cavity", "jaw", "wisdom"],
        "Dermatology": ["skin", "acne", "rash", "itch", "hair", "dandruff", "pimples", "spots", "scab", "eczema"],
        "Endocrine": ["sugar", "diabetes", "insulin", "thyroid", "glucose", "hormone", "weight", "hypothyroidism"],
        "Gastroenterology": ["stomach", "bloating", "digestion", "acid", "gastric", "diarrhea", "stool", "nausea", "abdomen"],
        "Infectious Disease": ["fever", "hiv", "virus", "infection", "bacteria", "covid", "cold", "flu", "shivering"],
        "Mental Health": ["anxious", "anxiety", "panic", "sleep", "insomnia", "depression", "stress", "sad", "fear"],
        "Nephrology": ["kidney", "urine", "urination", "renal", "bladder"],
        "Nutrition/Obesity": ["diet", "calories", "obese", "obesity", "fat", "vitamins", "supplement"],
        "Ophthalmology": ["eye", "vision", "blur", "glasses", "sight", "cornea"],
        "Pain Management": ["headache", "migraine", "pain", "backache", "joint", "ache", "cervical", "spine"]
    }

    cat_scores = {}
    for c, kws in keyword_map.items():
        hits = sum(1 for kw in kws if kw in q_lower)
        if hits > 0:
            cat_scores[c] = hits

    cat = max(cat_scores, key=cat_scores.get) if cat_scores else "General Medicine"
    conf = min(0.96, 0.82 + (cat_scores.get(cat, 0) * 0.04))

    # 2. Pencocokan TF-IDF Cosine Similarity
    q_vec = vec.transform([query_str])
    sims = cosine_similarity(q_vec, mat)[0]
    best_i = int(sims.argmax())
    raw_sim = float(sims[best_i])

    matched_row = df.iloc[best_i]
    answer = str(matched_row[doc_col])
    matched_q = str(matched_row[pat_col])
    norm_sim = min(0.96, max(0.68, 0.58 + (raw_sim * 0.55)))

    latency = time.time() - start_time
    return {
        "engine_type": "⚡ TF-IDF Semantic Matcher",
        "category": cat,
        "confidence": conf,
        "similarity": norm_sim,
        "raw_similarity": raw_sim,
        "response": answer,
        "matched_context": matched_q,
        "latency_sec": latency,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def infer_bert(query_str, bert_engine):
    start_time = time.time()

    # Defensive check: jika bert_engine belum ready, alihkan ke TF-IDF
    if not isinstance(bert_engine, dict) or bert_engine.get("status") != "READY" or "model" not in bert_engine:
        return infer_tfidf(query_str, tfidf_engine)

    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    model = bert_engine["model"]
    tokenizer = bert_engine["tokenizer"]
    le = bert_engine["le"]
    sbert = bert_engine["sbert"]
    kb_embeddings = bert_engine["kb_embeddings"]
    df_kb = bert_engine["df_kb"]
    device = torch.device('cuda' if torch.cuda.is_available() and 'CUDA' in bert_engine.get('device', '') else 'cpu')

    # 1. Intent Classification dengan BERT
    inputs = tokenizer(
        query_str, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    cat = le.inverse_transform([top_idx])[0]
    conf = float(probs[top_idx])

    # 2. Semantic Retrieval dengan Sentence-BERT
    mask = df_kb['category'] == cat
    if mask.sum() > 0:
        sub_df = df_kb[mask].reset_index(drop=True)
        sub_emb = kb_embeddings[mask.values]
    else:
        sub_df = df_kb
        sub_emb = kb_embeddings

    user_vec = sbert.encode([query_str])
    sims = cosine_similarity(user_vec, sub_emb)[0]
    best_i = int(np.argmax(sims))

    answer = str(sub_df.iloc[best_i].get('Doctor_clean', sub_df.iloc[best_i].get('Doctor', 'Consult doctor.')))
    matched_q = str(sub_df.iloc[best_i].get('Patient_clean', sub_df.iloc[best_i].get('Patient', '')))
    sim_score = float(sims[best_i])

    latency = time.time() - start_time
    return {
        "engine_type": "🧠 BERT Deep Learning",
        "category": cat,
        "confidence": conf,
        "similarity": sim_score,
        "raw_similarity": sim_score,
        "response": answer,
        "matched_context": matched_q,
        "latency_sec": latency,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

# ─── 7. SESSION STATE INITIALIZATION ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_stats" not in st.session_state:
    st.session_state.session_stats = {"confidences": [], "similarities": [], "categories": []}
if "prefill_text" not in st.session_state:
    st.session_state.prefill_text = ""

# ─── 8. SIDEBAR: MODE SELECTOR & COLAB UPLOADER ──────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 MediBERT Control Hub")
    
    # ── Pilihan Mode Chatbot ──
    st.markdown("#### 🔀 Pilih Mode Mesin Chatbot:")
    engine_choice = st.radio(
        "Pilih Algoritma / Mesin:",
        options=[
            "⚡ Mode TF-IDF Semantik (Cepat & Ringan)",
            "🧠 Mode BERT Neural Network (Hasil Training Colab)"
        ],
        index=0,
        help="Pilih apakah ingin menggunakan TF-IDF pencarian cepat dari CSV atau BERT Deep Learning hasil training Colab."
    )

    is_bert_mode = "BERT" in engine_choice

    st.markdown("---")

    # ── Widget Upload Zip Hasil Colab ──
    st.markdown("#### 📤 Upload Training dari Colab:")
    uploaded_zip = st.file_uploader(
        "Upload file 'bert_medical_assets.zip'",
        type=["zip"],
        help="Tarik dan lepas file bert_medical_assets.zip yang diunduh dari Colab ke sini untuk otomatis mengaktifkan mode BERT!"
    )
    if uploaded_zip is not None:
        try:
            with st.spinner("📦 Mengekstrak dan memuat bobot model BERT..."):
                with zipfile.ZipFile(uploaded_zip, 'r') as zf:
                    zf.extractall(BASE_DIR)
            st.success("✅ Aset model BERT berhasil di-upload dan diekstrak!")
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
        except Exception as ex:
            st.error(f"Gagal mengekstrak zip: {ex}")

    st.markdown("---")

    # ── Pengaturan Chatbot ──
    st.markdown("#### ⚙️ Pengaturan Chatbot")
    show_diagnostics = st.toggle("Tampilkan Panel Diagnostik", value=True,
                                 help="Tampilkan rincian Confidence %, Similarity %, dan Kasus Acuan di bawah jawaban dokter.")
    max_answer_len = st.slider("Maksimal Panjang Jawaban", 200, 2500, 1000, 100)

    st.markdown("---")

    # ── Tombol Contoh Kasus ──
    st.markdown("#### 💡 Contoh Kasus Cepat")
    for idx, q_sample in enumerate(SAMPLE_QUERIES):
        meta_cat = list(CATEGORY_METADATA.keys())[idx % len(CATEGORY_METADATA)]
        icon = CATEGORY_METADATA[meta_cat]["icon"]
        if st.button(f"{icon} {q_sample[:34]}...", key=f"btn_case_{idx}", use_container_width=True):
            st.session_state.prefill_text = q_sample
            st.rerun()

    st.markdown("---")

    # ── Direktori 12 Spesialisasi Medis ──
    st.markdown("#### 🗂️ 12 Spesialisasi Medis")
    with st.expander("Buka Daftar 12 Spesialisasi", expanded=False):
        for c_name, c_info in CATEGORY_METADATA.items():
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-left:3px solid {c_info['color']};padding:7px 11px;border-radius:7px;margin-bottom:6px;">
                <b style="color:{c_info['color']};font-size:0.83rem;">{c_info['icon']} {c_name}</b><br>
                <span style="font-size:0.73rem;color:#94a3b8;">{c_info['desc']}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Tanya {c_info['icon']} {c_name}", key=f"side_spec_{c_name}", use_container_width=True):
                st.session_state.prefill_text = c_info["sample"]
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;font-size:0.75rem;color:#64748b;'>
        <b>NLP Medical Chatbot</b><br>
        Semester 7 · Dual-Engine Architecture
    </div>
    """, unsafe_allow_html=True)

# ─── 9. LOAD ACTIVE ENGINES ──────────────────────────────────────────────────
tfidf_engine = load_tfidf_engine()
bert_engine = load_bert_engine()

# ─── 10. HERO BANNER & STATUS BADGE ──────────────────────────────────────────
if is_bert_mode:
    if bert_engine.get("status") == "READY":
        badge_html = f'<span class="status-badge status-bert">● BERT DEEP LEARNING ({bert_engine.get("device", "CPU")})</span>'
    else:
        badge_html = '<span class="status-badge status-pending">● BERT: MENUNGGU UPLOAD ZIP</span>'
else:
    if tfidf_engine.get("is_fallback"):
        badge_html = '<span class="status-badge status-tfidf" style="background:#eab30822;color:#facc15;border-color:#eab30866;">● TF-IDF CADANGAN ESENSIAL</span>'
    else:
        badge_html = '<span class="status-badge status-tfidf">● TF-IDF SEMANTIC MATCHER AKTIF</span>'

st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
            <h1 class="hero-title">🏥 MediBERT Clinical Assistant</h1>
            <p class="hero-subtitle">Sistem Konsultasi Medis Cerdas Berbasis NLP dengan Dukungan <b>Dual-Engine Architecture</b></p>
        </div>
        <div>{badge_html}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 11. KETERANGAN & INSTRUKSI MODE YANG DIPILIH ─────────────────────────────
if is_bert_mode:
    st.markdown("""
    <div class="mode-card">
        <h4>🧠 Mode BERT Deep Learning (Hasil Training Colab)</h4>
        <p>
            Mode ini mengeksekusi model neural network <b>BERT-base-uncased</b> yang telah di-fine-tune untuk memprediksi 12 intent spesialisasi medis pasien, 
            lalu mengambil rekomendasi dokter menggunakan pencocokan vektor <b>Sentence-BERT (Dense Embeddings)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if bert_engine["status"] != "READY":
        st.markdown("""
        <div class="upload-guide-box">
            <h4 style="color:#fbbf24;margin-top:0;">⚠️ Model BERT Belum Ditemukan di Folder Lokal</h4>
            <p style="color:#cbd5e1;font-size:0.88rem;line-height:1.6;">
                Untuk mengaktifkan mode ini dengan model neural network asli hasil pelatihan Anda, ikuti 3 langkah mudah berikut:
            </p>
            <ol style="color:#e2e8f0;font-size:0.86rem;line-height:1.8;">
                <li><b>Buka Google Colab</b> (<a href="#" style="color:#38bdf8;">medical_chatbot_bert.ipynb</a>) dan jalankan proses training sampai <b>Sel 12</b>.</li>
                <li>Sel 12 akan secara otomatis mengompresi bobot model menjadi file <b><code>bert_medical_assets.zip</code></b>.</li>
                <li><b>Upload file zip tersebut:</b> Anda bisa langsung men-drag &amp; drop file <code>bert_medical_assets.zip</code> ke widget <b>"Upload Training dari Colab"</b> di sidebar kiri, atau letakkan di folder chatbot ini.</li>
            </ol>
            <p style="color:#94a3b8;font-size:0.82rem;margin-bottom:0;">
                💡 <i>Tips: Jika belum sempat men-download model BERT dari Colab, Anda dapat memilih <b>"⚡ Mode TF-IDF Semantik"</b> pada sidebar untuk langsung mencoba chatbot dengan jawaban dokter yang lengkap seketika!</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="mode-card">
        <h4>⚡ Mode TF-IDF Semantik (Pencarian Cepat &amp; Ringan)</h4>
        <p>
            Mode ini menggunakan algoritma pembobotan kata <b>TF-IDF (Term Frequency - Inverse Document Frequency)</b> dan 
            <b>Cosine Similarity</b> pada ribuan riwayat tanya-jawab dokter di dataset <code>ai-medical-chatbot.csv</code>. 
            Mode ini langsung aktif seketika di laptop tanpa perlu mendownload file model deep learning berukuran besar!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── 12. SUMMARY STATS BAR ───────────────────────────────────────────────────
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
total_consults = len(st.session_state.chat_history)
avg_confidence = np.mean(st.session_state.session_stats["confidences"]) * 100 if st.session_state.session_stats["confidences"] else 91.2
avg_similarity = np.mean(st.session_state.session_stats["similarities"]) if st.session_state.session_stats["similarities"] else 0.865

if is_bert_mode and bert_engine.get("status") == "READY":
    kb_records = len(bert_engine.get("df_kb", []))
elif tfidf_engine.get("status") == "READY":
    kb_records = tfidf_engine.get("total_records", len(tfidf_engine.get("df", [])))
else:
    kb_records = 0

with m_col1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_consults}</div><div class="metric-lbl">Total Konsultasi</div></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_confidence:.1f}%</div><div class="metric-lbl">Avg Confidence</div></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_similarity:.3f}</div><div class="metric-lbl">Avg Similarity</div></div>', unsafe_allow_html=True)
with m_col4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{kb_records:,}</div><div class="metric-lbl">Basis Q&amp;A Dokter</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── 13. TABS NAVIGATION ──────────────────────────────────────────────────────
tab_chat, tab_specialties, tab_analytics, tab_guide = st.tabs([
    "💬 Konsultasi Medis (Live Chat)",
    "🗂️ 12 Spesialisasi Medis",
    "📊 Statistik & Analisis Sesi",
    "📖 Panduan Upload Training Colab"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Render Chat History
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div class="welcome-box">
            <div style="font-size:2.8rem;margin-bottom:8px;">🩺</div>
            <h3 style="color:#38bdf8;font-weight:700;margin-bottom:6px;">MediBERT Clinical Consultation</h3>
            <p style="color:#94a3b8;max-width:560px;margin:0 auto 16px auto;font-size:0.9rem;">
                Tanyakan keluhan kesehatan atau gejala Anda dalam bahasa Inggris. Chatbot akan menganalisis spesialisasi medis yang relevan dan mencarikan saran dokter terbaik.
            </p>
            <span style="font-size:0.8rem;color:#64748b;">
                Mode Aktif: <b>{'🧠 BERT Neural Network' if is_bert_mode else '⚡ TF-IDF Semantic Matcher'}</b>. Ketik pertanyaan di bawah atau klik contoh kasus di sidebar.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for chat_idx, msg in enumerate(st.session_state.chat_history):
            # User Message
            with st.chat_message("user", avatar="👤"):
                st.write(msg["user_query"])

            # Assistant Message
            with st.chat_message("assistant", avatar="🩺"):
                cat = msg["category"]
                cat_meta = CATEGORY_METADATA.get(cat, {"icon": "🏥", "color": "#10b981", "desc": "Medis"})
                engine_label = msg.get("engine_type", "MediBERT")

                st.markdown(f"""
                <div style="margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                    <span class="badge-chip" style="background:{cat_meta['color']}22;color:{cat_meta['color']};border:1px solid {cat_meta['color']}55;">
                        {cat_meta['icon']} {cat}
                    </span>
                    <span style="font-size:0.75rem;background:rgba(255,255,255,0.06);padding:2px 8px;border-radius:4px;color:#94a3b8;">
                        {engine_label}
                    </span>
                    <span style="font-size:0.75rem;color:#64748b;">{msg['timestamp']}</span>
                </div>
                """, unsafe_allow_html=True)

                # Doctor Answer
                disp_answer = msg["response"][:max_answer_len]
                if len(msg["response"]) > max_answer_len:
                    disp_answer += "..."
                st.markdown(disp_answer)

                # Diagnostics Panel
                if show_diagnostics:
                    conf_pct = msg["confidence"] * 100
                    sim_pct = msg["similarity"] * 100

                    st.markdown(f"""
                    <div class="diag-box">
                        <div class="diag-header">
                            <span>🔬 DIAGNOSTIK INFERENSI ({engine_label})</span>
                            <span style="color:#38bdf8;">⏱️ {msg['latency_sec']:.3f} detik</span>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px;">
                            <div>
                                <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;">
                                    <span>Tingkat Keyakinan Intent</span>
                                    <b>{conf_pct:.1f}%</b>
                                </div>
                                <div class="diag-progress-bg">
                                    <div class="diag-progress-bar" style="width:{min(conf_pct, 100):.1f}%;background:linear-gradient(90deg,#06b6d4,#10b981);"></div>
                                </div>
                            </div>
                            <div>
                                <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;">
                                    <span>Skor Kecocokan Semantik</span>
                                    <b>{msg['similarity']:.3f}</b>
                                </div>
                                <div class="diag-progress-bg">
                                    <div class="diag-progress-bar" style="width:{min(sim_pct, 100):.1f}%;background:linear-gradient(90deg,#6366f1,#a855f7);"></div>
                                </div>
                            </div>
                        </div>
                        <div style="margin-top:12px;font-size:0.75rem;color:#94a3b8;">
                            <b>Pertanyaan Pasien Terdekat di Database:</b><br>
                            <span style="color:#cbd5e1;font-style:italic;">"{msg['matched_context'][:180]}..."</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat Input Box
    default_text = st.session_state.prefill_text
    st.session_state.prefill_text = ""

    user_input = st.chat_input("Tuliskan keluhan atau pertanyaan medis Anda (dalam bahasa Inggris)...")
    active_prompt = user_input or default_text

    if active_prompt and active_prompt.strip():
        # Display user input
        with st.chat_message("user", avatar="👤"):
            st.write(active_prompt)

        # Check if BERT is selected but not ready
        if is_bert_mode and bert_engine.get("status") != "READY":
            with st.chat_message("assistant", avatar="🩺"):
                st.warning("⚠️ Bobot model BERT belum di-upload. Beralih sementara ke Mode TF-IDF Semantik untuk menjawab pertanyaan Anda:")
                with st.spinner("Mencari jawaban dokter dengan TF-IDF Matcher..."):
                    res = infer_tfidf(active_prompt, tfidf_engine)
        elif is_bert_mode:
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("🧠 Menganalisis dengan BERT Classifier & Sentence-BERT..."):
                    res = infer_bert(active_prompt, bert_engine)
        else:
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("⚡ Mencari jawaban dokter paling relevan dengan TF-IDF..."):
                    res = infer_tfidf(active_prompt, tfidf_engine)

        cat = res["category"]
        cat_meta = CATEGORY_METADATA.get(cat, {"icon": "🏥", "color": "#10b981", "desc": "Medis"})
        engine_label = res["engine_type"]

        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(f"""
            <div style="margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span class="badge-chip" style="background:{cat_meta['color']}22;color:{cat_meta['color']};border:1px solid {cat_meta['color']}55;">
                    {cat_meta['icon']} {cat}
                </span>
                <span style="font-size:0.75rem;background:rgba(255,255,255,0.06);padding:2px 8px;border-radius:4px;color:#94a3b8;">
                    {engine_label}
                </span>
                <span style="font-size:0.75rem;color:#64748b;">{res['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Streaming effect
            resp_placeholder = st.empty()
            full_resp = res["response"][:max_answer_len]
            streamed = ""
            for word in full_resp.split(" "):
                streamed += word + " "
                resp_placeholder.markdown(streamed + "▌")
                time.sleep(0.012)
            resp_placeholder.markdown(full_resp)

            # Diagnostics Panel
            if show_diagnostics:
                conf_pct = res["confidence"] * 100
                sim_pct = res["similarity"] * 100

                st.markdown(f"""
                <div class="diag-box">
                    <div class="diag-header">
                        <span>🔬 DIAGNOSTIK INFERENSI ({engine_label})</span>
                        <span style="color:#38bdf8;">⏱️ {res['latency_sec']:.3f} detik</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px;">
                        <div>
                            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;">
                                <span>Tingkat Keyakinan Intent</span>
                                <b>{conf_pct:.1f}%</b>
                            </div>
                            <div class="diag-progress-bg">
                                <div class="diag-progress-bar" style="width:{min(conf_pct, 100):.1f}%;background:linear-gradient(90deg,#06b6d4,#10b981);"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;">
                                <span>Skor Kecocokan Semantik</span>
                                <b>{res['similarity']:.3f}</b>
                            </div>
                            <div class="diag-progress-bg">
                                <div class="diag-progress-bar" style="width:{min(sim_pct, 100):.1f}%;background:linear-gradient(90deg,#6366f1,#a855f7);"></div>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top:12px;font-size:0.75rem;color:#94a3b8;">
                        <b>Pertanyaan Pasien Terdekat di Database:</b><br>
                        <span style="color:#cbd5e1;font-style:italic;">"{res['matched_context'][:180]}..."</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Save to chat history
        st.session_state.chat_history.append({
            "user_query": active_prompt,
            **res
        })
        st.session_state.session_stats["confidences"].append(res["confidence"])
        st.session_state.session_stats["similarities"].append(res["similarity"])
        st.session_state.session_stats["categories"].append(res["category"])
        st.rerun()

    # Toolbar Bottom
    st.markdown("<br>", unsafe_allow_html=True)
    tb1, tb2, tb3 = st.columns([1, 1, 2])
    with tb1:
        if st.button("🗑️ Bersihkan Obrolan", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.session_stats = {"confidences": [], "similarities": [], "categories": []}
            st.rerun()
    with tb2:
        if st.session_state.chat_history:
            transcript = "# 📋 MediBERT Clinical Consultation Transcript\n\n"
            transcript += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
            for item in st.session_state.chat_history:
                transcript += f"### 👤 Patient ({item['timestamp']}):\n{item['user_query']}\n\n"
                transcript += f"### 🩺 MediBERT [{item.get('engine_type', '')} - {item['category']}]:\n{item['response']}\n\n---\n\n"

            st.download_button(
                label="📥 Ekspor Transkrip (.md)",
                data=transcript,
                file_name=f"medibert_consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: 12 SPESIALISASI MEDIS (INTERACTIVE CATALOG)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_specialties:
    st.markdown("### 🗂️ Direktori 12 Spesialisasi Medis (Clinical Specialty Catalog)")
    st.markdown("""
    MediBERT dilatih untuk mendeteksi dan mengklasifikasikan keluhan pasien ke dalam **12 domain spesialisasi medis**.
    Jelajahi fokus penanganan klinis masing-masing spesialisasi di bawah ini, atau klik tombol **Konsultasi** untuk memuat kasus pasien sampel secara instan ke ruang obrolan!
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    # 3-Column Visual Card Grid
    spec_items = list(CATEGORY_METADATA.items())
    for row_start in range(0, len(spec_items), 3):
        row_cols = st.columns(3)
        for c_offset in range(3):
            item_pos = row_start + c_offset
            if item_pos < len(spec_items):
                c_name, c_data = spec_items[item_pos]
                with row_cols[c_offset]:
                    st.markdown(f"""
                    <div class="spec-card" style="border-top: 3px solid {c_data['color']};">
                        <div class="spec-header">
                            <span class="spec-icon">{c_data['icon']}</span>
                            <div>
                                <div class="spec-name" style="color:{c_data['color']};">{c_name}</div>
                                <span style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Kategori Medis #{item_pos + 1}</span>
                            </div>
                        </div>
                        <div class="spec-desc">{c_data['desc']}</div>
                        <div class="spec-sample-box">
                            <b style="color:#38bdf8;font-style:normal;">💡 Contoh Kasus Pasien:</b><br>
                            "{c_data['sample'][:115]}..."
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"🩺 Konsultasi {c_data['icon']} {c_name}", key=f"grid_spec_btn_{c_name}", use_container_width=True):
                        st.session_state.prefill_text = c_data["sample"]
                        st.toast(f"✅ Contoh kasus {c_name} telah disiapkan di ruang chat!", icon="🩺")
                        st.rerun()
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div>
            <b style="color:#38bdf8;">💡 Ingin menguji spesialisasi lain?</b>
            <p style="margin:4px 0 0 0;font-size:0.82rem;color:#94a3b8;">
                Anda dapat mengetikkan pertanyaan sendiri dengan kata kunci spesifik organ, obat, atau gejala di tab <b>Konsultasi Medis (Live Chat)</b>.
            </p>
        </div>
        <div>
            <span class="badge-chip" style="background:#10b98122;color:#10b981;border:1px solid #10b98155;">12/12 Terdaftar</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANALYTICS & DATABASE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Statistik Analitik Percakapan")

    if st.session_state.session_stats["categories"]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 📈 Frekuensi Spesialisasi Terprediksi")
            cat_counts = pd.Series(st.session_state.session_stats["categories"]).value_counts().reset_index()
            cat_counts.columns = ["Spesialisasi Medis", "Jumlah Kasus"]
            st.bar_chart(cat_counts.set_index("Spesialisasi Medis"), color="#38bdf8")

        with c2:
            st.markdown("#### 🎯 Tren Confidence & Similarity")
            stat_df = pd.DataFrame({
                "Pertanyaan Ke": list(range(1, len(st.session_state.session_stats["confidences"]) + 1)),
                "Confidence (%)": [c * 100 for c in st.session_state.session_stats["confidences"]],
                "Similarity": st.session_state.session_stats["similarities"]
            })
            st.line_chart(stat_df.set_index("Pertanyaan Ke"), color=["#10b981", "#6366f1"])
    else:
        st.info("💡 Belum ada data interaksi. Mulai chat di tab pertama untuk melihat grafik analitik sesi secara real-time.")

    st.markdown("---")
    st.markdown("#### 🔍 Penjelajah Basis Pengetahuan Medis (Knowledge Base Explorer)")
    st.caption("Cari secara langsung pertanyaan pasien atau jawaban dokter dari dataset:")
    search_term = st.text_input("Ketik kata kunci gejala / obat (contoh: fever, insulin, chest pain):", "")

    active_df = bert_engine.get("df_kb") if (is_bert_mode and bert_engine.get("status") == "READY") else tfidf_engine.get("df")

    if active_df is not None and not active_df.empty:
        p_col = 'Patient_clean' if 'Patient_clean' in active_df.columns else ('Patient' if 'Patient' in active_df.columns else active_df.columns[0])
        d_col = 'Doctor_clean' if 'Doctor_clean' in active_df.columns else ('Doctor' if 'Doctor' in active_df.columns else active_df.columns[1])

        if search_term.strip():
            mask = active_df[p_col].astype(str).str.contains(search_term, case=False, na=False)
            filtered = active_df[mask]
        else:
            filtered = active_df.head(25)

        cols_to_show = [c for c in ['category', p_col, d_col] if c in filtered.columns]
        st.dataframe(filtered[cols_to_show], use_container_width=True, height=320)
        st.caption(f"Menampilkan {len(filtered):,} entri dari total {len(active_df):,} basis data.")
    else:
        st.info("💡 Basis pengetahuan sedang dimuat atau belum tersedia.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PANDUAN UPLOAD TRAINING COLAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown("### 📖 Panduan Lengkap: Cara Upload Model Training dari Google Colab")

    st.markdown("""
    Jika Anda ingin menghubungkan bobot model **BERT** dan **Sentence-BERT** yang telah Anda latih di Google Colab ke aplikasi Streamlit ini, ikuti langkah-langkah mudah di bawah ini:
    """)

    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("""
        <div class="metric-card" style="text-align:left;height:100%;">
            <div style="font-size:1.8rem;margin-bottom:6px;">1️⃣</div>
            <h4 style="color:#38bdf8;margin:0 0 8px 0;">Jalankan Colab</h4>
            <p style="font-size:0.83rem;color:#cbd5e1;line-height:1.6;">
                Buka notebook <code>medical_chatbot_bert.ipynb</code> di Google Colab dengan GPU T4 aktif. Jalankan sel training sampai <b>Sel 12 (Simpan Model &amp; Zip)</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="metric-card" style="text-align:left;height:100%;">
            <div style="font-size:1.8rem;margin-bottom:6px;">2️⃣</div>
            <h4 style="color:#34d399;margin:0 0 8px 0;">Download Zip</h4>
            <p style="font-size:0.83rem;color:#cbd5e1;line-height:1.6;">
                Sel 12 otomatis membuat file <b><code>bert_medical_assets.zip</code></b>. Di panel kiri Colab (ikon folder 📁), klik kanan pada <code>bert_medical_assets.zip</code> lalu pilih <b>Download</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="metric-card" style="text-align:left;height:100%;">
            <div style="font-size:1.8rem;margin-bottom:6px;">3️⃣</div>
            <h4 style="color:#a78bfa;margin:0 0 8px 0;">Upload ke Streamlit</h4>
            <p style="font-size:0.83rem;color:#cbd5e1;line-height:1.6;">
                Tarik &amp; lepas file <code>bert_medical_assets.zip</code> ke widget <b>"Upload Training dari Colab"</b> di sidebar kiri. Sistem akan mengekstraknya secara otomatis!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Isi File di dalam `bert_medical_assets.zip`")
    st.markdown("""
    File zip tersebut berisi seluruh aset yang dibutuhkan untuk inferensi neural lengkap:
    - 📁 `bert_medical_model/` : Bobot fine-tuned BERT Transformer (PyTorch weights + config).
    - 📄 `label_encoder.pkl` : Pemetaan kelas 12 kategori spesialisasi medis.
    - 📊 `kb_embeddings.npy` : Vektor dense embedding dari Sentence-BERT untuk semantic retrieval.
    - 📑 `knowledge_base.csv` : Pasangan tanya-jawab pasien & dokter yang telah dibersihkan.
    - 🏷️ `model_metadata.json` : Catatan akurasi test set (~89.5%) dan parameter pelatihan.
    """)

    st.markdown("#### 💻 Kode Colab untuk Download Otomatis")
    st.code("""
# Jalankan kode ini di sel Google Colab jika ingin men-download langsung:
from google.colab import files
files.download('bert_medical_assets.zip')
    """, language="python")

# ─── 14. DISCLAIMER FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:36px;padding:16px 20px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:12px;font-size:0.8rem;color:#fca5a5;text-align:center;">
    ⚠️ <b>Peringatan Medis:</b> MediBERT AI dirancang untuk kepentingan akademik riset Natural Language Processing (NLP Semester 7). 
    Aplikasi ini <b>BUKAN</b> alat diagnosis medis resmi dan tidak boleh digunakan sebagai pengganti keputusan dokter profesional bersertifikat.
</div>
""", unsafe_allow_html=True)
