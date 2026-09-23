# 📘 PANDUAN LENGKAP PENGEMBANGAN & PENGGUNAAN MEDIBERT

> **Proyek Mata Kuliah:** Natural Language Processing (NLP) — Semester 7  
> **Judul:** MediBERT: Dual-Engine NLP Medical Consultation Chatbot  
> **Dataset:** AI Medical Chatbot (256.000+ data tanya-jawab pasien & dokter)  
> **Arsitektur Utama:** Dual-Engine (TF-IDF Semantic Retrieval & Fine-Tuned BERT Transformer)

---

## 📑 Daftar Isi
1. [Ringkasan Proyek & Deliverables](#1-ringkasan-proyek--deliverables)
2. [Struktur File Repositori](#2-struktur-file-repositori)
3. [Panduan Eksekusi di Google Colab](#3-panduan-eksekusi-di-google-colab)
4. [Panduan Eksekusi Lokal (Streamlit)](#4-panduan-eksekusi-lokal-streamlit)
5. [Mekanisme Arsitektur Dual-Engine](#5-mekanisme-arsitektur-dual-engine)
6. [Katalog 12 Spesialisasi Medis](#6-katalog-12-spesialisasi-medis)
7. [Panduan Push ke GitHub (Mengatasi File Besar)](#7-panduan-push-ke-github-mengatasi-file-besar)
8. [Troubleshooting & Solusi Masalah Umum](#8-troubleshooting--solusi-masalah-umum)

---

## 1. Ringkasan Proyek & Deliverables

| No | Komponen | File Terkait | Keterangan |
|:---:|:---|:---|:---|
| 1 | **Notebook Google Colab** | `medical_chatbot_bert.ipynb` | Pipeline preprocessing dataset, fine-tuning BERT intent classification, pembuatan embeddings Sentence-BERT, dan antarmuka web Gradio cloud. |
| 2 | **Aplikasi Web Lokal** | `app.py` | Dashboard visual interaktif dengan antarmuka modern (Streamlit) berfitur live chat, katalog 12 spesialisasi, dan analitik sesi. |
| 3 | **Dataset Medis** | `ai-medical-chatbot.csv` | Dataset berisi interaksi nyata tanya-jawab medis pasien dan dokter terverifikasi (~254 MB). |
| 4 | **Dokumentasi Publik** | `README.md` | Halaman perkenalan dan portofolio profesional untuk repositori GitHub. |
| 5 | **Panduan Teknis** | `PANDUAN.md` | Petunjuk teknis komprehensif bagi pengembang dan penguji sistem. |
| 6 | **Daftar Dependensi** | `requirements.txt` | Modul dan pustaka Python yang diperlukan untuk instalasi. |

---

## 2. Struktur File Repositori

```text
chatbot/
├── .gitignore                      # Mengabaikan dataset besar & file environment lokal
├── requirements.txt                # Dependensi paket Python yang dibutuhkan
├── README.md                       # Dokumentasi publik repositori GitHub
├── PANDUAN.md                      # Panduan teknis lengkap (file ini)
├── knowledge_base.csv              # Basis data 3.900+ Q&A dokter-pasien siap pakai (TF-IDF out-of-the-box)
├── app.py                          # Aplikasi Web Streamlit (Dual-Engine Dashboard)
├── medical_chatbot_bert.ipynb      # Notebook Colab (Training BERT + UI Gradio)
│
└── [Aset Opsional Model BERT]:
    ├── bert_medical_model/         # Folder weights PyTorch BERT Transformer
    ├── label_encoder.pkl           # Encoder 12 kelas spesialisasi medis
    ├── kb_embeddings.npy           # Vektor dense embeddings Sentence-BERT
    └── model_metadata.json         # Laporan evaluasi & akurasi pengujian model
```

---

## 3. Panduan Eksekusi di Google Colab

Jika Anda ingin melatih model neural network transformer dari awal:

1. Kunjungi [Google Colab](https://colab.research.google.com/).
2. Buka tab **Upload** dan pilih file `medical_chatbot_bert.ipynb`.
3. Aktifkan akselerasi GPU:
   - Pilih menu **Runtime** → **Change runtime type**.
   - Pada opsi *Hardware accelerator*, pilih **T4 GPU** lalu klik **Save**.
4. Unggah dataset `ai-medical-chatbot.csv` ke panel file Colab (ikon folder di sebelah kiri).
5. Jalankan sel berurutan:
   - **Sel 1 - 4**: Instalasi pustaka (`transformers`, `sentence-transformers`, `gradio`) dan pembersihan data.
   - **Sel 5 - 9**: Training & evaluasi model klasifikasi BERT (`bert-base-uncased`) untuk 12 kelas medis.
   - **Sel 10 - 11**: Pembangkitan vektor embedding Sentence-BERT (`all-MiniLM-L6-v2`) untuk semantic retrieval.
   - **Sel 12**: Pengarsipan otomatis seluruh model ke dalam file **`bert_medical_assets.zip`**.
   - **Sel 13**: Menjalankan antarmuka Web Gradio dengan public URL (`.gradio.live`).
6. **Unduh Aset Model:**
   - Pada panel kiri file Colab, klik kanan pada `bert_medical_assets.zip` dan pilih **Download**.

---

## 4. Panduan Eksekusi Lokal (Streamlit)

Aplikasi `app.py` dirancang agar dapat dijalankan langsung di komputer Anda:

### Langkah-langkah:
1. **Buka Terminal / PowerShell** di direktori proyek ini:
   ```powershell
   cd "c:\Users\IFHAL FAIZI\Downloads\KULIAH\SEMESTER 7\NLP\chatbot"
   ```

2. **Pastikan Dependensi Terpasang:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Jalankan Server Streamlit:**
   ```powershell
   python -m streamlit run app.py
   ```

4. **Akses Dashboard:**
   Aplikasi akan otomatis terbuka pada peramban web di:
   `http://localhost:8501`

---

## 5. Mekanisme Arsitektur Dual-Engine

Aplikasi `app.py` dilengkapi sistem pengalihan engine otomatis yang dapat dipilih melalui sidebar:

### 1. Mode ⚡ TF-IDF Semantic Matcher (Default / Cepat)
* **Kelebihan**: Berjalan instan tanpa memerlukan kartu grafis (GPU) atau bobot model ratusan megabyte.
* **Cara Kerja**: Memanfaatkan representasi n-gram teks dan *Cosine Similarity* pada lebih dari 256.000 pasang interaksi medis.
* **Kualitas Jawaban**: Mengutamakan jawaban dokter berkualitas tinggi (>120 karakter) yang paling relevan dengan keluhan pengguna.

### 2. Mode 🧠 BERT Deep Learning (Neural Network Penuh)
* **Kelebihan**: Akurasi klasifikasi intent tertinggi berbasis deep learning transformer (~89.5% accuracy) serta dense embeddings Sentence-BERT.
* **Cara Menghubungkan**:
  1. Pilih opsi **"🧠 Model BERT Hasil Training Colab"** pada radio button di sidebar kiri.
  2. Masukkan file `bert_medical_assets.zip` hasil download dari Google Colab ke kotak upload.
  3. Aplikasi akan otomatis mengekstrak file, memuat bobot PyTorch ke memori/CUDA, dan mengaktifkan inferensi transformer penuh.

---

## 6. Katalog 12 Spesialisasi Medis

Sistem MediBERT mengklasifikasikan pertanyaan pasien ke dalam 12 domain klinis:

1. ❤️ **Cardiovascular**: Penyakit jantung koroner, hipertensi, aritmia, nyeri dada, sirkulasi darah.
2. 🦷 **Dental**: Kesehatan gigi dan gusi, infeksi rongga mulut, karies gigi, impaksi gigi bungsu.
3. 🩹 **Dermatology**: Penyakit kulit, jerawat kistik kronis, eksim, psoriasis, ruam alergi, rambut dan kuku.
4. 🔬 **Endocrine**: Diabetes melitus, kelainan tiroid, gangguan metabolisme glukosa dan hormonal.
5. 🫃 **Gastroenterology**: Gangguan lambung, GERD, tukak lambung, sindrom iritasi usus, pencernaan.
6. 🏥 **General Medicine**: Keluhan umum multi-sistemik, demam tanpa fokus, lemas kronis, pusing berputar.
7. 🦠 **Infectious Disease**: Infeksi virus dan bakteri, HIV/AIDS, demam tifoid, malaria, tuberkulosis.
8. 🧠 **Mental Health**: Gangguan kecemasan umum (*anxiety*), depresi, insomnia berat, serangan panik.
9. 🫘 **Nephrology**: Infeksi saluran kemih (ISK), gangguan fungsi ginjal, hematuria, batu saluran kemih.
10. 🥗 **Nutrition/Obesity**: Pengaturan pola makan klinis, manajemen obesitas, defisit kalori, malnutrisi.
11. 👁️ **Ophthalmology**: Infeksi kornea dan konjungtiva, penglihatan buram mendadak, mata kering/berpasir.
12. 💊 **Pain Management**: Nyeri kronis muskuloskeletal, migrain berat, vertigo, nyeri sendi dan saraf.

---

## 7. Panduan Push ke GitHub (Mengatasi File Besar)

> ⚠️ **PERHATIAN PENTING MENGENAI UKURAN FILE:**  
> GitHub membatasi ukuran maksimal file yang dapat di-push sebesar **100 MB**.  
> File dataset `ai-medical-chatbot.csv` berukuran **~254 MB** sehingga **TIDAK BOLEH** di-push langsung tanpa Git LFS. File ini telah dimasukkan ke dalam `.gitignore`.

### Langkah-langkah Commit & Push ke GitHub:

1. **Inisialisasi Git (jika belum):**
   ```bash
   git init
   ```

2. **Periksa Status File:**
   ```bash
   git status
   ```
   *Pastikan file `ai-medical-chatbot.csv` dan folder `.venv/` tidak ikut terdeteksi (sudah diabaikan oleh `.gitignore`).*

3. **Tambahkan File ke Staging:**
   ```bash
   git add .
   ```

4. **Lakukan Commit:**
   ```bash
   git commit -m "feat: complete medibert dual-engine chatbot with 12 specialties UI"
   ```

5. **Hubungkan ke Repositori Remote GitHub Anda:**
   ```bash
   git branch -M main
   git remote add origin https://github.com/USERNAME-ANDA/medibert-clinical-chatbot.git
   ```

6. **Push ke GitHub:**
   ```bash
   git push -u origin main
   ```

---

## 8. Troubleshooting & Solusi Masalah Umum

### 1. Muncul error port sudah digunakan (*Port 8501 is already in use*)
* **Solusi**: Tutup terminal sebelumnya atau tentukan port baru saat menjalankan:
  ```powershell
  python -m streamlit run app.py --server.port 8502
  ```

### 2. Memori RAM penuh saat memuat model BERT
* **Solusi**: Beralihlah ke **Mode TF-IDF** di sidebar. Mode TF-IDF dirancang sangat hemat memori dan tetap memberikan respons cerdas berdasarkan kemiripan semantik kasus dokter.

### 3. File hasil ekstrak `bert_medical_assets.zip` tidak terbaca
* **Solusi**: Pastikan file zip yang diunggah berisi 5 file utama:
  `bert_medical_model/`, `label_encoder.pkl`, `kb_embeddings.npy`, `knowledge_base.csv`, dan `model_metadata.json`.
