# Deteksi Kanker Paru (GSE4115) dengan Seleksi Fitur Lion Optimization Algorithm (LOA) & Ensemble Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Optimization](https://img.shields.io/badge/Optimization-Niapy-green)

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi kode untuk **Tugas Akhir (Final Thesis)** yang berfokus pada klasifikasi data ekspresi gen (Dataset **GSE4115**) untuk memprediksi kanker paru-paru pada perokok.

Tantangan utama pada data *microarray* adalah dimensi yang sangat tinggi (*High Dimensionality*). Oleh karena itu, proyek ini menerapkan pendekatan **Hybrid Feature Selection**:
1.  **Filter Method:** Variance Threshold.
2.  **Wrapper Method:** Lion Optimization Algorithm (LOA).

Model klasifikasi akhir menggunakan metode **Ensemble Voting** yang menggabungkan Random Forest, AdaBoost, dan XGBoost.

## 📂 Struktur & Alur Kerja (Pipeline)
Proyek ini dibagi menjadi 3 tahap utama yang harus dijalankan secara berurutan:

### 1. `Preprocessing_&_Split_Data.ipynb`
Tahap awal pembersihan dan reduksi dimensi awal.
- **Input:** Dataset mentah `GSE4115_data_set.csv`.
- **Proses:**
  - Transpose data (jika format *probes x samples*).
  - Label Encoding (Diagnosed vs Not Diagnosed).
  - Imputasi nilai kosong (NaN).
  - **Variance Threshold:** Reduksi fitur dari ~22.000 menjadi ~1.000 fitur.
  - Standard Scaling.
- **Output:** File `train.csv` dan `test.csv`.

### 2. `Seleksi_Fitur_LOA.ipynb`
Tahap seleksi fitur tingkat lanjut menggunakan algoritma metaheuristik.
- **Input:** `train.csv` dan `test.csv` (dari tahap 1).
- **Proses:**
  - Implementasi **Lion Optimization Algorithm (LOA)** menggunakan library `niapy`.
  - Pencarian subset fitur optimal menggunakan 3 evaluator: Random Forest, AdaBoost, dan XGBoost.
  - Cross-Validation pada fungsi fitness untuk mencegah overfitting.
- **Output:** File CSV berisi indeks fitur terpilih (`X_train_rf.csv`, `X_train_ab.csv`, dll).

### 3. `Ensemble_dan_Validasi_dan_Visualisasi FINAL.ipynb`
Tahap pemodelan, tuning, dan evaluasi akhir.
- **Input:** Dataset original dan dataset hasil seleksi fitur LOA.
- **Proses:**
  - Pelatihan model Baseline (tanpa tuning).
  - Hyperparameter Tuning menggunakan `GridSearchCV`.
  - Pembangunan model **Voting Classifier (Hard Voting)**.
- **Visualisasi:** Confusion Matrix & Perbandingan Metrik (Accuracy, Precision, Recall, F1).

## 🛠️ Setup Environment

### Prasyarat
Pastikan Anda telah menginstal Python. Library utama yang digunakan adalah:
- `numpy`, `pandas` (Manipulasi Data)
- `scikit-learn` (Modeling)
- `xgboost` (Modeling)
- `niapy` (Algoritma Optimasi LOA)
- `matplotlib`, `seaborn` (Visualisasi)

### Instalasi Dependensi
Anda dapat menginstal seluruh dependensi menggunakan pip:

```bash
pip install numpy pandas scikit-learn xgboost niapy matplotlib seaborn
