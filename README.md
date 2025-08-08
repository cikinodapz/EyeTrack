# Dokumentasi Proyek

## Judul

**XGBoost Ensemble dengan Circular–Entropy Features untuk Klasifikasi Gaya Belajar dari Eye‑Tracking**

## Ringkasan Eksekutif

- **Tujuan:** Mengklasifikasikan gaya belajar siswa menjadi **Sequential (1)** vs **Random (2)** menggunakan sinyal eye‑tracking.
- **Label:** `1 = Sequential (urut)`, `2 = Random (acak)`.
- **Distribusi data:** Seimbang (contoh yang diberikan: `label=1` = 125.160 baris, `label=2` = 125.160 baris).
- **Model v1 (single XGBoost + auto-scaler):**
  - *Validation:* Acc **0.7525**, AUC **0.8318**, threshold **0.498** (scaler **robust**).
  - *Test:* Acc **0.7514**, AUC **0.8337**.
- **Model v2 (ensemble XGBoost, soft‑voting lintas scaler & seed):**
  - *Validation:* Acc **0.8181**, AUC **0.9013**, threshold **0.510**.
  - *Test:* Acc **0.8216**, AUC **0.9068**.
- **Kesimpulan:** Ensembling meningkatkan akurasi ±**7 poin** dan AUC ±**0.07–0.08** dibanding v1.

---

## 1) Latar Belakang & Formulasi Masalah

Dosen menyatakan **target** yang diprediksi adalah dua tipe gaya belajar: **Sequential** (urut) dan **Random** (acak). Notebook mengubah problem ke **biner**:

- `y = 1` untuk **Random (label=2)**,
- `y = 0` untuk **Sequential (label=1)**.

Tujuan evaluasi: mencapai akurasi tinggi **tanpa overfitting**, serta skor AUC yang baik untuk kestabilan threshold.

---

## 2) Data & Sumber

- **File input:** `truncated_dataset-seqglo.csv`.
- **Kolom penting awal:** `nama`, `time`, `gazeX`, `gazeY`, `label`.
- **Sorting:** Data diurutkan per subjek `nama` dan waktu `time` untuk menjaga temporal order.
- **Split:** `train_test_split(stratify=y, test_size=0.2, random_state=42)` → test ±20%.

> Catatan: karena data time‑series multi‑subjek, pemrosesan delta/rolling di‑reset saat ganti subjek agar tidak “bocor” antar sesi.

---

## 3) Pra‑Pemrosesan

- **Pembersihan NaN/Inf:** Semua fitur numerik dikonversi ke numerik, nilai `inf/NaN` diganti **median** kolom.
- **Skalering:**
  - **v1:** *Auto‑scaler selection* – coba `none`, `standard`, `robust`, `quantile` di validation, pilih terbaik (hasil: **robust**).
  - **v2:** *Ensemble across scalers* – gunakan **robust**, **quantile**, **standard** secara bersamaan, lalu **soft‑voting** (rata‑rata probabilitas).

---

## 4) Rekayasa Fitur (Ringkasan)

Seluruh fitur dihitung per subjek dengan **window rolling** `w=21` serta statistik angular & langkah gerak mata.

### 4.1. Deltas & Circular Stats

- `dx = diff(gazeX)`, `dy = diff(gazeY)`, `step = sqrt(dx^2 + dy^2)`.
- `angle = atan2(dy, dx)`, `Δangle` di‑wrap ke `[-π, π]`.
- **Circular rolling:** `mean(cos Δangle)`, `mean(sin Δangle)`, **MRL** (mean resultant length), **circular variance** (=1−MRL).

### 4.2. Rolling Statistics (w=21)

- `r_std_x`, `r_std_y`, `r_mean_step`, `r_std_step`.
- `r_mean_abs_dang`, `r_std_dang`, `r_skew_step`, `r_kurt_step`.
- Kuantil & *straightness*: `r_q25_step`, `r_q75_step`, `r_straight_ratio`.
- **Bounding box** lokal: `bbox_w`, `bbox_h`.

### 4.3. Entropy & Histogram Features

- **Entropy** rolling: `r_entropy_dang` (Δangle), `r_entropy_step` (step).
- **Histogram** rolling untuk Δangle dan step dengan bin global stabil:
  - `h_dang_bin{i}`, `h_dang_p{i}`, `h_dang_sum`.
  - `h_step_bin{i}`, `h_step_p{i}`, `h_step_sum`.

### 4.4. Saccade Rate/Ratio

- Definisi **small/large** step via kuantil global (≈Q25 & Q90):
  - `r_rate_small`, `r_rate_large`, `r_ratio_small_large`.

### 4.5. Fitur dasar tambahan

- `dx`, `dy`, `abs_dx`, `abs_dy`, `step`.

> Total **>27** fitur eksplisit + puluhan fitur histogram/probabilitas dari Δangle & step.

---

## 5) Model & Konfigurasi

### 5.1. XGBoost (v1)

- **Objective:** `binary:logistic`
- **Tree method:** `hist` (opsional ganti `gpu_hist` bila tersedia)
- **Hiperparameter inti (v1):**
  - `n_estimators=750`, `learning_rate=0.055`
  - `max_depth=5`, `min_child_weight=8`, `gamma=0.25`
  - `subsample=0.85`, `colsample_bytree=0.85`
  - `reg_alpha=0.6`, `reg_lambda=2.2`

### 5.2. Ensemble XGBoost (v2)

- **Base config:** `n_estimators=1200`, `learning_rate=0.045`, `tree_method="hist"`, `n_jobs=4`.
- **Keberagaman model:**
  - **Scaler modes:** `robust`, `quantile`, `standard`.
  - **Seeds:** `[42, 7, 1029]`.
  - **Parameter mixes (contoh):**
    - Mix‑A: `max_depth=5`, `min_child_weight=7`, `subsample=0.85`, `colsample_bytree=0.85`, `gamma=0.25`, `reg_alpha=0.6`, `reg_lambda=2.2`.
    - Mix‑B: `max_depth=6`, `min_child_weight=5`, `subsample=0.80`, `colsample_bytree=0.80`, `gamma=0.20`, `reg_alpha=0.5`, `reg_lambda=2.6`.
- **Total model:** 3 scaler × 3 seed × 2 mix = **18** base models (dilatih & diprediksi, lalu **dirata‑rata** probabilitasnya).
- **Threshold search:** dilebarkan (≈0.15–0.85), ambil **best thr** dari validation (**0.510**) untuk uji.

---

## 6) Skema Validasi & Anti‑Overfitting

- **Hold‑out:** 80% train, 20% test (stratified).
- **Validation internal:** subset dari train untuk memilih scaler (v1) / threshold & blending (v2).
- **Langkah pencegahan overfitting:**
  - Fitur berbasis **rolling** & **entropy** untuk merangkum konteks, bukan titik tunggal.
  - **Regulasi** (gamma, reg\_alpha, reg\_lambda) & **subsample/colsample**.
  - **Ensembling** (v2) mengurangi variansi model.
  - **Threshold** diambil dari validation, lalu **dibekukan** untuk test.

> Rekomendasi tambahan: pastikan pemisahan **per subjek** (subject‑wise split) jika ada kebocoran temporal antar sesi orang yang sama. Saat ini split dilakukan per‑baris (stratified) – bagus untuk baseline, tapi split per subjek sering lebih ketat.

---

## 7) Hasil & Evaluasi

### 7.1. v1 – Single XGBoost (auto‑scaler)

- **Validation:**
  - Best scaler: **robust** | Acc **0.7525** | AUC **0.8318** | thr **0.498**.
- **Test:**
  - Acc **0.7514** | AUC **0.8337**.
  - Confusion matrix `[[TN, FP], [FN, TP]]`: `[[18606, 6426], [6019, 19013]]`.
  - Report: Prec/Rec/F1 \~ Sequential: **0.756/0.743/0.749** | Random: **0.747/0.760/0.753**.

### 7.2. v2 – Ensemble XGBoost (soft‑voting)

- **Validation:** Acc **0.8181** | AUC **0.9013** | thr **0.510**.
- **Test:**
  - Acc **0.8216** | AUC **0.9068**.
  - Confusion matrix `[[TN, FP], [FN, TP]]`: `[[21036, 3996], [4937, 20095]]`.
  - Report: Prec/Rec/F1 \~ Sequential: **0.810/0.840/0.825** | Random: **0.834/0.803/0.818**.

**Analisis:**

- V2 menangkap pola urut vs acak lebih baik → **FP & FN turun** signifikan dibanding v1.
- AUC > 0.90 menunjukkan separabilitas yang kuat; threshold dapat disetel sesuai preferensi *precision* vs *recall*.

---

## 8) Cara Menjalankan (Jupyter)

1. Buka `XGBoost_v1.ipynb` atau `XGBoost_v2.ipynb`.
2. Pastikan paket: `numpy`, `pandas`, `scikit‑learn`, `xgboost`.
3. Letakkan `truncated_dataset-seqglo.csv` pada direktori kerja yang sama.
4. Jalankan sel berurutan dari atas (v2 akan melatih beberapa model untuk ensembling → waktu eksekusi lebih lama).
5. (Opsional) Jika punya GPU, di v2 set `TREE_METHOD = "gpu_hist"`.

---

## 9) Artefak Keluaran (v2)

- Model & scaler diserialisasi ke:
  - `xgb_rowlevel_models.pkl`, `xgb_rowlevel_scalers.pkl` (via `joblib`).
- **Catatan:** Saat *inference*, ulangi **seluruh pipeline feature engineering & scaling** persis seperti saat training, lalu muat model & scaler.

---

## 10) Keterbatasan & Saran Lanjutan

- **Split subject‑wise** untuk memastikan generalisasi antar individu.
- Coba **early‑stopping** (jika versi XGBoost mendukung) atau kurangi `n_estimators` bila overfit.
- Tambah fitur: **blink rate**, **fixation duration**, atau **temporal embeddings** sederhana.
- Uji **calibration** (Platt/Isotonic) jika probabilitas akan dipakai untuk keputusan operasional.
- Evaluasi **PR‑AUC** bila distribusi target berubah (tidak lagi seimbang).

---

## 11) Changelog v1 → v2

- **Penambahan ensembling** lintas scaler, seed, dan mix parameter.
- **Threshold search** lebih luas (≈0.15–0.85).
- **Kinerja meningkat:** Acc dari \~**0.75** → **0.82**, AUC dari \~**0.83** → **0.91**.
- **Penyimpanan artefak** (models & scalers) untuk inference.

---

## 12) Parameter Kunci (Ringkas)

- **v1 (XGB\_KW):** `n_estimators=750`, `lr=0.055`, `max_depth=5`, `min_child_weight=8`, `gamma=0.25`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_alpha=0.6`, `reg_lambda=2.2`, `tree_method="hist"`.
- **v2 (BASE\_KW):** `n_estimators=1200`, `lr=0.045`, `tree_method="hist"`; *diversitas* via scaler `robust/quantile/standard`, seeds `[42,7,1029]`, dan **2 mix** parameter (lihat §5.2).

---

## 13) Interpretasi Praktis

- **Sequential (1):** pergerakan lebih terarah/urut → *entropy Δangle* & *circular variance* cenderung **lebih rendah**, *straightness ratio* **lebih tinggi**.
- **Random (2):** pola lompatan (saccade) yang lebih bervariasi → *entropy/variasi* **lebih tinggi**, porsi *large steps* naik, orientasi Δangle lebih menyebar (MRL turun).


