# Bushfire Warning Classification: Perth, Western Australia

> Multi-class machine learning untuk memprediksi **tingkat peringatan bushfire** (Advice, Watch and Act, Emergency Warning) berdasarkan kondisi meteorologi, kekeringan, vegetasi, dan topografi di Perth, Western Australia.
>
> Proyek ini merupakan **pengembangan (development & extension)** di atas paper acuan **Utamima, Kanedi & Sohel (2026)** yang terbit di *Environmental Challenges*, dengan dua kontribusi orisinal: **custom feature engineering (feature creation)** dan **implementasi SMOTE buatan sendiri (from-scratch)** untuk menangani class imbalance.

![status](https://img.shields.io/badge/status-completed-success)
![python](https://img.shields.io/badge/Python-3.x-blue)
![platform](https://img.shields.io/badge/platform-Google%20Colab-orange)
![methodology](https://img.shields.io/badge/methodology-CRISP--DM-informational)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Overview

Bushfire adalah salah satu hazard paling kritis di Australia. Sejak **15 Juli 2024**, Western Australia secara resmi mengadopsi **Australian Warning System (AWS)** yang menggunakan tiga tingkat peringatan bushfire: **Advice (kuning)**, **Watch and Act (oranye)**, dan **Emergency Warning (merah)**.

Project ini membangun pipeline klasifikasi multi-class yang memetakan kondisi lingkungan ke tingkat peringatan, mengikuti kerangka **CRISP-DM** dari *business understanding* hingga *evaluation*. Fokus utamanya bukan sekadar mengejar akurasi, tetapi menangani **class imbalance** secara serius (Emergency Warning jauh lebih jarang daripada Advice) dan **memperkaya sinyal prediktif lewat feature engineering**.

**Tujuan:**
- Mengklasifikasikan tingkat peringatan bushfire ke dalam 3 kelas (Advice, Watch and Act, Emergency Warning).
- Menangani distribusi kelas yang timpang tanpa mengorbankan kelas minoritas (kelas paling berbahaya).
- Menghasilkan model yang reproducible dan dapat dijalankan end-to-end di Google Colab.

---

## Dataset

| Item | Detail |
|---|---|
| **Region** | Perth, Western Australia |
| **Jumlah baris** | 1.288 baris |
| **Target** | Tingkat peringatan bushfire: **Advice**, **Watch and Act**, **Emergency Warning** (multi-class) |
| **Tipe fitur** | Meteorologi, kekeringan, vegetasi/bahan bakar, dan topografi |
| **Jumlah fitur final** | 18 fitur (setelah feature creation dan feature selection) |

Fitur mentah mencakup variabel meteorologi (`Rainfall`, `Maximum Temperature`, `RH_min`, `Sunshine`, serta waktu dan arah angin), indikator kekeringan (`DrySpellLength`, `AntecedentRain14`), kondisi bahan bakar/vegetasi (`dw_grass`), dan topografi (`Aspect`). Target berupa tingkat peringatan sesuai **Australian Warning System** sebagaimana diterapkan oleh DFES Western Australia.

---

## Methodology (CRISP-DM)

1. **Business Understanding:** Memahami kebutuhan early-warning bushfire & dampak kesalahan klasifikasi kelas berbahaya.
2. **Data Understanding:** Eksplorasi distribusi kelas, korelasi fitur meteorologi/kekeringan/vegetasi/topografi, dan deteksi class imbalance.
3. **Data Preparation:** Cleaning, penanganan duplikat & outlier, encoding, scaling, dan **feature engineering** (lihat bagian Kontribusi). Feature selection menggunakan **Cramér's V** untuk fitur kategorikal dan uji statistik filter untuk fitur numerik, menyisakan 18 fitur.
4. **Modeling:** Training beberapa model klasifikasi multi-class dengan penanganan imbalance via **custom SMOTE** (dalam `ImbPipeline`, sehingga oversampling hanya pada data training).
5. **Evaluation:** Penilaian dengan metrik yang sensitif terhadap kelas minoritas (F1-score, precision, recall, bukan hanya accuracy).
6. **Deployment (light):** Notebook reproducible yang bisa dijalankan ulang di Google Colab.

---

## Contributions (Reference Paper)

Bagian ini menjelaskan **apa yang saya kembangkan sendiri** di atas paper acuan Utamima et al. (2026). Paper tersebut menjadi *baseline / benchmark* konseptual; project ini menambahkan dua komponen orisinal.

### 1. Feature Engineering, Feature Creation
Saya menurunkan fitur-fitur baru dari variabel mentah untuk memperkuat sinyal prediktif terhadap tingkat peringatan:

- **Cyclical encoding (sin/cos):** untuk variabel yang bersifat siklik, yaitu **bulan**, **jam (Time)**, **day-of-year**, **waktu hembusan angin maksimum**, dan **arah angin maksimum**. Encoding ini menjaga kontinuitas siklik (mis. Desember dekat dengan Januari) yang hilang jika dipakai sebagai angka biasa.
- **Binning topografi:** `Aspect` (0 sampai 360 derajat) dikelompokkan menjadi arah mata angin (**North, East, South, West**).
- **Binning musim & dekomposisi tanggal:** `month` dipetakan menjadi **season** (Summer, Autumn, Winter, Spring), serta ekstraksi komponen tanggal (year, day-of-year, day-of-week, dsb).

Setelah feature selection, fitur cyclical hasil rekayasa ini (`time_sin/cos`, `doy_sin/cos`, `month_sin/cos`, `wind_sin/cos`) terpilih sebagai bagian dari 18 fitur final, menandakan kontribusinya terhadap model.

### 2. Custom SMOTE Implementation
Alih-alih memakai konfigurasi default, saya merancang **strategi oversampling per-kelas** yang konservatif agar kelas minoritas terangkat tanpa menghasilkan sampel sintetis berlebihan:

- **ADVICE (mayoritas):** dibiarkan, 759 sampel.
- **WATCH AND ACT (kelas tengah):** dinaikkan ke 60% mayoritas, 184 menjadi 455 sampel.
- **EMERGENCY WARNING (kelas terkecil):** dinaikkan ke 40% mayoritas, 87 menjadi 303 sampel.

SMOTE dijalankan **di dalam `ImbPipeline`**, sehingga hanya diterapkan pada fold training dan **tidak bocor** ke data test.

> **Hubungan dengan paper acuan:** Paper Utamima et al. (2026) membingkai masalah ini sebagai klasifikasi multi-class atas kategori peringatan bushfire di Western Australia, dengan integrasi catatan peringatan resmi, observasi meteorologi, dan indikator kekeringan, evaluasi menggunakan *random split* dan *time-forward seasonal holdout*, serta **association rule mining** untuk relasi kondisi ke peringatan yang interpretable. Project saya mengambil framing multi-class yang sama tetapi menambahkan **feature creation** dan **custom SMOTE** sebagai kontribusi metodologis tersendiri.
>
> *Catatan transparansi:* paper acuan tidak secara eksplisit (pada bagian yang dapat saya verifikasi) menyebutkan SMOTE; penanganan imbalance lewat custom SMOTE ini adalah pendekatan saya, bukan replikasi langsung dari paper.

---

## Models & Results

Tiga model utama dieksplorasi dengan **GridSearchCV** (StratifiedKFold 3-fold, scoring `f1_macro`):

| Model | Test Accuracy | Test F1 Macro | Macro Precision | Macro Recall |
|---|---|---|---|---|
| **Random Forest** (best) | 0.89 | **0.82** | 0.83 | 0.81 |
| XGBoost | 0.90 | 0.80 | 0.83 | 0.78 |
| SVM | 0.80 | 0.67 | 0.66 | 0.68 |

**Best model dipilih berdasarkan F1 Macro**, bukan accuracy, karena metrik ini lebih adil terhadap kelas minoritas. Meski XGBoost sedikit unggul di accuracy (0.90), **Random Forest** lebih kuat mengenali kelas berbahaya (recall Emergency Warning 0.77 vs 0.68 pada XGBoost).

**Random Forest, rincian per-kelas:**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Advice | 0.93 | 0.95 | 0.94 | 190 |
| Watch and Act | 0.75 | 0.72 | 0.73 | 46 |
| Emergency Warning | 0.81 | 0.77 | 0.79 | 22 |

Sebuah **stacking ensemble** (dengan tuning) juga dicoba, tetapi **tidak mengungguli** Random Forest (F1 Macro 0.74), sehingga Random Forest tetap menjadi model final.

---

## How to Run

Project dijalankan di **Google Colab**:

1. **Clone atau buka repositori**, lalu buka notebook di folder `notebooks/` melalui Google Colab.
2. **Siapkan dataset.** Unggah `dataset-bushfires-p2.csv` (tersedia di folder `data/`) ke Colab. Notebook saat ini membaca dataset dari path Google Drive pribadi, jadi sesuaikan path pada `pd.read_csv(...)` agar menunjuk ke lokasi file Anda.
3. **Jalankan semua cell:** klik `Runtime`, lalu `Run all`.

Sebagian besar dependency (pandas, numpy, scikit-learn, xgboost, imbalanced-learn) sudah tersedia secara default di Colab.

---

## Tech Stack

- **Python 3.x**
- **pandas**, **numpy**, untuk data wrangling & feature engineering
- **scikit-learn**, untuk modeling & evaluation
- **XGBoost**, untuk salah satu model kandidat
- **imbalanced-learn**, untuk `ImbPipeline` & oversampling (custom SMOTE)
- **Google Colab**, runtime environment

---

## References

Paper acuan / benchmark project ini:

> Utamima, A., Kanedi, F. J., & Sohel, F. (2026). Environmental drivers of bushfire warning categories: insights from official records, meteorology, and drought data in Western Australia. *Environmental Challenges, 23*, 101509. https://doi.org/10.1016/j.envc.2026.101509

- Journal: *Environmental Challenges* (Elsevier), ISSN 2667-0100
- Open Access (CC BY-NC 4.0)
- ScienceDirect (PII): S2667010026001034

Referensi pendukung (Australian Warning System):
> Department of Fire and Emergency Services (DFES), Western Australia. *Australian Warning System.* https://www.dfes.wa.gov.au/hazard-information/warning-systems/australian-warning-system

---

## 👤 Author

**Muhammad Faishal Ardiansyah**
Information Systems Student @ Institut Teknologi Sepuluh Nopember (ITS), Surabaya
Aspiring Data Scientist, interested in machine learning & data-driven decision making

- GitHub: [@Ishalllll](https://github.com/Ishalllll)

---

## 📝 License

Released under the **MIT License**.

---
