# Drought Severity Classification — Indonesia

Klasifikasi biner tingkat keparahan kekeringan di Indonesia (**mild** vs **severe**) berbasis variabel meteorologi **TerraClimate**, mengikuti alur **CRISP-DM**. Fokus utama: **feature engineering berbasis domain knowledge hidrologi** dan penanganan **class imbalance** yang sadar terhadap *data leakage*.

## 📊 Dataset

- **Sumber:** [TerraClimate](https://www.climatologylab.org/terraclimate.html) — Climatology Lab (diakses via Google Earth Engine)
- **Cakupan:** Data iklim bulanan per-kecamatan di Indonesia, 6.000 baris × 18 kolom
- **Target:** `new_label`: biner, diturunkan dari **PDSI (Palmer Drought Severity Index)**:
  - `severe`: PDSI < −2 (kekeringan moderat hingga ekstrem)
  - `mild`: PDSI ≥ −2 (kondisi mild hingga basah)
- **Catatan label:** Threshold ini **terinspirasi skema Palmer (PDSI)** namun **disederhanakan menjadi biner** dengan cutoff yang dipilih untuk kebutuhan modeling — bukan replikasi tabel kategori Palmer asli.

## 🔍 Workflow

1. **Setup & Load**: Memuat dataset TerraClimate.
2. **Formatting & Labeling**: Membentuk label biner dari threshold PDSI, rename kolom agar interpretable, buat kolom `date`.
3. **EDA**: Distribusi, missing value, outlier, korelasi antar fitur.
4. **Preprocessing**: Pengembalian *scale factor* GEE, imputasi `kecamatan_id`, frequency encoding wilayah, IQR capping, `StandardScaler`.
5. **Feature Engineering**: Feature creation berbasis domain knowledge (lihat di bawah).
6. **Modeling**: Random Forest, Decision Tree, SVM (GridSearchCV + StratifiedKFold, scoring `f1_macro`).
7. **Evaluation**: Accuracy, F1 macro, classification report, confusion matrix.

## ⭐ Feature Engineering (Domain Knowledge)

Inti kontribusi project ini: **10 fitur turunan** dirancang dari prinsip hidrologi & klimatologi, bukan asal kombinasi kolom.

### Tier 1 — Water Balance (berbasis indeks kekeringan dengan rujukan ilmiah)

| Fitur | Formula | Rasional | Rujukan |
|---|---|---|---|
| `water_balance` | precipitation - PET | Basis indeks **SPEI**; neraca air (defisit/surplus) | Vicente-Serrano et al. (2010) |
| `aridity_index` | precipitation / PET | Rasio kekeringan iklim (P/PET) | UNEP (1992) |
| `evaporative_stress` | AET / PET | Stres air vegetasi (**ESI/ESR**) | Anderson et al. (2007, 2011) |
| `climatic_water_deficit` | PET - AET | Kekurangan air iklim (konsep CWD hidrologi) | - |

### Tier 2 — Fitur Fisis & Logis (besaran standar / penalaran domain)

| Fitur | Formula | Rasional |
|---|---|---|
| `diurnal_temperature_range` | max_temp - min_temp | Amplitudo suhu harian; proksi kecerahan & kekeringan |
| `relative_humidity_proxy` | VAP / (VAP + VPD) | Proksi kelembapan relatif (*custom*, tanpa rujukan kanonik) |
| `runoff_ratio` | runoff / precipitation | Fraksi curah hujan yang menjadi limpasan |
| `month_sin`, `month_cos` | sin/cos(2π·month/12) | *Cyclic encoding* musiman (bulan bersifat siklik) |
| `monsoon_phase` | binning bulan | Fase monsun: barat (basah, Nov–Mar) / timur (kering, Mei–Sep) / transisi |

> ⚠️ `relative_humidity_proxy` adalah proksi buatan sendiri tanpa referensi ilmiah baku — disertakan apa adanya, tidak diklaim sebagai indeks resmi.

## 🛡️ Penanganan Leakage & Imbalance

- **Anti-leakage:** `pdsi` **dibuang** dari fitur (karena label diturunkan darinya), begitu pula `drought_class`. Kolom `year` juga dibuang karena data hanya mencakup 2 tahun (risiko *temporal leakage*).
- **Imbalance:** **Custom SMOTE** dengan *sampling strategy* per-kelas yang konservatif (kelas mayoritas tidak disentuh, minoritas di-boost terkontrol).

## 🤖 Modeling & Hasil

> 📌 Metrik di bawah berasal dari run notebook saat ini. Jika threshold label diubah, jalankan ulang & perbarui tabel ini.

| Model | CV F1 Macro | Test Accuracy | Test F1 Macro |
|---|---|---|---|
| **Random Forest** ⭐ | 0.930 | **0.954** | **0.940** |
| SVM | 0.907 | 0.940 | 0.927 |
| Decision Tree | 0.905 | 0.941 | 0.926 |

**Best model — Random Forest (per-kelas):**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| mild | 0.96 | 0.98 | 0.97 | 847 |
| severe | 0.94 | 0.89 | 0.91 | 315 |

## 📝 Catatan & Limitasi

Fitur-fitur turunan dibangun dari variabel yang juga merupakan **komponen penyusun PDSI**, sedangkan label berasal dari PDSI. Karena itu, performa tinggi sebagian mencerminkan **rekonstruksi hubungan formula** PDSI, bukan murni prediksi kekeringan dari sinyal independen — sebuah trade-off yang disadari dan layak disebut secara jujur.

## ▶️ How to Run

Project dijalankan di **Google Colab**:
1. Jalankan cell `files.upload()` lalu unggah `dataset_6000_rows_cols_dropped.csv`.
2. `Runtime → Run all`.

## 📚 References

> Palmer, W. C. (1965). *Meteorological Drought*. U.S. Weather Bureau, Research Paper No. 45. https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
>
> Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index (SPEI). *Journal of Climate*.
>
> United Nations Environment Programme (UNEP). (1992). *World Atlas of Desertification*.
>
> Anderson, M. C., et al. (2007, 2011). Evaporative Stress Index (ESI) — thermal-based actual-to-potential evapotranspiration ratio for drought monitoring.
>
> TerraClimate: Climatology Lab. https://www.climatologylab.org/terraclimate.html

## 👤 Author

**Muhammad Faishal Ardiansyah** — [@Ishalllll](https://github.com/Ishalllll)
