# Customer Segmentation with K-Means

Eksperimen machine learning untuk melakukan segmentasi pelanggan menggunakan algoritma K-Means pada dataset Mall Customers. Proyek ini mencakup data preprocessing, exploratory data analysis (EDA), hyperparameter tuning, experiment tracking dengan MLflow, model serving menggunakan Flask, serta monitoring dengan Prometheus dan Grafana.

## Overview

Tujuan proyek ini adalah mengelompokkan pelanggan berdasarkan karakteristik demografis dan perilaku belanja untuk membantu analisis target pelanggan dan perencanaan strategi pemasaran.

Dataset berisi 200 pelanggan dengan informasi:

- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

CustomerID digunakan sebagai identifier dan dihapus sebelum proses clustering.

## Dataset

Sumber dataset: [Mall Customer Segmentation Data on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

File utama:

- Mall_Customers_raw.csv - dataset mentah dengan 200 baris dan 5 kolom
- preprocessing/Mall_Customers_preprocessed.csv - dataset hasil preprocessing dengan 200 baris dan 4 fitur model

## Workflow

~~~text
Raw dataset
    |
    v
Preprocessing and EDA
    |
    v
K-Means training and tuning for k = 2..8
    |
    v
MLflow experiment tracking and model artifacts
    |
    v
Flask inference service
    |
    v
Prometheus metrics and Grafana monitoring
~~~

## Preprocessing

Preprocessing diimplementasikan dalam preprocessing/automation.py dan notebook eksplorasi.

1. Menghapus kolom CustomerID.
2. Menghapus baris duplikat.
3. Mengisi missing value numerik dengan median.
4. Mengisi missing value kategorikal dengan modus.
5. Menangani outlier numerik menggunakan IQR clipping.
6. Mengubah Gender menjadi nilai numerik menggunakan LabelEncoder.
7. Melakukan normalisasi fitur numerik menggunakan MinMaxScaler.

Jalankan preprocessing dengan:

~~~bash
python preprocessing/automation.py
~~~

GitHub Actions juga menjalankan preprocessing secara otomatis setiap kali ada push ke branch main, kemudian menyimpan perubahan pada dataset hasil preprocessing.

## Model and Tuning

Model yang digunakan adalah KMeans dengan konfigurasi:

- Initialization: k-means++
- n_init: 10
- max_iter: 300
- random_state: 42
- Candidate cluster counts: k = 2, 3, 4, 5, 6, 7, 8

Pemilihan model dilakukan menggunakan silhouette score. Seluruh eksperimen dan artefak model dicatat menggunakan MLflow.

## Results

Hasil tuning yang tersimpan di MLflow:

| Number of clusters | Silhouette score |
| ---: | ---: |
| 2 | 0.5164 |
| 3 | 0.4547 |
| 4 | 0.3598 |
| 5 | 0.3666 |
| 6 | 0.3754 |
| 7 | 0.3856 |
| 8 | 0.3959 |

Model terbaik menggunakan k=2 dengan silhouette score 0.5164.

Silhouette score adalah metrik evaluasi internal untuk clustering, bukan accuracy seperti pada supervised learning. Label cluster bersifat arbitrer, sehingga setiap cluster perlu diprofilkan berdasarkan rata-rata usia, pendapatan, dan spending score sebelum diberi nama bisnis.

Artefak visual:

- Submissions/Membangun_model/elbow_plot.png
- Submissions/Membangun_model/cluster_visualization.png
- Submissions/Membangun_model/screenshot_dashboard.png
- Submissions/Membangun_model/screenshot_artefak.png

## Experiment Tracking

Eksperimen dicatat dalam MLflow dengan experiment name Mall_Customer_Clustering. Repository menyertakan parameter, metrik, input example, model artifacts, dan metadata environment pada folder Submissions/Membangun_model/mlruns/.

Tracking repository DagsHub:

https://dagshub.com/Neuraly4/SMSML-Daffa-Naufal-Mumtaz-Heryadi

## Model Serving

Model hasil training digunakan oleh Flask inference service dengan endpoint:

~~~text
POST http://localhost:5000/predict
~~~

Contoh payload:

~~~json
{
  "Gender": "Female",
  "Age": 30,
  "Annual Income (k$)": 70,
  "Spending Score (1-100)": 75
}
~~~

Contoh response:

~~~json
{
  "cluster": 1,
  "status": "success"
}
~~~

Script serving dan inference berada di:

~~~text
Submissions/Monitoring dan Logging/3. prometheus_exporter.py
Submissions/Monitoring dan Logging/7. inference.py
~~~

## Monitoring

Inference service mengekspor metrik Prometheus pada port 8000, sedangkan endpoint inference berjalan pada port 5000.

Metrik yang dipantau mencakup:

- Total request
- Processing time atau latency
- Memory usage
- Jumlah prediksi per cluster
- Model confidence atau simulated score
- Ringkasan usia, pendapatan, dan spending score input
- Target customer berdasarkan income dan spending score
- Jumlah input anomaly atau invalid input

Konfigurasi Prometheus tersedia di:

~~~text
Submissions/Monitoring dan Logging/2. prometheus.yml
~~~

## Repository Structure

~~~text
.
├── .github/workflows/main.yml
├── Mall_Customers_raw.csv
├── preprocessing/
│   ├── automation.py
│   ├── Eksperimen_Daffa Naufal Mumtaz Heryadi.ipynb
│   └── Mall_Customers_preprocessed.csv
└── Submissions/
    ├── Eksperimen_SML_Daffa-Naufal-Mumtaz-Heryadi.txt
    ├── Workflow-CI.txt
    ├── Membangun_model/
    │   ├── modelling.py
    │   ├── modelling_tuning.py
    │   ├── modelling_tuningdagshub.py
    │   ├── requirements.txt
    │   ├── elbow_plot.png
    │   ├── cluster_visualization.png
    │   └── mlruns/
    └── Monitoring dan Logging/
        ├── 2. prometheus.yml
        ├── 3. prometheus_exporter.py
        ├── 7. inference.py
        ├── 8. best_model(2).pkl
        └── monitoring and serving evidence/
~~~

## Local Setup

Gunakan Python 3.10 atau versi yang lebih baru. Instalasi dependency utama:

~~~bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install pandas numpy scikit-learn matplotlib seaborn mlflow dagshub flask requests prometheus-client joblib
~~~

### Run preprocessing

~~~bash
python preprocessing/automation.py
~~~

### Run model tuning

~~~bash
cd "Submissions/Membangun_model"
python modelling_tuning.py
~~~

### Run inference and monitoring

Jalankan service terlebih dahulu:

~~~bash
cd "Submissions/Monitoring dan Logging"
python "3. prometheus_exporter.py"
~~~

Kemudian, pada terminal terpisah, jalankan generator request:

~~~bash
cd "Submissions/Monitoring dan Logging"
python "7. inference.py"
~~~

Prometheus dapat diarahkan ke exporter pada localhost:8000 menggunakan konfigurasi pada 2. prometheus.yml.

## Limitations

- Dataset hanya berisi 200 pelanggan dan digunakan untuk eksperimen, bukan data produksi.
- Model melakukan clustering sehingga tidak menghasilkan label pelanggan yang sudah didefinisikan sebelumnya.
- Script inference menggunakan data dummy untuk menghasilkan traffic monitoring.
- Nilai confidence yang diekspor pada monitoring merupakan simulated score, bukan probabilitas resmi dari K-Means.
- Dashboard Grafana dan deployment monitoring perlu dikonfigurasi secara terpisah.

## Security Note

Jangan menyimpan token DagsHub, API key, atau credential lain langsung di dalam source code. Gunakan environment variable atau GitHub Secrets. Jika token pernah ter-commit pada repository publik, segera revoke dan generate token baru sebelum repository digunakan kembali.

## License

Tambahkan lisensi yang sesuai apabila repository ini akan didistribusikan atau digunakan sebagai open-source project.
