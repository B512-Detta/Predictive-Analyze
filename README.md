# Laporan Proyek Machine Learning - Bernadetta Sri Endah Dwi 

## Domain Proyek

**Latar Belakang**
Walmart adalah salah satu perusahaan ritel terbesar di dunia. Kemampuan untuk memprediksi penjualan mingguan sangat penting untuk pengambilan keputusan bisnis, termasuk manajemen inventaris, perencanaan promosi, dan pengelolaan rantai pasokan. Data historis yang mencakup variabel eksternal seperti suhu, harga bahan bakar, tingkat pengangguran, dan hari libur nasional dapat dimanfaatkan untuk membangun model prediktif yang akurat.

**Mengapa Masalah Ini Penting?**
- Membantu Walmart memastikan ketersediaan stok di minggu-minggu sibuk.
- Mengurangi biaya operasional dengan manajemen stok yang efisien.
- Mendukung perencanaan strategi promosi berdasarkan pola penjualan.

## Business Understanding
### Problem Statements
1. Bagaimana cara memprediksi penjualan mingguan di setiap toko Walmart berdasarkan data historis?
2. Apa saja variabel eksternal yang memengaruhi penjualan mingguan?
3. Bagaimana membangun model prediktif yang mampu menghasilkan prediksi dengan akurasi tinggi?

### Goals
1. Membuat model machine learning yang dapat memprediksi variabel target Weekly_Sales.
2. Mengidentifikasi variabel yang memiliki hubungan kuat dengan penjualan mingguan.
3. Mengoptimalkan model dengan parameter terbaik untuk meningkatkan akurasi prediksi.

### Solution statements
1. Membangun model baseline menggunakan algoritma Linear Regression.
2. Menggunakan algoritma Random Forest Regressor untuk menangkap hubungan non-linear pada data.
3. Melakukan hyperparameter tuning pada Random Forest untuk meningkatkan performa model.

## Data Understanding
### Dataset
| Jenis              | Keterangan                    |
|--------------------|-------------------------------|
| Title              | Walmart Condensed Sales Data  |
| Source             | https://www.kaggle.com/datasets/souravprakashai/walmart-condensed-sales-data|
| License            | https://cdla.dev/permissive-1-0/|
| Visibility         | Public                        |
| Tags               | Retail and Shopping, Tabular, Data Visualization|
| Usability          | 9.41                          |

**Dataset yang digunakan adalah data historis penjualan Walmart, yang mencakup fitur berikut:**
- Jumlah Data: 421.570 baris dan 8 kolom.
- Kondisi Data:
    - Missing Values: Tidak ada nilai kosong.
    - Duplicate Values: Tidak ada duplikasi.
    - Outliers: Outlier ditemukan pada variabel Weekly_Sales.

**Variable - variable pada dataset:**
- Store: ID toko.
- Date: Tanggal penjualan.
- Weekly_Sales: Penjualan mingguan (target).
- Holiday_Flag: Indikator apakah minggu tersebut adalah minggu liburan (1: Ya, 0: Tidak).
- Temperature: Suhu rata-rata mingguan.
- Fuel_Price: Harga bahan bakar rata-rata mingguan.
- CPI: Indeks Harga Konsumen.
- Unemployment: Tingkat pengangguran.

### Exploratory Data Analysis (EDA)
- Distribusi Penjualan Mingguan (Weekly_Sales): Distribusi menunjukkan adanya outlier, terutama di minggu-minggu dengan liburan nasional.
- Korelasi Antar Variabel: Variabel seperti Holiday_Flag dan Temperature menunjukkan hubungan moderat terhadap Weekly_Sales.
- Tren Penjualan Mingguan: Tren menunjukkan fluktuasi penjualan yang signifikan selama liburan.

## Data Preparation
1. Normalisasi Format Tanggal: Kolom Date yang berisi campuran format (dd/mm/yyyy dan dd-mm-yyyy) dinormalisasi menggunakan fungsi .str.replace() dan dikonversi ke tipe datetime.
2. Feature Engineering:
    - Menambahkan fitur baru seperti Year, Month, dan Week dari kolom Date.
3. Scaling: Variabel numerik (Temperature, Fuel_Price, CPI, Unemployment, Weekly_Sales) dinormalisasi menggunakan MinMaxScaler agar berada dalam rentang [0,1].
4. Train-Test Split: Dataset dibagi menjadi 80% training dan 20% testing.

## Modeling
1. Model 1: Linear Regression (Baseline)
Linear Regression digunakan sebagai baseline model:
Hasil Evaluasi:
MAE: 0.12
MSE: 0.02
R²: 0.16
Linear Regression menunjukkan performa yang kurang baik karena hanya menangkap hubungan linear antar variabel.

2. Model 2: Random Forest Regressor
Random Forest digunakan untuk menangkap hubungan non-linear:
Hasil Evaluasi:
MAE: 0.02
MSE: 0.00
R²: 0.96
Random Forest memberikan performa yang jauh lebih baik dibandingkan Linear Regression.

- Kelebihan dan Kekurangan Algoritma
Linear Regression:
Kelebihan: Sederhana, cepat, mudah diinterpretasi.
Kekurangan: Tidak efektif untuk hubungan non-linear.
Random Forest Regressor:
Kelebihan: Menangkap hubungan non-linear, robust terhadap outlier.
Kekurangan: Lebih lambat dan sulit diinterpretasi.

| Model              | MAE      | MSE      | R²      |
|--------------------|----------|----------|---------|
| Linear Regression  | 0.12     | 0.02     | 0.16    |
| Random Forest      | 0.02     | 0.00     | 0.96    |


## Evaluation
Metrik Evaluasi yang Digunakan :
1. Mean Absolute Error (MAE): Mengukur rata-rata absolut error prediksi.
2. Mean Squared Error (MSE): Memberikan penalti lebih besar pada error yang besar.
3. R² (R-squared): Proporsi variansi target yang dijelaskan oleh model.

**Hasil Evaluasi**
Random Forest memberikan performa terbaik dengan R² sebesar 0.96, menunjukkan model dapat menjelaskan 96% variansi data.

**Dampak Terhadap Business Understanding**
1. Menjawab Problem Statement:
Model Random Forest berhasil memprediksi penjualan mingguan dengan akurasi tinggi (R² = 0.96).
2. Mencapai Goals:
Model berhasil mengidentifikasi hubungan signifikan antara variabel seperti Holiday_Flag dan Weekly_Sales.
3. Solusi yang Dirancang:
Hyperparameter tuning memberikan dampak positif terhadap peningkatan performa model.

**Kesimpulan**
Random Forest adalah model terbaik untuk prediksi penjualan mingguan Walmart.
Model dapat digunakan untuk membantu manajemen inventaris dan perencanaan strategi promosi.

**Saran**
Untuk pengembangan lebih lanjut, pertimbangkan untuk memasukkan variabel tambahan seperti jenis produk atau data promosi.
