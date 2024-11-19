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
- Jumlah Data: 6435 baris dan 8 kolom.
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
  
  ![image](https://github.com/user-attachments/assets/b6e30a51-628d-440a-bb28-ffae821c7bce)
  Sebagian besar penjualan mingguan terkonsentrasi pada nilai rendah (0.1 hingga 0.3 setelah normalisasi). Berikutnya distribusi bersifat positif skewed, menunjukkan ada beberapa outlier dengan 
  penjualan sangat tinggi. Dan yang terakhir, distribusi ini penting untuk memahami pola penjualan secara keseluruhan dan menentukan jika diperlukan perlakuan khusus terhadap outlier.

- Tren Penjualan Mingguan: Tren menunjukkan fluktuasi penjualan yang signifikan selama liburan.

  ![image](https://github.com/user-attachments/assets/dcaa53d6-426b-4479-9fd5-6a86957a7cc1)
  Puncak penjualan terjadi di sekitar bulan liburan, seperti Natal dan Thanksgiving, di mana penjualan naik secara signifikan. Berikutnya, penjualan menunjukkan pola musiman, dengan penurunan      yang konsisten setelah periode liburan. Dan yang terakhir, informasi ini membantu Walmart merencanakan stok lebih baik selama periode puncak.

- Penjualan Berdasarkan Holiday Flag
  Distribusi Weekly_Sales berdasarkan nilai Holiday_Flag (0 untuk minggu non-liburan, 1 untuk minggu liburan)

  ![image](https://github.com/user-attachments/assets/2f35092a-f12c-499b-80ed-649ca22695c6)
  a. Minggu Non-Liburan (Holiday_Flag = 0):
  Median penjualan lebih rendah dibandingkan minggu liburan.
  Terdapat beberapa outlier dengan penjualan sangat tinggi, namun distribusi umumnya lebih terkonsentrasi pada nilai rendah.
  b. Minggu Liburan (Holiday_Flag = 1):
  Median penjualan lebih tinggi dibandingkan minggu non-liburan.
  Variasi penjualan lebih besar, yang menunjukkan dampak signifikan dari liburan terhadap pola penjualan.

## Data Preparation
1. Normalisasi Format Tanggal: Kolom Date yang berisi campuran format (dd/mm/yyyy dan dd-mm-yyyy) dinormalisasi menggunakan fungsi .str.replace() dan dikonversi ke tipe datetime.
   Pada dataset ini ternyata terdapat perbedaan format jadi perlu dihandle dengan beberapa cara seperti mengganti data tanggal '/' dengan '-' setelah itu mengubah ke format datetime yang telah      ditentukan. Saya juga menambahkan teknik handle lainnya yaitu menghapus baris yang memiliki data tidak valid (Nat) namun ternyata tidak ada data yang tida valid.
2. Feature Engineering:
    - Menambahkan fitur baru seperti Year, Month, dan Week dari kolom Date.
3. Scaling: Variabel numerik (Temperature, Fuel_Price, CPI, Unemployment, Weekly_Sales) dinormalisasi menggunakan MinMaxScaler agar berada dalam rentang [0,1].
4. Train-Test Split: Dataset dibagi menjadi 80% training dan 20% testing. Bagian memilih fitur(x) dan target (y), Kolom target (Weekly_Sales) dipisahkan dari fitur-fitur prediktor untuk         
   memastikan proses modeling berjalan dengan benar. Fitur prediktor meliputi informasi toko, data ekonomi, dan waktu.

## Modeling
1. Model 1: Linear Regression
    - Cara Kerja:
      Linear Regression adalah algoritma yang memodelkan hubungan linear antara variabel prediktor dan target. Model ini menghitung parameter (koefisien) yang meminimalkan jumlah kuadrat error         (least squares).
    - Parameter:
      Parameter default digunakan:
      fit_intercept=True: Mengestimasi intercept (bias).
      normalize=False: Tidak dilakukan normalisasi tambahan.
2. Model 2 : Random Forest Regressor
   - Cara Kerja:
     Random Forest adalah algoritma ensemble yang menggabungkan prediksi dari beberapa pohon keputusan. Model ini bekerja dengan membagi data secara acak, membuat beberapa pohon keputusan, dan 
     menggabungkan hasil prediksi.
   - Parameter Default:
     n_estimators=100: Menggunakan 100 pohon dalam ensemble.
     max_depth=None: Tidak ada batasan kedalaman pohon.
     random_state=42: Untuk hasil yang konsisten.
   - Hyperparameter Tuning:
     Hyperparameter dioptimalkan menggunakan GridSearchCV.
     Parameter terbaik:
     n_estimators=200
     max_depth=20
- Kelebihan dan Kekurangan Algoritma
Linear Regression:
Kelebihan: Sederhana, cepat, mudah diinterpretasi.
Kekurangan: Tidak efektif untuk hubungan non-linear.
Random Forest Regressor:
Kelebihan: Menangkap hubungan non-linear, robust terhadap outlier.
Kekurangan: Lebih lambat dan sulit diinterpretasi.


## Evaluation
Metrik Evaluasi yang Digunakan :
1. Mean Absolute Error (MAE): Mengukur rata-rata absolut error prediksi.
2. Mean Squared Error (MSE): Memberikan penalti lebih besar pada error yang besar.
3. R² (R-squared): Proporsi variansi target yang dijelaskan oleh model.

A. Model 1: Linear Regression (Baseline)
Linear Regression digunakan sebagai baseline model:
Hasil Evaluasi:
MAE: 0.12
MSE: 0.02
R²: 0.16
Linear Regression menunjukkan performa yang kurang baik karena hanya menangkap hubungan linear antar variabel.

B. Model 2: Random Forest Regressor
Random Forest digunakan untuk menangkap hubungan non-linear:
Hasil Evaluasi:
MAE: 0.02
MSE: 0.00
R²: 0.96
Random Forest memberikan performa yang jauh lebih baik dibandingkan Linear Regression.


| Model              | MAE      | MSE      | R²      |
|--------------------|----------|----------|---------|
| Linear Regression  | 0.12     | 0.02     | 0.16    |
| Random Forest      | 0.02     | 0.00     | 0.96    |

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


##Referensi:
1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. DOI: 10.1023/A:1010933404324.
2. Kaggle. (2014). Walmart Recruiting - Store Sales Forecasting. Retrieved from https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting.
