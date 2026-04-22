---
title: Untitled

---

# Analisa Data Kesuburan Tanah
### 1. Lakukan analisis data dengan menggunakan 
* **K-Nearest Neighbors (KNN)**
### 2. Lakukan Pemrosesan Data tersebut
### 3. Hitung Metrik Evaluasi

| Metrik | Keterangan | 
| -------- | -------- | 
| Accuracy     | Persentase prediksi benar dari total data    |
| Precision | Ketepatan prediksi kelas positif |
| Recall | Kemampuan mendeteksi seluruh kelas positif |
| F1-Score | Harmonic mean antara Precision dan Recall |



## 
### Deskripsi Umum
Dataset berisi 2.000 sampel data tanah dengan 10 fitur agronomis dan 1 kolom label yang membagi kondisi tanah menjadi dua kelas: Subur dan Tidak Subur. Data mengandung missing values (data hilang)

![dataset](dataset_kesuburan_tanah_missing.xlsx)

## Informasi Dataset

| Column 1 | Column 2 | 
| -------- | -------- | 
|   Jumlah Sampel   |   2.000 baris   | 
|   Jumlah Fitur   |   10 fitur (9 numerik, 1 kategorikal)   | 
|   Jumlah Kelas   |   2 kelas   | 
|   Target / Label   |   Subur / Tidak Subur   | 

## langkah-langkah 
### 1. Import Data ke KNIME

Tambahkan node Excel Reader, kemudian:

Masukkan dataset kesuburan tanah (.xlsx)
Klik Execute untuk menampilkan data


---

### 2. Pemrosesan Data
a. Missing Value

Tambahkan node Missing Value karena dataset mengandung data kosong.
Pengaturan:

Numeric → Mean / Median
Nominal → Most Frequent

Tujuan:
Menghindari error saat proses klasifikasi karena KNN tidak bisa membaca data kosong.

---

b. Normalisasi Data

Gunakan node Normalizer dengan metode:
Min-Max (0–1)
Algoritma KNN menggunakan perhitungan jarak (distance), sehingga semua fitur harus berada dalam skala yang sama agar tidak bias.


---

c. Transformasi Data Kategorikal

Gunakan node One to Many
Fungsi:
Mengubah data kategorikal menjadi numerik (One Hot Encoding)

KNN hanya dapat memproses data numerik


---

### 3. Pembagian Data

Gunakan node Table Partitioner
Pengaturan:

Training Data = 70%
Testing Data = 30%

Data dibagi menjadi data latih dan data uji
Model dilatih menggunakan training data dan diuji menggunakan testing data


---

### 4. Penerapan Algoritma KNN

Gunakan node K Nearest Neighbor
Pengaturan:

Target Column → label (Subur / Tidak Subur)
K (jumlah tetangga) → 3

Nilai K kecil (seperti 3) cukup baik untuk dataset ini
Menghindari overfitting dan menjaga akurasi


---

### 5. Proses Prediksi
Data training digunakan untuk membangun model
Data testing digunakan untuk melakukan prediksi
Output:
Label prediksi (Subur / Tidak Subur)


---

### 6. Evaluasi Model

Gunakan node Scorer
Hasil yang diperoleh:

📌 Accuracy
Persentase jumlah prediksi yang benar dari total data
📌 Precision
Tingkat ketepatan model dalam memprediksi kelas positif (Subur)
📌 Recall
Kemampuan model dalam mendeteksi seluruh data kelas positif
📌 F1-Score
Rata-rata harmonis antara Precision dan Recall


---

### 7. Confusion Matrix

Confusion Matrix digunakan untuk melihat detail hasil klasifikasi:
![image](hasil_confusion.jpeg)


---

### 8. Interpretasi Hasil
![image](hasil_prosesdata.jpeg)