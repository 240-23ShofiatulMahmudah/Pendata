---
title: Klasifikasi Naive Bayes Menggunakan KNIME

---

# Klasifikasi Naive Bayes Menggunakan KNIME

## Deskripsi Tugas

Tugas ini bertujuan untuk melakukan klasifikasi data menggunakan metode **Naive Bayes** dengan bantuan software **KNIME**. Proses dilakukan melalui beberapa tahapan mulai dari membaca data, preprocessing, hingga evaluasi model.

---
## Dataset yang digunakan:

![image](dataset.png)
---

## Alur Workflow KNIME

Berikut urutan node yang digunakan:

![image](progres1.png)
---

## Penjelasan Tiap Node
📥 **Excel Reader**
Membaca dataset dari file Excel / CSV

🎨 **Color Manager**
Memberi warna berdasarkan kelas (misalnya Buy: Yes/No)

📈 **Scatter Plot**
Menampilkan grafik hubungan antar fitur
Hanya untuk visualisasi (tidak mempengaruhi model)

🔀 **Table Partitioner**
Membagi data menjadi:
Training (latih model)
Testing (uji model)

✂️ **Column Filter**
Memilih kolom yang digunakan
Biasanya:
Ambil fitur (Age, Income, dll)
Pastikan label (Buy) tetap ada untuk training & testing

📏 **Normalizer**
Mengubah skala data (misalnya jadi 0–1)
Hanya untuk data training

🔁 **Normalizer (Apply)**
Menerapkan normalisasi ke data testing
Menggunakan model dari Normalizer training

🤖 **Naive Bayes Learner**
Melatih model klasifikasi dari data training
Wajib pilih:
Classification column = Buy

🔮 **Naive Bayes Predictor**
Menghasilkan prediksi dari data testing
Output:
Prediction
Probability

📊 **Scorer**
Mengevaluasi hasil prediksi
Output:
Accuracy
Confusion Matrix

---


## Hasil yang Diperoleh

Setelah menjalankan workflow:

* Didapatkan hasil prediksi pada data testing
* Evaluasi model ditampilkan dalam bentuk confusion matrix
* Nilai akurasi menunjukkan tingkat keberhasilan model

---

## Kesimpulan

Metode Naive Bayes dapat digunakan untuk melakukan klasifikasi data dengan cukup baik. Dengan workflow KNIME, proses klasifikasi menjadi lebih mudah karena tidak memerlukan coding manual.

---

## Lampiran (Opsional)

* Hasil prediksi
![image](hasil88.png)

