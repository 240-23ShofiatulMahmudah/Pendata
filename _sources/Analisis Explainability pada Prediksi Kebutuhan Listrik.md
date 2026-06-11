---
title: Analisis Explainability pada Prediksi Kebutuhan Listrik

---

# Analisis Explainability pada Prediksi Kebutuhan Listrik

## 1. Analisa prediksi tentang apa?
Kasus pada notebook tersebut digunakan untuk memprediksi kebutuhan listrik (electricity demand) di wilayah Victoria, Australia.

Dataset yang digunakan adalah vic_electricity yang berisi:

| Variabel    | Keterangan              |
| ----------- | ----------------------- |
| Time        | Waktu pencatatan        |
| Date        | Tanggal                 |
| Demand      | Permintaan listrik (MW) |
| Temperature | Suhu udara              |
| Holiday     | Hari libur atau bukan   |

Pada contoh tersebut, data diubah menjadi frekuensi harian (daily) dan model digunakan untuk:

    Memprediksi Demand (kebutuhan listrik) beberapa hari ke depan berdasarkan data kebutuhan listrik sebelumnya dan suhu udara.
    
---
## 2. Bagaimana bentuk data trainingnya?

### Input (X)
**a. Lag Demand**
Nilai demand pada hari-hari sebelumnya:

* lag_1 = demand 1 hari sebelumnya
* lag_2 = demand 2 hari sebelumnya
* lag_3 = demand 3 hari sebelumnya
* lag_4 = demand 4 hari sebelumnya
* lag_5 = demand 5 hari sebelumnya
* lag_6 = demand 6 hari sebelumnya
* lag_7 = demand 7 hari sebelumnya

**b. Variabel eksternal (Exogenous Variable)**
* Temperature

Sehingga bentuk X_train adalah:
| lag_1  | lag_2  | lag_3  | lag_4  | lag_5  | lag_6  | lag_7  | Temperature |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----------- |
| 205338 | 211066 | 213792 | 258955 | 275490 | 227778 | 82531  | 24.09       |
| 200693 | 205338 | 211066 | 213792 | 258955 | 275490 | 227778 | 20.22       |

---
### Output (y)
**Demand hari berikutnya**
| Tanggal    | y      |
| ---------- | ------ |
| 2012-01-07 | 200693 |
| 2012-01-08 | 200061 |
| 2012-01-09 | 216201 |

Artinya model belajar:

    "Jika saya mengetahui demand 7 hari terakhir dan suhu hari ini, berapa demand listrik hari berikutnya?

---
## 3. Apa itu Lag?
Dalam time series, lag adalah nilai masa lalu yang digunakan sebagai fitur (input) untuk memprediksi masa depan.

Misalkan data demand:
| Hari   | Demand |
| ------ | ------ |
| Senin  | 100    |
| Selasa | 120    |
| Rabu   | 130    |
| Kamis  | ?      |

Ketika ingin memprediksi Kamis:

| Feature | Nilai        |
| ------- | ------------ |
| lag_1   | 130 (Rabu)   |
| lag_2   | 120 (Selasa) |
| lag_3   | 100 (Senin)  |

Target: 
| y            |
| ------------ |
| Demand Kamis |

Secara umum:

lag_1 = nilai 1 periode sebelumnya
lag_2 = nilai 2 periode sebelumnya
lag_n = nilai n periode sebelumnya

Pada notebook ini digunakan:

**lags = 7**
yang berarti model melihat data demand selama 7 hari terakhir untuk memprediksi hari berikutnya.

---
## 4. Jelaskan proses analisis yang dilakukan

### Tahap 1 – Mengambil Dataset
Dataset vic_electricity diambil dari package skforecast.

Berisi:

* Demand listrik
* Temperatur
* Waktu pencatatan

---
### Tahap 2 – Resampling Data
Data awal berupa data per jam.

Kemudian diubah menjadi data harian:
```
data = data.resample('D').agg({
    'Demand': 'sum',
    'Temperature': 'mean'
})
```

Hasilnya:

* Demand harian
* Suhu rata-rata harian

---
### Tahap 3 – Split Data
Dataset dibagi menjadi:

**Training** -> untuk belajar

2011–2014

**Testing** -> untuk evaluasi prediksi

Akhir 2014

```
data_train
data_test
```
---

### Tahap 4 – Membuat Model Forecasting
Digunakan:
```
ForecasterRecursive
```

dengan algoritma:
```
LGBMRegressor
```

dan:
```
lags = 7
```

Artinya:

Model menggunakan:

* demand 7 hari terakhir
* temperatur

untuk memprediksi demand berikutnya.

---
### Tahap 5 – Training Model
```
forecaster.fit(
    y=data_train['Demand'],
    exog=data_train['Temperature']
)
```

Pada tahap ini model belajar pola:

* konsumsi listrik
* pengaruh suhu terhadap konsumsi listrik

---
### Tahap 6 – Feature Importance

Dilakukan analisis fitur yang paling berpengaruh.

Hasil dokumentasi menunjukkan:
| Feature     | Importance |
| ----------- | ---------- |
| Temperature | 570        |
| lag_1       | 470        |
| lag_3       | 387        |
| lag_2       | 362        |
| lag_7       | 325        |

Artinya:

Faktor paling berpengaruh:
1. Temperature
1. Demand kemarin (lag_1)
1. Demand beberapa hari sebelumnya

Kesimpulan:

    Suhu memiliki pengaruh terbesar terhadap kebutuhan listrik.
    
---
### Tahap 7 – SHAP Analysis
```
shap.TreeExplainer()
```

untuk mengetahui:

* mengapa model membuat prediksi tertentu
* fitur mana yang menaikkan atau menurunkan prediksi

SHAP membantu menjawab:

            "Mengapa demand hari ini diprediksi tinggi?"

Misalnya karena:

* suhu tinggi
* demand minggu lalu tinggi

---
### Tahap 8 – SHAP Summary Plot
Menampilkan ranking kontribusi fitur.

Tujuannya:

* mengetahui fitur paling penting
* mengetahui arah pengaruh fitur

Contoh:

* Temperature tinggi → demand naik
* Temperature rendah → demand turun

---
### Tahap 9 – SHAP Force Plot
Digunakan untuk menjelaskan satu prediksi tertentu.

Misalnya:

Prediksi Demand = 240.000 MW

Kemudian SHAP menunjukkan:

* Temperature menambah +15.000
* lag_1 menambah +10.000
* lag_4 mengurangi −5.000

Sehingga dapat diketahui asal-usul prediksi tersebut.

---
### Tahap 10 – Prediksi Masa Depan
Model memprediksi 10 hari ke depan:
```
predictions = forecaster.predict(
    steps=10,
    exog=data_test['Temperature']
)
```
Contoh hasil:
| Tanggal    | Prediksi Demand |
| ---------- | --------------- |
| 2014-12-22 | 241514          |
| 2014-12-23 | 226165          |
| 2014-12-24 | 220506          |

---
## Kesimpulan
Kasus pada notebook ini adalah peramalan kebutuhan listrik harian (electricity demand forecasting) menggunakan algoritma LightGBM dan framework Skforecast. Input model berupa 7 lag demand sebelumnya dan temperatur, sedangkan outputnya adalah demand listrik hari berikutnya. Setelah model dilatih, dilakukan analisis interpretabilitas menggunakan Feature Importance dan SHAP untuk mengetahui faktor-faktor yang paling mempengaruhi hasil prediksi. Dari hasil analisis, Temperature dan Demand beberapa hari sebelumnya (lag) merupakan variabel yang paling berpengaruh terhadap prediksi kebutuhan listrik.