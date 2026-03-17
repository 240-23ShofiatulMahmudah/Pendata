---
title: Macam-Macam_Normalisasi_Data_Preprocessing

---

# Macam-Macam_Normalisasi_Data_Preprocessing
Normalisasi data merupakan proses transformasi nilai dalam dataset ke dalam skala tertentu agar semua atribut memiliki rentang nilai yang sebanding. Dalam analisis data dan machine learning, normalisasi sangat penting karena beberapa algoritma sangat sensitif terhadap perbedaan skala data. Misalnya jika suatu atribut memiliki nilai dalam jutaan sementara atribut lain hanya bernilai satuan, maka atribut dengan nilai besar akan mendominasi proses perhitungan. Dengan melakukan normalisasi, semua atribut akan berada dalam rentang yang sebanding sehingga proses analisis menjadi lebih akurat.
## 1.1 Min-Max Normalization
Min-Max Normalization adalah metode normalisasi yang mengubah nilai data ke dalam rentang tertentu, biasanya antara 0 sampai 1. Metode ini bekerja dengan mengurangi setiap nilai data dengan nilai minimum kemudian membaginya dengan selisih antara nilai maksimum dan minimum dari dataset tersebut. Teknik ini sering digunakan dalam berbagai algoritma machine learning karena mudah diterapkan dan dapat menjaga distribusi data tetap proporsional.
**Rumus:**
$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$
Contoh:

IPK
data = 2,3,4
min = 2
max = 4
IPK = 3
$$
x' = \frac{3 - 2}{4 - 2}
$$
$$
x' = 0.5
$$

**code sklearn**
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    data = np.array([[2],[3],[4]])

    scaler = MinMaxScaler()

    hasil = scaler.fit_transform(data)

    print(hasil)

## 1.2 Z-Score Normalization
Z-Score Normalization atau Standardization adalah metode normalisasi yang mengubah data berdasarkan nilai rata-rata (mean) dan standar deviasi. Metode ini menghasilkan distribusi data dengan rata-rata nol dan standar deviasi satu. Teknik ini sangat berguna ketika data memiliki distribusi normal dan sering digunakan dalam berbagai algoritma statistik maupun machine learning seperti logistic regression dan support vector machine.
**Rumus:**
$$
z = \frac{x - \mu}{\sigma}
$$
Dimana
μ = mean
σ = standar deviasi

**Code sklearn**
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    hasil = scaler.fit_transform(data)

    print(hasil)

## 1.3 Decimal Scaling
Decimal Scaling merupakan metode normalisasi yang dilakukan dengan cara membagi nilai data dengan bilangan 10 berpangkat tertentu. Nilai pangkat tersebut ditentukan berdasarkan jumlah digit terbesar pada dataset. Tujuan dari metode ini adalah untuk mengubah nilai data menjadi lebih kecil tanpa mengubah pola distribusi data secara signifikan. Metode ini cukup sederhana namun jarang digunakan dibandingkan metode normalisasi lainnya.
**Rumus:**$$
x' = \frac{x}{10^j}
$$
Contoh:
data = 2000000
j = 7
$$
x' = 0.2
$$


## 1.4 Mean Normalization
Mean Normalization merupakan metode normalisasi yang dilakukan dengan cara mengurangi nilai data dengan nilai rata-rata kemudian membaginya dengan selisih antara nilai maksimum dan minimum. Metode ini menghasilkan data yang berada di sekitar nilai nol sehingga distribusi data menjadi lebih seimbang. Teknik ini sering digunakan ketika ingin mempertahankan hubungan distribusi data namun tetap menyesuaikan skala antar atribut.
**Rumus:**
$$
x' = \frac{x - \mu}{x_{max} - x_{min}}
$$
