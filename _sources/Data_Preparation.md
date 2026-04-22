---
title: Data_Preparation

---

# Data_Understanding

### 1. Sumber Data

Dataset yang digunakan adalah Iris Flower Dataset yang diperoleh dari platform Kaggle. Dataset ini merupakan dataset klasik dalam bidang data mining dan machine learning yang berisi data pengukuran bunga iris dari tiga spesies berbeda, yaitu Setosa, Versicolor, dan Virginica. Dataset ini digunakan sebagai studi kasus untuk penerapan metode klasifikasi dalam penambangan data.
Berikut link dari dataset 
[Dataset Iris](https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download)

### 2. Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini adalah Iris Flower Dataset yang terdiri dari 150 data dengan 5 atribut. Dataset ini digunakan untuk mengklasifikasikan bunga iris ke dalam tiga spesies berdasarkan ukuran bagian bunganya. 
### 3. Eksplorasi Dataset

Dalam proses mengidentifikasi dataset ini, python membantu untuk mempermudah pengidentifikasiannya.
#### Eksplorasi Dataset dengan menggunakan Python:
##### - Upload file CSV
```
from google.colab import files
files.upload()
```
Proses upload file CSV dilakukan untuk memasukkan dataset ke dalam lingkungan Google Colaboratory agar dapat diproses menggunakan bahasa pemrograman Python. Dataset diunggah secara manual dari perangkat pengguna menggunakan fungsi files.upload() yang disediakan oleh Google Colab.

##### - Struktur Dataset

```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("IRIS.csv")
df.head()
```

Berikut tampilan dari code di atas setelah di jalankan:

|index|sepal\_length|sepal\_width|petal\_length|petal\_width|species|
|---|---|---|---|---|---|
|0|5\.1|3\.5|1\.4|0\.2|Iris-setosa|
|1|4\.9|3\.0|1\.4|0\.2|Iris-setosa|
|2|4\.7|3\.2|1\.3|0\.2|Iris-setosa|
|3|4\.6|3\.1|1\.5|0\.2|Iris-setosa|
|4|5\.0|3\.6|1\.4|0\.2|Iris-setosa|

 Dataset memiliki 5 kolom, yaitu:
 1. Sepal Length (cm) (Numeric)
    Menunjukkan panjang kelopak luar bunga iris dalam satuan sentimeter.
 3. Sepal Width (cm) (Numeric)
    Menunjukkan lebar kelopak luar bunga iris dalam satuan sentimeter.
 5. Petal Length (cm) (Numeric)
    Menunjukkan panjang mahkota bunga iris dalam satuan sentimeter.
 7. Petal Width (cm) (Numeric)
    Menunjukkan lebar mahkota bunga iris dalam satuan sentimeter.
 9. Species (Kategorikal - String)
     Menunjukkan jenis bunga iris, yang terdiri dari tiga kategori:

    - Setosa
    - Versicolor
    - Virginica

##### -Statistik Deskriptif Awal

```
df.describe()
```
Statistik deskriptif digunakan untuk mengetahui gambaran umum data seperti nilai rata-rata, nilai minimum, dan maksimum dari setiap atribut numerik.

Dataset Iris memiliki 4 atribut numerik, yaitu:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Berikut contoh statistik deskriptifnya:

|index|sepal\_length|sepal\_width|petal\_length|petal\_width|
|---|---|---|---|---|
|count|150\.0|150\.0|150\.0|150\.0|
|mean|5\.843333333333334|3\.0540000000000003|3\.758666666666666|1\.1986666666666668|
|std|0\.8280661279778629|0\.4335943113621737|1\.7644204199522617|0\.7631607417008414|
|min|4\.3|2\.0|1\.0|0\.1|
|25%|5\.1|2\.8|1\.6|0\.3|
|50%|5\.8|3\.0|4\.35|1\.3|
|75%|6\.4|3\.3|5\.1|1\.8|
|max|7\.9|4\.4|6\.9|2\.5|

Berdasarkan hasil analisis statistik deskriptif, dataset Iris terdiri dari 150 data pada setiap atribut numerik, sehingga dapat disimpulkan tidak terdapat missing value. Nilai rata-rata (mean) menunjukkan bahwa panjang sepal bunga iris adalah 5.84 cm, lebar sepal 3.05 cm, panjang petal 3.76 cm, dan lebar petal 1.20 cm. Standar deviasi tertinggi terdapat pada atribut petal length sebesar 1.76, yang menunjukkan bahwa variasi panjang petal lebih besar dibandingkan atribut lainnya. Nilai minimum dan maksimum menunjukkan rentang data yang masih dalam batas wajar, yaitu sepal length antara 4.3 hingga 7.9 cm dan petal width antara 0.1 hingga 2.5 cm. Selain itu, nilai kuartil (25%, 50%, dan 75%) menunjukkan distribusi data yang relatif stabil tanpa adanya penyimpangan ekstrem yang signifikan. Secara umum, dataset memiliki distribusi yang baik dan layak digunakan untuk proses analisis dan pemodelan pada tahap selanjutnya.

##### - Pengecekan Data Duplikat

```
df.duplicated().sum()
```
Code di atas merupakan code untuk pengecekan data duplikat, berikut untuk hasil dari pengecekan data duplikat:

```
np.int64(3)
```
Jadi setelah dilakukan pengecekan, terdapat data duplikat yaitu ada 3 data duplikat.

##### - Pengecekan Data Null

```
df.isnull().sum()
```

Berikut untuk tampilan hasil dari pengecekan data Null:

|||
|---|---|
|sepal_length|0|
|sepal_width|0|
|petal_length|0|
|petal_width|0|
|species|0|


### 4. Verifikasi Data

Berdasarkan proses verifikasi, dataset Iris terdiri dari 150 data dengan 4 atribut numerik dan 1 atribut kategorikal (species). Hasil pengecekan menunjukkan bahwa seluruh atribut memiliki jumlah data yang sama (count = 150), sehingga dapat disimpulkan tidak terdapat missing value pada dataset. Tipe data juga telah sesuai, di mana atribut sepal length, sepal width, petal length, dan petal width bertipe numerik, sedangkan species bertipe kategorikal. Berdasarkan hasil pengecekan menggunakan Pandas, ditemukan sebanyak 3 data duplikat pada dataset. Keberadaan data duplikat ini tidak menunjukkan kesalahan pada dataset, namun perlu ditangani pada tahap data preparation agar tidak memengaruhi proses pemodelan. 

### 5. Visualisasi Data

##### - Distribusi Jumlah Data per Species

![original image](https://cdn.mathpix.com/snip/images/Gt4lANFXJ1tvApce6vXv0vo4hu08VXJb7_bTkbfgn-M.original.fullsize.png)

  Grafik bar digunakan untuk melihat jumlah data pada setiap species. Berdasarkan grafik, diketahui bahwa setiap species yaitu Iris-setosa, Iris-versicolor, dan Iris-virginica masing-masing memiliki 50 data. Hal ini menunjukkan bahwa dataset dalam kondisi seimbang (balanced dataset), sehingga tidak terdapat ketimpangan jumlah data antar kelas. Kondisi ini sangat baik untuk proses modeling karena dapat membantu menghasilkan model klasifikasi yang lebih akurat dan tidak bias terhadap kelas tertentu.

##### - Distribusi Data Fitur Numerik pada Dataset Iris

![original image](https://cdn.mathpix.com/snip/images/6u595jYczu5oHxd1yDMvEGkfvzQYsyvFrhOICQnLnGY.original.fullsize.png)


Berdasarkan histogram, dapat disimpulkan bahwa seluruh fitur numerik memiliki distribusi yang bervariasi dan tidak terdapat anomali yang ekstrem. Fitur petal_length dan petal_width menunjukkan pola distribusi yang lebih jelas dalam membedakan kelompok data, sehingga kedua fitur ini sangat berpotensi menjadi fitur penting dalam proses klasifikasi pada tahap modeling. Visualisasi ini membantu dalam memahami karakteristik dan penyebaran data pada tahap Data_Understanding dalam metodologi CRISP-DM.

##### - Analisis Penyebaran Data dan Deteksi Outlier Menggunakan Boxplot

![original image](https://cdn.mathpix.com/snip/images/YfGQj5-WjGnpPQF6mWpdtoRSXTPrJIw5sXZ6l769Xo0.original.fullsize.png)

Berdasarkan boxplot, dapat disimpulkan bahwa setiap fitur memiliki penyebaran data yang berbeda. Fitur sepal_width menunjukkan adanya beberapa outlier, sedangkan fitur lainnya memiliki distribusi yang relatif normal tanpa outlier yang signifikan. Fitur petal_length dan petal_width memiliki variasi data yang cukup besar, sehingga berpotensi menjadi fitur penting dalam membedakan species pada tahap modeling. Visualisasi boxplot ini membantu dalam memahami distribusi data dan mendeteksi outlier pada tahap Data_Understanding dalam metodologi CRISP-DM.

#### - Analisa Korelasi

- Analisis korelasi sepal length dan sepal width:

![original image](https://cdn.mathpix.com/snip/images/sRKmyd2sVULbo8caUWfLT9eqt_lCk_W0yBKY_EifcEA.original.fullsize.png)


Analisis korelasi juga dilakukan pada atribut sepal length dan sepal width menggunakan scatter plot dengan bantuan Python. Berdasarkan hasil visualisasi, terlihat bahwa sebaran data tidak membentuk pola yang jelas, yang menunjukkan bahwa hubungan antara kedua atribut tersebut relatif lemah dibandingkan dengan atribut pada bagian petal.

- Analisis korelasi petal length dan petal width

![original image](https://cdn.mathpix.com/snip/images/8bCKkvYBfRcBPq1RZkXrOMc8sxvl-fqDQ-RhbE9lwRo.original.fullsize.png)

Analisis korelasi dilakukan secara visual menggunakan scatter plot dengan bantuan bahasa pemrograman Python. Scatter plot digunakan untuk melihat hubungan antara atribut petal length dan petal width. Berdasarkan hasil visualisasi, terlihat bahwa titik-titik data membentuk pola yang jelas, yang menunjukkan adanya hubungan atau korelasi yang kuat antara kedua atribut tersebut.


### Eksplorasi Dataset dengan menggunakan Aplikasi Orange:

##### - Statistik Deskriptif 

Statistik deskriptif digunakan untuk memahami karakteristik dasar dataset Iris Flower. Analisis ini dilakukan menggunakan Column Statistics pada aplikasi Orange Data Mining. Hasil statistik deskriptif menunjukkan nilai minimum, maksimum, rata-rata, dan standar deviasi dari setiap atribut, sehingga memberikan gambaran umum mengenai sebaran dan variasi data.

Berikut contoh statistik deskriptifnya:

![original image](https://cdn.mathpix.com/snip/images/1QfYry6Ge2WhH-6mPvVdIRTrstVxnt-bPx3SCKn8EtY.original.fullsize.png)

##### - Analisa kolerasi

Analisis korelasi antar atribut dilakukan secara visual menggunakan widget Scatter Plot pada aplikasi Orange Data Mining. Scatter Plot digunakan untuk melihat hubungan antara dua atribut numerik berdasarkan pola sebaran data. Berdasarkan hasil visualisasi, terlihat bahwa beberapa atribut, khususnya pada bagian petal, memiliki hubungan yang kuat, sedangkan atribut lainnya menunjukkan hubungan yang lebih lemah. Analisis ini membantu dalam memahami keterkaitan antar atribut sebelum masuk ke tahap pemodelan.

##### Berikut contoh Analisa Kolerasinya:

- Analisis korelasi sepal length dan sepal width:

![original image](https://cdn.mathpix.com/snip/images/iaM0IQD8x8ELhZrhLL7d07iobX8siwpqS55Ud7nF7uA.original.fullsize.png)


Berdasarkan hasil visualisasi scatter plot antara sepal length dan sepal width, terlihat bahwa titik-titik data tersebar secara acak dan tidak membentuk pola linear yang jelas. Hal ini menunjukkan bahwa hubungan antara kedua atribut tersebut relatif lemah. Dengan kata lain, perubahan nilai sepal length tidak secara konsisten diikuti oleh perubahan sepal width.

- Analisis korelasi petal length dan petal width

![original image](https://cdn.mathpix.com/snip/images/iT4Xj8fRcAwT4AI_oDVDJKNmDaNI3cv0BFxkJoB2VBw.original.fullsize.png)

Berbeda dengan atribut sepal, scatter plot antara petal length dan petal width menunjukkan pola sebaran yang lebih teratur dan cenderung membentuk hubungan linear positif. Titik-titik data terlihat mengikuti arah tertentu, yang menandakan adanya korelasi yang kuat antara kedua atribut petal tersebut.

## Pengukuran Jarak Pada Dataset Iris Flower

### 1. Similarity
Similarity (kemiripan) dalam penambangan data adalah ukuran yang digunakan untuk mengukur secara numerik tingkat kesamaan antara dua objek data berdasarkan atribut-atribut yang dimilikinya. Nilai similarity menunjukkan seberapa mirip dua objek tersebut dalam suatu dataset.

Similarity biasanya dinyatakan dalam bentuk nilai dengan rentang (range) antara 0 sampai 1:

- Nilai 0 → menunjukkan bahwa dua objek sangat berbeda atau tidak memiliki kesamaan.

- Nilai 1 → menunjukkan bahwa dua objek sangat mirip atau identik.

- Semakin tinggi nilai similarity, maka semakin mirip kedua objek tersebut.

Dengan demikian, similarity memberikan gambaran kuantitatif mengenai tingkat kemiripan antar data dalam bentuk angka yang dapat dihitung dan dianalisis secara matematis.

### 2. Dissimilarity

Dissimilarity adalah ukuran numerik dari perbedaan antara dua objek data berdasarkan atribut-atribut yang dimilikinya. Nilai dissimilarity digunakan untuk menunjukkan seberapa besar tingkat ketidaksamaan atau perbedaan antar objek dalam suatu dataset.

Dissimilarity menggambarkan jarak atau perbedaan secara kuantitatif, sehingga dapat dihitung dan dianalisis menggunakan metode matematis.

Secara umum, interpretasi nilai dissimilarity adalah sebagai berikut:

- Nilai dissimilarity akan sangat rendah apabila dua objek semakin mirip.
- Nilai dissimilarity akan semakin besar apabila dua objek semakin berbeda.
- Nilai minimum dissimilarity adalah 0, yang menunjukkan bahwa dua objek identik atau tidak memiliki perbedaan sama sekali.

Dalam beberapa metode, nilai dissimilarity dapat berada dalam rentang tertentu, misalnya [0,1], namun pada metode lain seperti Euclidean Distance, nilainya dapat lebih dari 1 tergantung pada skala data.

#### - Manhattan Distance

Manhattan Distance adalah metode pengukuran jarak yang digunakan untuk menghitung tingkat perbedaan (dissimilarity) antara dua objek dengan menjumlahkan selisih absolut dari setiap atributnya.
Metode ini menghitung jarak berdasarkan jalur horizontal dan vertikal (seperti pola jalan berbentuk grid), bukan garis lurus. Semakin besar nilai Manhattan Distance, maka semakin besar perbedaan antar objek. Nilai minimum jaraknya adalah 0, yang menunjukkan bahwa dua objek identik.
Rumusnya:

$d(i,j) = {|x_{i1} - x_{j1}|+| x_{i2} - x_{j2}|^2 + \ldots + |x_{ip} - x_{jp}|}$


#### - Eucledien Distance

Euclidean Distance merupakan salah satu metode pengukuran jarak yang termasuk dalam kategori dissimilarity, karena metode ini mengukur tingkat perbedaan antara dua objek berdasarkan nilai atribut numeriknya. Semakin besar hasil perhitungan Euclidean Distance, maka semakin besar perbedaan (dissimilarity) antara dua objek tersebut. Sebaliknya, jika hasilnya mendekati 0, maka kedua objek semakin mirip. 
Rumusnya:

$d(i,j) = \sqrt{(|x_{i1} - x_{j1}|^2 +| x_{i2} - x_{j2}|^2 + \ldots + |x_{ip} - x_{jp}|^2})$

#### - Minkowski Distance

Minkowski Distance adalah bentuk generalisasi dari Euclidean dan Manhattan. Artinya Euclidean dan Manhattan adalah bagian dari kelompok
$$
\begin{aligned}
d(i,j) &= \sqrt[h]{|x_{i1}-x_{j1}|^{h} + |x_{i2}-x_{j2}|^{h} + \cdots + |x_{ip}-x_{jp}|^{h}} \\[6pt]
\text{dengan } \quad
i &= (x_{i1}, x_{i2}, \dots, x_{ip}), \\
j &= (x_{j1}, x_{j2}, \dots, x_{jp})
\end{aligned}
$$
i = Melambangkan objek ke-i
j = Melambangkan objek ke-j
xi1 = Nilai fitur 1 dari objek ke-i
xj1 = Nilai fitur 1 dari objek ke-j


### Mengukur jarak dissimilarity pada suatu dataset

#### - Mengukur jarak dataset Iris
Pada dataset Iris diatas, tipe data yang dimiliki adalah bernilai numerik, karena hal itu penghitungan jarak untuk tipe data Iris kali ini akan digunakan metode Eucledian Distance
##### - Perhitungan manual 
Contoh perhitungan secara manual pada baris 2 dan 3 dengan kolom 1:
$$
d(2,1) = \sqrt{(4{,}9 - 5{,}1)^2 + (3 - 3{,}5)^2 + (1{,}4 - 1{,}4)^2 + (0{,}2 - 0{,}2)^2}
$$

$$
= \sqrt{0{,}04 + 0{,}25}
$$

$$
= 0{,}538
$$

$$
d(3,1) = \sqrt{(4{,}7 - 5{,}1)^2 + (3{,}2 - 3{,}5)^2 + (1{,}3 - 1{,}4)^2 + (0{,}2 - 0{,}2)^2}
$$

$$
= \sqrt{0{,}16 + 0{,}09 + 0{,}01 + 0}
$$

$$
= 0{,}50990
$$

##### - Perhitungan Python

```
from scipy.spatial.distance import pdist, squareform

X = df.select_dtypes(include=['float64', 'int64'])
dist = pdist(X, metric='euclidean')
distance_matrix = squareform(dist)

distance_df = pd.DataFrame(distance_matrix)
print(distance_df.iloc[:5, :5])
```

Terdapat tambahan library yang digunakan yaitu library scipy untuk mempermudah kita menghitung Euclidean Distance tanpa perlu coding manual. Selanjutnya data ditampilkan hanya sebesar 5x5 dihitung dari yang pertama

Output yang dihasilkan:
|   | 0        | 1        | 2        | 3        | 4        |
|---|----------|----------|----------|----------|----------|
| 0 | 0.000000 | 0.538516 | 0.509902 | 0.648074 | 0.141421 |
| 1 | 0.538516 | 0.000000 | 0.300000 | 0.331662 | 0.608276 |
| 2 | 0.509902 | 0.300000 | 0.000000 | 0.244949 | 0.509902 |
| 3 | 0.648074 | 0.331662 | 0.244949 | 0.000000 | 0.648074 |
| 4 | 0.141421 | 0.608276 | 0.509902 | 0.648074 | 0.000000 |


##### - Perhitungan Orange
![original image](https://cdn.mathpix.com/snip/images/9PGgc7ny0uksXf9OeJWUUqJ1W74_wYqqPmGLGPGhBL8.original.fullsize.png)
3 perhitungan diatas, memiliki kesamaan nilai, artinya perhitungan yang dilakukan sudah dilakukan secara benar dan urut

## Mengukur Jarak Dataset Tipe Data Campuran
Dataset yang digunakan dalam contoh ini adalah dataset berjudul Student Performance. Dataset ini berisi beberapa atribut dengan tipe data yang berbeda, yaitu tipe data ordinal, numerik, binary, dan kategorikal. Karena adanya perbedaan tipe data tersebut, maka dataset ini termasuk dalam kategori dataset dengan tipe data campuran.

Pada contoh perhitungan ini, jarak dihitung antara data ke-1 dan data ke-4 dengan mempertimbangkan beberapa atribut yang memiliki tipe data berbeda. Setiap tipe data dihitung menggunakan metode yang sesuai, kemudian hasilnya dijumlahkan untuk memperoleh jarak total antar data.

### Ordinal
Atribut parental level of education termasuk dalam tipe data ordinal, karena memiliki tingkatan atau urutan. Oleh karena itu, setiap kategori terlebih dahulu diberikan nilai peringkat sebagai berikut:

     Ordinal =  Some high school     1
                High school          2
                Some collage         3
                Associate's degree   4
                Bachelor's degree    5
                Master's degree      6
Karena nilai ordinal tidak dapat langsung digunakan dalam perhitungan jarak, maka dilakukan proses normalisasi agar nilainya berada pada rentang 0 sampai 1.

Rumus normalisasi yang digunakan adalah:
     $z = \frac{r - \min}{\max - \min}$

Hasil normalisasi dari setiap peringkat adalah sebagai berikut:

    Normalisasi = r - min / max - min
    1 = 1-1 / 6-1 = 0/5 = 0
    2 = 2-1 / 6-1 = 1/5 = 0,2
    3 = 3-1 / 6-1 = 2/5 = 0,4
    4 = 4-1 / 6-1 = 3/5 = 0,6
    5 = 5-1 / 6-1 = 4/5 = 0,8
    6 = 6-1 / 6-1 = 5/5 = 1
Nilai hasil normalisasi ini kemudian digunakan dalam proses perhitungan jarak.

#### Numerik
Untuk atribut numerik seperti math score, reading score, dan writing score, jarak dihitung menggunakan rumus Euclidean distance.
$$
d(1,4) = \sqrt{(0.6-0.6)^2 + (47-72)^2 + (57-72)^2 + (44-74)^2}
$$

$$
= \sqrt{0 + 625 + 225 + 900}
$$

$$
= \sqrt{1750}
$$

$$
= 41.833
$$

Hasil tersebut menunjukkan jarak antara data ke-1 dan data ke-4 berdasarkan atribut numerik yang digunakan.

#### Binary
Pada atribut bertipe binary, perhitungan dilakukan menggunakan dua pendekatan yaitu binary simetris dan binary asimetris.

##### - Binary Simetris
Nilai parameter yang diperoleh adalah:

    d(1,4) --> Simetris   q = 0
                          r = 0
                          s = 1
                          t = 0

Rumus yang digunakan: $\frac{r+s}{q+r+s+t}$

Perhitungannya adalah:  $\frac{0+1}{0+0+1+0} = \frac{1}{1} =1$

##### - Binary Asimetris
Nilai parameter yang diperoleh adalah:

    Asimetris   q = 0
                r = 1
                s = 0
                t = 1

Rumus yang digunakan: $\frac{r+s}{q+r+s}$  

Perhitungannya adalah: $\frac{1+0}{0+1+0} = \frac{1}{1} =1$

Sehingga hasil jarak untuk atribut binary adalah:
      
      d(1,4) = 2

#### Kategorikal

Untuk atribut kategorikal, perhitungan dilakukan menggunakan rumus:

     p-m/p
dengan:
p = jumlah atribut
m = jumlah atribut yang sama

Perhitungannya adalah:

    d(1,4) = 1-0 / 1 
           = 1/1 
           = 1

#### Hasil Akhir Jarak
Setelah semua tipe data dihitung, maka nilai jarak total antara data ke-1 dan data ke-4 diperoleh dengan menjumlahkan seluruh hasil perhitungan sebelumnya.

    d(1,4) = 44,833     3

Jumlah atribut yang digunakan adalah 3, sehingga nilai tersebut merupakan hasil pengukuran jarak antar dua data berdasarkan seluruh atribut yang digunakan.