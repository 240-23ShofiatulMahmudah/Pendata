# Prediksi Kadar $NO_2$ di daerah Sumenep Madura

## Latar Belakang

Pencemaran udara menjadi salah satu permasalahan lingkungan yang semakin meningkat akibat aktivitas industri, transportasi, dan pertumbuhan penduduk. Salah satu polutan yang perlu diperhatikan adalah Nitrogen Dioksida (NO₂), yaitu gas yang dihasilkan dari proses pembakaran bahan bakar fosil seperti kendaraan bermotor dan kegiatan industri. Paparan NO₂ dalam kadar tinggi dapat menyebabkan gangguan pernapasan serta berdampak negatif terhadap lingkungan.
Oleh karena itu, pemantauan dan prediksi kadar NO₂ perlu dilakukan untuk mengetahui kondisi kualitas udara dan mendukung upaya pengendalian pencemaran. Dalam penelitian ini, digunakan data time series harian NO₂ untuk memprediksi kadar NO₂ pada periode berikutnya menggunakan metode KNN Regression.

## A. Pengumpulan Data

Pengumpulan data time series harian kadar $NO_2$ di daerah Sumenep. Pengumpulan data di ambil dari website https://dataspace.copernicus.eu/, sebelumnya buat akun terlebih dahulu di website tersebut.

Pertama, kalian bisa menggunakan Google Collaboration untuk menginstall openeo:

```
pip install openeo
```

Kemudian tuliskan code ini di bawahnya:

```
import openeo
```

```
connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
```

Pada saat menjalankan baris code diatas (connection), nanti akan diminta authentikasi seperti output berikut:

```
Visit (link authentikasi) 📋 to authenticate.
✅ Authorized successfully
Authenticated using device code flow.
```

Setelah mengklik link authentikasi tersebut, kalian bisa login menggunakan akun "copernicus" yang sudah kalian buat sebelumnya.

Berikut code untuk titik koordinat yang akan diambil data $NO_2$

```
aoi = {
    "type": "Polygon",
    "coordinates": [
        [
            [113.5243492, -6.9050051],
            [114.3366706, -6.8563145],
            [114.4541553, -7.3828604],
            [113.6141396, -7.3401378],
            [113.6063073, -7.3362537],
            [113.5243492, -6.9050051]
        ]
    ]
}

s5post = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=["2023-10-01", "2025-10-01"],
    spatial_extent={
        "west": 113.5243492,
        "south": -7.3828604,
        "east": 114.4541553,
        "north": -6.8563145
    },
    bands=["NO2"],
)

# Aggregate per hari
s5p_no2_daily = s5post.aggregate_temporal_period(
    reducer="mean",
    period="day"
)

# Hitung rata-rata NO2 pada AOI
s5p_no2_aoi = s5p_no2_daily.aggregate_spatial(
    reducer="mean",
    geometries=aoi
)
```

Untuk mengambil titik koordinat kalian dapat mengunjungi website ini https://geojson.io/\#map=14.8/-7.04732/112.69463
Di dalam website tersebut kalian bisa memilih daerah mana yang akan diambil datanya dengan cara memberi shape kotak di daerah tersebut.

![original image](https://cdn.mathpix.com/snip/images/hAZYOM6H9qZqInvYR2H6u1JAAdh-vRTSW7VpnDNkLno.original.fullsize.png)

Di panel sebelah kanan terdapat data JSON yang berupa koordinat daerah yang kalian pilih, kalian salin terus sesuaikan dengan code diatas di bagian variabel “aoi” dan spatial_extent.

Kemudian kalian tambahkan baris code dibawah untuk memulai pengambilan data:

```
job = s5post.execute_batch(title="NO2 in Sumenep", outputfile="NO2Sumenep.nc")
```

Tunggu sampai proses pengambilan data selesai, output proses seperti berikut:

```
0:00:00 Job 'j-2606030334024825a6f043f098bb9dd3': send 'start'
0:00:06 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:00:11 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:00:18 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:00:26 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:00:36 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:00:48 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:01:04 Job 'j-2606030334024825a6f043f098bb9dd3': queued (progress 0%)
0:01:23 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:01:47 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:02:17 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:02:54 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:03:41 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:04:40 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:05:40 Job 'j-2606030334024825a6f043f098bb9dd3': running (progress N/A)
0:06:40 Job 'j-2606030334024825a6f043f098bb9dd3': finished (progress 100%)
```

Ketika proses pengambilan data, aktivitas kalian akan terekam di halaman https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.dataspace.copernicus.eu%2Fopeneo%2F1.2, di halaman tersebut terdapat nama dataset dan status pengambilan data.

![original image](https://cdn.mathpix.com/snip/images/PYEpsdFIyS9SpdMoH3m6iCezaxxHcIE5-zXgrcVNlp4.original.fullsize.png)

## B. Preproccessing Data

Setelah pengambilan data selesai, data tersebut dapat diunduh di halaman https://editor.openeo.org/?server=https%3A%2F%2Fopeneo.dataspace.copernicus.eu%2Fopeneo%2F1.2  File akan berbentuk .nc. Kita hanya memerlukankolom date dan NO2 menggunakan code dibawah:

```
import netCDF4

file_path = "NO2Sumenep.nc"
ds = netCDF4.Dataset(file_path)

# Lihat seluruh variabel yang tersedia
print("📦 Variabel dalam file:")
print(ds.variables.keys())
# dict_keys(['t', 'x', 'y', 'crs', 'NO2'])

# Ambil NO2
no2 = ds.variables["NO2"][:]

# Ambil Time
time = ds.variables["t"][:]

# Konversi waktu ke format tanggal jika punya atribut 'units'
try:
    time_units = ds.variables["t"].units
    dates = netCDF4.num2date(time, units=time_units)
except Exception:
    dates = time  # fallback kalau tidak ada units

# Tampilkan struktur data NO2
print(type(no2))
# type <class 'numpy.ma.core.MaskedArray'>

print(len(no2))
# banyaknya data record NO2 725

print(len(no2[0]))
# panjang data perbaris 9

print(len(no2[0][0]))
# panjang perdata 8

print(no2[0][0][0])
# 3.7701793e-05
```

Dari code diatas kita mengetahui bentuk data dari kolom NO2 nya.

Untuk melihat 10 data pertama adalah:

```
print("Contoh data pertama:")
for i in range(0, 10):
    print(no2[i])
```

Data NO₂ yang diperoleh dalam satu hari terdiri dari beberapa nilai pengukuran. Oleh karena itu, dilakukan perhitungan rata-rata sehingga setiap hari hanya memiliki satu nilai data. Meskipun demikian, masih terdapat beberapa data yang tidak tersedia (missing value), sebagaimana terlihat pada output berikut:

    [1.6708310795365833e-05 -- -- -- -1.5141125004447531e-05 -- -- --
    1.2840599083574489e-05 -- 2.1466044927365147e-05 9.259913895220961e-06
    2.5322374312963802e-06 1.3587216926680412e-05 1.7222517101345147e-07
    4.851592620980227e-06 1.5277844795491546e-05 -2.810011665133061e-06]
    
### 1. Penyelesaian Missing Value Menggunakan Interpolasi Linear

Pada tahap ini, dilakukan proses penanganan missing value yang terdapat pada data NO₂. Langkah ini penting untuk memastikan kualitas data sebelum digunakan dalam proses pemodelan dan analisis.

```
import numpy as np
import pandas as pd

# Interpolasi Linear
no2_filled = np.zeros_like(no2)
# Untuk jaga-jaga jika terdapat '--' tidak berubah menjadi 0
no2_filled = no2_filled.filled(0)

# loop tiap grid (y,x)
for i in range(no2.shape[1]):     # 9 baris
    for j in range(no2.shape[2]): # 8 kolom
        series = pd.Series(no2[:, i, j])
        no2_filled[:, i, j] = series.interpolate(method='linear', limit_direction='both').to_numpy()
```

Dengan menggunakan kode di atas, nilai yang hilang (missing value) pada data NO₂ dapat ditangani secara otomatis melalui metode interpolasi linear.

### 2. Menghitung Rata-rata Data dan Mengubah Format Datetime

Setelah nilai yang hilang berhasil ditangani, data NO₂ dirata-ratakan agar setiap record hanya memiliki satu nilai pengamatan. Selanjutnya, data tanggal diambil dan disimpan ke dalam array. Format datetime juga disederhanakan dari 2023-10-04 00:00:00 menjadi 2023-10-04 karena analisis yang dilakukan berfokus pada data harian, sehingga informasi waktu tidak digunakan.

```
new_dates = []
new_no2 = []
for i in range(len(dates)):
    # ubah format datetime
    new_date = dates[i].strftime('%Y-%m-%d')
    new_dates.append(new_date)
    new_no2.append(np.mean(no2_filled[i]))
```


### 3. Simpan data dalam bentuk File CSV

Setelah itu kita akan membentuk data menjadi DataFrame Pandas untuk disimpan menjadi CSV.

```
df = pd.DataFrame({
    "date": dates,
    "NO2": new_no2
})

# Simpan ke CSV
df.to_csv("NO2_Sumenep_timeseries.csv", index=False)
```


### 4. Pengecekan Kelengkapan Data Time Series Harian pada CSV

Setelah data NO₂ berhasil disimpan ke dalam file CSV, dilakukan pemeriksaan terhadap kelengkapan data time series harian. Kode yang digunakan untuk melakukan pengecekan ditunjukkan di bawah ini:

```
import pandas as pd
import numpy as np

df = pd.read_csv("NO2_Sumenep_timeseries.csv")

# Pastikan kolom 'date' bertipe datetime
df['date'] = pd.to_datetime(df['date'])

# Buat rentang tanggal lengkap
start_date = "2023-10-01"
end_date = "2025-09-30"
full_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Cek tanggal yang hilang
missing_dates = full_range.difference(df['date'])

print(f"Jumlah hari missing: {len(missing_dates)}")
print("Daftar tanggal missing:")
print(missing_dates)
```

```
Jumlah hari missing: 6
Daftar tanggal missing:
DatetimeIndex(['2023-11-11', '2024-01-01', '2024-03-11', '2024-03-23',
               '2024-08-12', '2025-01-31'],
              dtype='datetime64[ns]', freq=None)
```

Pada dataset yang digunakan, masih ditemukan 6 hari yang tidak memiliki data (missing value). Kode yang digunakan untuk memperbaiki data tersebut ditunjukkan di bawah ini.

```
import pandas as pd

# Pastikan datetime dan sorting
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Buat rentang tanggal lengkap
full_range = pd.date_range(start="2023-10-01", end="2025-09-30", freq='D')

# Reindex agar tanggal yang hilang muncul sebagai NaN
df = df.set_index('date').reindex(full_range)
df.index.name = 'date'

# Interpolasi linear berdasarkan indeks waktu
df['NO2'] = df['NO2'].interpolate(method='time')

# (Opsional) jika masih ada NaN di bagian awal/akhir bisa gunakan forward/backward fill
df['NO2'] = df['NO2'].fillna(method='bfill').fillna(method='ffill')

# Simpan kembali ke CSV
df.to_csv("no2_timeseries_interpolated.csv")
```

Setelah dilakukan pengecekan kembali terhadap data harian, tidak ditemukan lagi nilai yang hilang (missing value) pada dataset.

```
Jumlah hari missing: 0
Daftar tanggal missing:
DatetimeIndex([], dtype='datetime64[ns]', freq='D')
```


### 5. Deteksi Outlier IQR

Setelah proses pengisian missing value menggunakan metode interpolasi linear selesai dilakukan, tahap berikutnya adalah mendeteksi outlier pada data menggunakan metode Interquartile Range (IQR).

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("no2_timeseries_interpolated.csv")

df['date'] = pd.to_datetime(df['date'])

# Hitung IQR
Q1 = df['NO2'].quantile(0.25)
Q3 = df['NO2'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outlier
outliers_iqr = df[(df['NO2'] < lower_bound) | (df['NO2'] > upper_bound)]

print("Jumlah Outlier (IQR):", len(outliers_iqr))
print(outliers_iqr[['date', 'NO2']].head())
```

Output/terminal:

```
Jumlah Outlier (IQR): 7
          date       NO2
74  2023-12-14  0.000025
75  2023-12-15  0.000027
77  2023-12-17  0.000027
78  2023-12-18  0.000026
103 2024-01-12  0.000031
```

Untuk men-visualisasi outlier:

```
# === Visualisasi ===
plt.figure(figsize=(15,5))
plt.plot(df['date'], df['NO2'], label="NO2", linewidth=1)

# Titik Outlier
plt.scatter(outliers_iqr['date'], outliers_iqr['NO2'], 
            color='red', marker='o', label="Outliers")

# Garis batas atas & bawah
plt.axhline(upper_bound, color='orange', linestyle='dashed', label="Upper Bound (IQR)")
plt.axhline(lower_bound, color='blue', linestyle='dashed', label="Lower Bound (IQR)")

plt.title("Deteksi Outlier Data NO2 (Metode IQR)")
plt.xlabel("Tanggal")
plt.ylabel("Kadar NO2")
plt.legend()
plt.tight_layout()
plt.xticks(
    ticks=[df['date'].iloc[0], df['date'].iloc[-1]],
    labels=[df['date'].iloc[0].strftime('%Y-%m-%d'),
            df['date'].iloc[-1].strftime('%Y-%m-%d')]
)
plt.show()
```

![original image](https://cdn.mathpix.com/snip/images/AyA0fbNIhaI36uD14hD19Dw_Gl1oQAd-QghyBVB8nqE.original.fullsize.png)

Tahap selanjutnya adalah menghilangkan data yang terdeteksi sebagai outlier. Mengingat dataset berbentuk time series, nilai yang dihapus akan digantikan dengan hasil interpolasi linear sehingga tidak menimbulkan kekosongan data pada rentang waktu pengamatan.

```
# Tandai outlier menjadi NaN
df['NO2_cleaned'] = df['NO2'].mask((df['NO2'] < lower_bound) | (df['NO2'] > upper_bound))

print("Jumlah nilai yang dinyatakan sebagai outlier:", df['NO2_cleaned'].isna().sum())

# Interpolasi linear untuk mengisi kembali nilai outlier
df['NO2_filled'] = df['NO2_cleaned'].interpolate(method='linear')

# Jika masih tersisa NaN di ujung data, isi dengan forward/backward fill
df['NO2_filled'] = df['NO2_filled'].bfill().ffill()
# df['NO2_filled'] = df['NO2_filled'].fillna(method='bfill').fillna(method='ffill')

print("Jumlah missing setelah interpolasi:", df['NO2_filled'].isna().sum())
```

Visualisasi data setelah menghapus Outlier dan mengisi kembali menggunakan Interpolasi Linear:

```
plt.figure(figsize=(15,5))
# Plot data hasil interpolasi
plt.plot(df['date'], df['NO2_filled'], label="NO2 (Interpolated)", linewidth=1)
# Tampilkan hanya tanggal awal dan akhir di sumbu X
plt.xticks(
    ticks=[df['date'].iloc[0], df['date'].iloc[-1]],
    labels=[df['date'].iloc[0].strftime('%Y-%m-%d'),
            df['date'].iloc[-1].strftime('%Y-%m-%d')]
)
plt.title("Plot Data NO2 Setelah Outlier Removal & Interpolasi")
plt.xlabel("Tanggal")
plt.ylabel("Kadar NO2")
plt.legend()
plt.tight_layout()
plt.show()
```

![original image](https://cdn.mathpix.com/snip/images/dtYVI3WsvrTMAqFy1oGiQlt9K3zHMYKarqUPO1eV32c.original.fullsize.png)

## C. Modelling menggunakan metode KNN Regression

Pada tahap ini, data time series harian kadar NO₂ di Kabupaten Sumenep digunakan untuk memprediksi kadar NO₂ pada hari berikutnya. Data akan diubah terlebih dahulu sehingga nilai NO₂ pada beberapa hari sebelumnya dapat digunakan sebagai fitur prediksi. Selanjutnya, akan dianalisis korelasi antara nilai NO₂ saat ini dengan nilai pada hari-hari sebelumnya, serta dibandingkan pengaruh jumlah hari sebelumnya (lag) terhadap kinerja model KNN Regression.

### 1. Uji Korelasi Data

Sebelum melakukan pemodelan, data time series terlebih dahulu diubah menjadi bentuk supervised learning. Pada tahap ini, nilai NO₂ dari 30 hari sebelumnya (t-30 hingga t-1) digunakan sebagai fitur, sedangkan nilai pada hari ke-t dijadikan sebagai label. Selanjutnya, dilakukan uji korelasi untuk melihat hubungan antara setiap fitur dengan label yang akan diprediksi.

```
# MODELLING
import pandas as pd

def create_supervised(data, n_lag=4):
    df_supervised = pd.DataFrame()
    
    # Membuat fitur t-4 sampai t-1
    for i in range(n_lag, 0, -1):
        df_supervised[f'NO2(t-{i})'] = data.shift(i)
    
    # Label hari H
    df_supervised['NO2(t)'] = data
    
    # Hapus baris yang masih mengandung NaN akibat shift
    df_supervised.dropna(inplace=True)
    
    return df_supervised

# contoh penggunaan
supervised_df30 = create_supervised(df['NO2_filled'], n_lag=30)

# Ambil semua lag dan kolom target
lag_cols = supervised_df30.drop(columns="NO2(t)").columns
correlations = supervised_df30[lag_cols].corrwith(supervised_df30['NO2(t)'])

# Tampilkan nilai korelasi
print(correlations)
```

Output/Terminal:

```
NO2(t-30)    0.370712
NO2(t-29)    0.366606
NO2(t-28)    0.344649
NO2(t-27)    0.348135
NO2(t-26)    0.304259
NO2(t-25)    0.325997
NO2(t-24)    0.317188
NO2(t-23)    0.314006
NO2(t-22)    0.295512
NO2(t-21)    0.314576
NO2(t-20)    0.322678
NO2(t-19)    0.315124
NO2(t-18)    0.330163
NO2(t-17)    0.330909
NO2(t-16)    0.362243
NO2(t-15)    0.316075
NO2(t-14)    0.326677
NO2(t-13)    0.355730
NO2(t-12)    0.389623
NO2(t-11)    0.426148
NO2(t-10)    0.426888
NO2(t-9)     0.458736
NO2(t-8)     0.460564
NO2(t-7)     0.465242
NO2(t-6)     0.456107
NO2(t-5)     0.507097
NO2(t-4)     0.535467
NO2(t-3)     0.612302
NO2(t-2)     0.682434
NO2(t-1)     0.778969
dtype: float64
```

Nilai uji korelasi berkisar antara -1 hingga 1. Fitur yang memiliki nilai korelasi di atas 0,5 dianggap memiliki hubungan yang cukup kuat dengan target. Oleh karena itu, dipilih fitur t-1 hingga t-5 untuk digunakan dalam pemodelan.

