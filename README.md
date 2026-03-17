# Penambangan Data - A
Repository ini dibuat untuk Mata Kuliah Penambangan Data Kelas A Semester 4, Program Studi Teknik Informatika.

Nama: ShofiatulMahmudah
NIM: 24041100023

## Panduan Cepat Jupyter Book (materi-pendat)

### 1. Persiapan Environment (PowerShell)
```powershell
# Dari folder Pendata
..\.venv\Scripts\Activate.ps1

# Jika terkendala execution policy
# Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
```

### 2. Instalasi Dependensi
```powershell
pip install "jupyter-book==1.0.0" ghp-import
```
**Catatan:**
- `jupyter-book==1.0.0` dipakai agar versi Jupyter Book konsisten untuk mata kuliah ini dan tetap mendukung alur `jupyter-book create`.
- `ghp-import` dipakai untuk deploy GitHub Pages dari hasil build HTML.

### 3. Membuat Struktur Buku Baru
```powershell
jupyter-book create materi-pendat
```

### 4. Menambahkan Halaman Baru
Buat file markdown, misalnya `materi-pendat/logika.md`.
Daftarkan ke daftar isi di `materi-pendat/_toc.yml`.
Contoh:
```yaml
format: jb-book
root: intro
chapters:
  - file: pertemuan1
  - file: Penambangan_Data_A_Pertemuan_2
  - file: pertemuan3
  - file: logika
```

### 5. Build Website
```powershell
jupyter-book build materi-pendat
```
Hasil website ada di: `materi-pendat/_build/html/index.html`

## Panduan Git (Push ke origin)

### 1. Cek perubahan
```powershell
git status
```

### 2. Stage semua perubahan
```powershell
git add .
```

### 3. Commit
```powershell
git commit -m "update materi pertemuan 3"
```

### 4. Push ke branch utama
```powershell
git push origin main
```

## Deploy GitHub Pages
Setelah build berhasil, deploy folder HTML ke branch `gh-pages`:
```powershell
ghp-import -n -p -f materi-pendat/_build/html
```
Keterangan opsi:
- `-n`: membuat file `.nojekyll`
- `-p`: langsung push ke remote
- `-f`: force overwrite branch `gh-pages`

URL situs biasanya: `https://username-kamu.github.io/nama-repo-kamu/`
