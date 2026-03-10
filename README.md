# Penambangan Data — Shofiatul Mahmudah (240411100023)

Dokumentasi tugas mata kuliah **Penambangan Data** menggunakan **Jupyter Book** dan di-deploy ke **GitHub Pages**.

---

## Tutorial: Setup Environment sampai Deploy ke GitHub Pages

### 1. Persiapan Repository GitHub

1. Buat repository baru di GitHub (misal: `jokian-pendata-sofi`)
2. Clone repository ke komputer lokal:
   ```bash
   git clone https://github.com/USERNAME/jokian-pendata-sofi.git
   cd jokian-pendata-sofi
   ```

### 2. Membuat Virtual Environment Python

```bash
# Buat virtual environment
py -3 -m venv .venv

# Aktivasi virtual environment (Windows PowerShell)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1

# Aktivasi virtual environment (Command Prompt)
.\.venv\Scripts\activate.bat

# Aktivasi virtual environment (Linux/Mac)
source .venv/bin/activate
```

### 3. Install Jupyter Book dan ghp-import

```bash
pip install "jupyter-book<2.0.0" ghp-import
```

> **Catatan:** Gunakan `jupyter-book<2.0.0` untuk kompatibilitas.

### 4. Membuat Struktur Jupyter Book

```bash
jupyter-book create materi-pendat
```

Struktur folder yang dihasilkan:

```
materi-pendat/
├── _config.yml          # Konfigurasi buku
├── _toc.yml             # Table of Contents
├── intro.md             # Halaman utama
├── Bussiness-Understanding.md
├── Data-Understanding.md
├── Data-Preparation.md
├── Modelling.md
├── Deployment.md
├── Pertemuan3/           # Gambar KNN Imputation
├── Pertemuan3_KNN_Imputation.ipynb
├── DataCampuranPertemuan3/
│   └── Penguins/
│       ├── Penguins.ows
│       └── Penguins.sql
└── logo.png
```

### 5. Konfigurasi `_config.yml`

```yaml
title: Penambangan Data - Shofiatul Mahmudah
author: Shofiatul Mahmudah (240411100023)
logo: logo.png
execute:
  execute_notebooks: 'off'
repository:
  url: https://github.com/USERNAME/jokian-pendata-sofi
  branch: main
html:
  use_issues_button: true
  use_repository_button: true
```

### 6. Konfigurasi `_toc.yml`

```yaml
format: jb-book
root: intro
chapters:
  - file: Bussiness-Understanding
  - file: Data-Understanding
  - file: Data-Preparation
  - file: Modelling
  - file: Deployment
```

### 7. Build Jupyter Book

```bash
jupyter-book build materi-pendat
```

Hasil build ada di folder `materi-pendat/_build/html/`. Buka `index.html` untuk preview lokal.

### 8. Deploy ke GitHub Pages menggunakan ghp-import

```bash
# Pastikan sudah commit dan push semua file ke branch main terlebih dahulu
git add .
git commit -m "Add jupyter book content"
git push origin main

# Deploy ke GitHub Pages
ghp-import -n -p -f materi-pendat/_build/html
```

**Penjelasan parameter ghp-import:**
| Parameter | Fungsi |
|-----------|--------|
| `-n` | Menambahkan file `.nojekyll` (bypass Jekyll processing) |
| `-p` | Push otomatis ke remote `origin` |
| `-f` | Force push ke branch `gh-pages` |

### 9. Aktifkan GitHub Pages

1. Buka repository di GitHub
2. Pergi ke **Settings** → **Pages**
3. Di bagian **Source**, pilih:
   - **Branch:** `gh-pages`
   - **Folder:** `/ (root)`
4. Klik **Save**
5. Tunggu beberapa menit, website akan tersedia di: `https://USERNAME.github.io/jokian-pendata-sofi/`

### 10. Update dan Re-deploy

Setiap kali ada perubahan konten:

```bash
# 1. Edit file markdown
# 2. Build ulang
jupyter-book build materi-pendat

# 3. Deploy ulang
ghp-import -n -p -f materi-pendat/_build/html
```

---

## Struktur Pertemuan

| Pertemuan | Topik | File |
|:---------:|-------|------|
| 1 | Business Understanding | `Bussiness-Understanding.md` |
| 2 | Data Understanding | `Data-Understanding.md` |
| 3 | Data Preparation + KNN Imputation | `Data-Preparation.md` |
| 4 | Modelling | `Modelling.md` |
| 5 | Deployment | `Deployment.md` |

---

**Shofiatul Mahmudah** — 240411100023 — Teknik Informatika
