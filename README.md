# Rekomendasi wisata sadar musim — riset dan HKI

Implementasi awal berdasarkan proposal asistensi Malik Hariyanto:
**LDA + FastText + BiLSTM multi-output → personalisasi BGE**.
Pembeda terhadap penelitian induk adalah **FastText menggantikan Word2Vec**.

Tersedia: workbench Streamlit untuk audit data, template pelabelan,
pipeline tiga skenario, baseline Word2Vec opsional, dan inferensi classifier lokal.
**Retrieval BGE dan deployment final belum tersedia.** Konvensi label dataset
sedang diklarifikasi; uji sintetis tidak menjadi hasil kualitas model riset.

## Jalankan workbench

Python 3.12; audit data tidak memerlukan GPU, API key, atau basis data.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m streamlit run app.py
```

Unggah workbook di panel kiri, atau gunakan path lokal:

```bash
WISATA_DATA_PATH=data/koreksi_dataset.xlsx python -m streamlit run app.py
python -m wisata.audit data/koreksi_dataset.xlsx --output artifacts/audit_dataset.json
python -m wisata.annotations data/koreksi_dataset.xlsx artifacts/anotasi.csv --limit 200
```

Audit tidak mengubah workbook. Unggahan berada di memori sesi, tanpa cache global
lintas pengguna. Dataset, proposal, dan artefak model diabaikan Git.

## Training dan inferensi classifier

```bash
python -m pip install -r requirements-model.txt
python -m pytest -q
# Isi dan validasi label terlebih dahulu sesuai docs/ANOTASI.md.
python -m wisata.train artifacts/anotasi.csv --scenario fusion --output artifacts/fusion
WISATA_MODEL_DIR=artifacts/fusion python -m streamlit run app.py
```

Dependensi model: PyTorch CPU, Gensim, scikit-learn, Sastrawi.
Training menolak label kosong/tidak approved. Artefak model hanya dimuat dari
path lokal yang diatur operator. Baca metodologi untuk ketiga skenario, definisi
metrik, split, dan keterbatasannya.

## Streamlit Community Cloud

Pilih repository dan branch kontribusi, entrypoint **app.py**, Python **3.12**.
requirements.txt menyiapkan workbench audit. Unggah workbook lalu periksa ringkasan
dan filter katalog. Deployment publik belum dilakukan. Serving classifier memerlukan
dependensi model dan artefak; serving rekomendasi BGE masih tahap lanjutan.
Ukur RAM/startup/latensi sebelum menentukan arsitektur serving akhir.
Jangan melatih model dari interaksi pengguna.

## Dokumentasi

- [Mulai di sini: penjelasan sederhana](docs/PENJELASAN_SEDERHANA.md)
- [Audit bahan dan data](docs/AUDIT_RISET.md)
- [Metode dan alur program](docs/METODOLOGI.md)
- [Format label dan anotasi](docs/ANOTASI.md)
- [Literatur dan status akses](docs/LITERATUR.md)
- [Persiapan bukti program HKI](docs/PERSIAPAN_HKI.md)

Daftar pencipta, pemegang hak, dan izin distribusi karya/data ditetapkan bersama
tim riset. Kontribusi ini tidak menetapkan lisensi atas dataset atau karya tim.
