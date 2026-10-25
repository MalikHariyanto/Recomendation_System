# Rekomendasi wisata Madura — riset dan HKI

Metode utama: **LDA + FastText + BiLSTM multi-output → retrieval BGE-M3**.
Pembeda dari penelitian induk: **FastText menggantikan Word2Vec**.
Notebook Adaptive Fusion BGE/E5 menjadi referensi tambahan.

Implementasi mencakup label turunan dataset, pemilihan cakupan Madura, tiga
skenario classifier, inferensi classifier, dan kode pencarian BGE dengan ulasan
pendukung. **Indeks BGE asli berisi 5.071 ulasan dari 20 destinasi telah dibangun dan diuji
secara fungsional.** Evaluasi awal terbatas pada known-item; deployment publik
belum dilakukan. Lihat docs/HASIL_EKSPERIMEN.md untuk bukti dan batas hasil.

## Kebijakan data

- Rating 1–2 negatif, 3 netral, 4–5 positif.
- Musim mempertahankan Kemarau dan Kemarau & Penghujan dari rekomendasi_cuaca.
- Jenis wisata mempertahankan kategori asli wisata.
- Wilayah mengikuti proposal kolaborasi: Madura. Manifest memuat 20 destinasi
  included dan 2 kandidat pending_location. Workbook asli tetap utuh.
- 5.071 ulasan bersih masuk cakupan penelitian yang ditetapkan.
- Label diberi status derived beserta sumbernya, bukan anotasi manusia.

## Jalankan audit dan aplikasi

Python 3.12, tanpa GPU atau API key untuk audit/classifier CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
WISATA_DATA_PATH=data/koreksi_dataset.xlsx python -m streamlit run app.py
```

Dataset dapat diunggah melalui aplikasi jika tidak ada file lokal. Aplikasi
memisahkan katalog Madura dari seluruh workbook untuk audit. Data unggahan
berada dalam memori sesi, tanpa cache global lintas pengguna.

## Label dan eksperimen data nyata

```bash
python -m pip install -r requirements-model.txt
python -m wisata.labeling data/koreksi_dataset.xlsx --output artifacts/madura_labels.csv
python -m wisata.experiments artifacts/madura_labels.csv --output artifacts/experiments-dataset-v1 --epochs 10
WISATA_DATA_PATH=data/koreksi_dataset.xlsx WISATA_MODEL_DIR=artifacts/experiments-dataset-v1/fusion python -m streamlit run app.py
```

Tiga skenario memakai split yang sama. Cache preprocessing memeriksa seluruh
teks berurutan; output lama hanya digunakan ulang jika hash label dan konfigurasi
cocok. Ringkasan hasil ditulis ke summary.csv; manifest dan test_predictions.csv
disimpan per skenario. Folder output training individual harus baru/kosong.

## Indeks rekomendasi BGE

```bash
python -m pip install -r requirements-retrieval.txt
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DISABLE_XET=1
python -m wisata.retrieval artifacts/madura_labels.csv --classifier artifacts/experiments-dataset-v1/fusion --precision bfloat16 --output artifacts/bge-madura
WISATA_INDEX_DIR=artifacts/bge-madura WISATA_MODEL_DIR=artifacts/experiments-dataset-v1/fusion WISATA_DATA_PATH=data/koreksi_dataset.xlsx python -m streamlit run app.py
```

Sesuaikan HF_HOME dengan lokasi cache yang dapat ditulis pada mesin lain.
Encoder dan revisi di-pin. Query serta ulasan dinormalisasi L2, cosine dihitung
per ulasan, skor destinasi adalah rata-rata lima skor tertinggi. Filter musim
memakai label sumber; filter sentimen memakai prediksi BiLSTM jika classifier
disertakan, atau rating-derived bila tidak. Sumber ditampilkan pada hasil.
Tidak ada fallback model palsu. Checksum metadata/embedding diperiksa saat load.

Bobot resmi berhasil diunduh melalui library Hugging Face dan checksum SHA-256
checkpoint asli telah diperiksa. Indeks di environment ini memakai BF16 pada CPU
dengan dukungan AVX512 BF16; query memakai presisi yang sama dari manifest.
Default CLI adalah float32 untuk kompatibilitas, gunakan flag di atas untuk
mereproduksi presisi indeks ini. Perbandingan 16 teks memberi cosine antar-embedding
FP32/BF16 minimum 0,9982; ini pemeriksaan numerik, bukan bukti kualitas ranking identik.

```bash
python -m wisata.evaluate_index artifacts/bge-madura --output artifacts/bge-madura/known_items.json
```

Cache bobot dan artefak lokal dapat disimpan melalui snapshot cloud. Konfigurasi
install/startup dan kebutuhan domain tersimpan sebagai draft; review/save lalu
publish melalui environment settings untuk menyimpan versi lingkungan.

## Streamlit Community Cloud

Entrypoint app.py, Python 3.12. requirements.txt menyiapkan halaman audit.
Serving classifier/BGE perlu dependensi tambahan dan artefak yang sesuai;
jangan mengklaim deployment rekomendasi siap hanya dari halaman audit.
RAM, cold start, dan latensi encoder besar perlu diukur setelah bobot tersedia.
Training tidak dijalankan dari interaksi pengguna.

## Dokumentasi

- [Penjelasan sederhana](docs/PENJELASAN_SEDERHANA.md)
- [Panduan penggunaan aplikasi](docs/PANDUAN_PENGGUNA.md)
- [Audit sumber dan dataset](docs/AUDIT_RISET.md)
- [Metode dan alur program](docs/METODOLOGI.md)
- [Kebijakan label dan wilayah](docs/ANOTASI.md)
- [Hasil tiga eksperimen data nyata](docs/HASIL_EKSPERIMEN.md)
- [Literatur dan status akses](docs/LITERATUR.md)
- [Persiapan bukti program HKI](docs/PERSIAPAN_HKI.md)

Dataset, proposal, label per ulasan, dan artefak model diabaikan Git.
Daftar pencipta, pemegang hak, dan izin distribusi ditetapkan bersama tim riset.
