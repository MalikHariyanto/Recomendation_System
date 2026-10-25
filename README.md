# Malik — Adaptive Fusion BGE-M3 + E5

Tugas aktif Malik adalah **Adaptive Fusion**, menggunakan BGE-M3 sebagai baseline
serta E5 sebagai komponen/pembanding. Notebook lengkap menghitung ketiganya dengan
korpus, query, dan qrels yang sama. Jalur classifier proposal sebelumnya tetap
tersedia sebagai [arsip](docs/ARSIP_LDA_FASTTEXT.md), bukan metode tugas ini.

## Mulai dari Colab

Buka [notebook Malik](notebooks/Malik_AdaptiveFusion_BGE_E5_Colab.ipynb), pilih
**T4 GPU**, simpan dataset di **MyDrive/Rekomendasi_Wisata/koreksi_dataset.xlsx**,
lalu **Runtime → Run all** dan izinkan akses Drive.

Tidak perlu setup Python atau Git Bash di laptop. Lihat
[panduan tugas Malik](docs/TUGAS_MALIK_FUSION.md) untuk langkah dan lokasi hasil.

## Hasil yang disimpan

- PDF otomatis setelah setiap tahap, termasuk tahap gagal.
- Tabel CSV lengkap, grafik, log, dan ringkasan JSON.
- Perbandingan BGE, E5, fusion, baseline statis/RRF, ablasi, serta known-item.
- Bundle model terpilih dari validasi untuk Streamlit; oracle tidak boleh dipilih.

Setiap run memiliki folder tersendiri. Model/revisi dan cache diperiksa; gate serta
preprocessing diverifikasi setelah ekspor. PDF mengandung ringkasan tabel, sementara
nilai lengkap disimpan di CSV. Hasil mock diberi watermark dan dilarang untuk
klaim penelitian maupun deployment.

## Streamlit hasil terpilih

Gunakan **deploy/fusion/streamlit_app.py** untuk entrypoint Streamlit Cloud dan
requirements di folder yang sama. Pasang bundle hasil Colab pada server dan atur
**FUSION_BUNDLE_DIR** ke folder hasil ekstraknya. Untuk lokal, gunakan fusion_app.py.
Aplikasi memilih encoder/model menurut manifest hasil validasi, bukan memaksakan
fusion menang. Model besar dan embedding tidak dimasukkan Git.

Notebook sudah diuji secara teknis; encoder E5 asli berhasil melewati preflight
CPU. **Run penelitian penuh di sesi Colab/GPU pengguna dan deployment publik belum
terjadi.** Tidak ada pemenang penelitian yang diklaim dari mock.

## Pengembangan

```bash
python -m pip install -r requirements-model.txt -r requirements-fusion-dev.txt
python scripts/build_fusion_notebook.py
python -m pytest -q
```

Sumber notebook pengguna disimpan di notebooks/source untuk atribusi dan audit;
notebook yang dijalankan adalah versi Malik di atas. Workbook, model, laporan
hasil, dan keluaran notebook tetap berada di Drive/artifacts, tidak di-commit.
