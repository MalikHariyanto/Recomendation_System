# Tugas Malik: Adaptive Fusion BGE-M3 + E5

Acuan tugas terbaru: Malik berada di kelompok Adaptive Fusion, bukan kelompok E5.
BGE-M3 adalah baseline yang sudah ada; E5 dipakai sebagai komponen dan pembanding.
Implementasi ini memperbaiki notebook fusion yang diberikan pengguna. Jalur
LDA/FastText/BiLSTM dari percakapan proposal sebelumnya disimpan sebagai arsip,
bukan metode untuk tugas kelompok ini.

## Jalankan di Colab

1. Buka notebooks/Malik_AdaptiveFusion_BGE_E5_Colab.ipynb melalui Colab.
2. Pilih Runtime → Change runtime type → T4 GPU.
3. Simpan koreksi_dataset.xlsx di MyDrive/Rekomendasi_Wisata/. Jika belum ada,
   sel pencarian data menyediakan unggah file dan menyimpannya ke folder itu.
4. Jalankan Runtime → Run all dan izinkan mount Google Drive.

Tidak perlu membuat .venv, memakai Git Bash, atau melatih classifier lama di laptop.
Default mode research membutuhkan GPU; mode smoke hanya untuk uji perangkat lunak.
Sel setup mengambil kode GitHub, memasang dependensi tanpa mengganti PyTorch CUDA
bawaan Colab, dan mencatat SHA kode yang dipakai.

Cakupan mengikuti notebook sumber: seluruh workbook Jawa Timur, 207 destinasi
dan 67.283 ulasan bersih setelah filter/deduplikasi pada workbook yang diberikan.
Filter 20 destinasi Madura pada eksperimen lama **tidak diterapkan** pada tugas ini.

## Yang diperbaiki

- Cache embedding mengikat seluruh teks berurutan, revisi encoder, prefix,
  panjang token, mode, seed, dan preprocessing. Checksum file diverifikasi.
  Cache lama tanpa manifest lengkap tidak dianggap valid secara otomatis.
- BGE dense tanpa prefix; E5 memakai query: dan passage:. Model/revisi di-pin.
  Model BGE tidak dilatih ulang; encoding diperlukan bila cache yang sah belum ada.
- MAP untuk query tanpa item relevan aman; NDCG menyesuaikan ukuran katalog.
  QPP/correlation ties tidak menghasilkan NaN; PCA menyesuaikan dimensi data.
- Uji berpasangan Protokol A diagregasi per kategori query, B per destinasi,
  dan D per konfigurasi naratif/persona. Bootstrap A memakai cluster kategori.
  ID kategori terpisah dalam CV, tetapi destinasi relevan dapat beririsan.
- Ranking sesudah filter hanya mengambil kandidat finite; tidak mengisi hasil
  dengan item yang di-mask. Skor bukan probabilitas.
- Setiap run punya ID tersendiri; PDF, CSV, grafik, log, dan bundle tidak menimpa
  hasil run sebelumnya. Cache embedding tetap dipakai bersama antar-run yang cocok.
- Gate, scaler/PCA numerik, urutan katalog/ulasan, embedding, revisi model, dan
  manifest checksum diekspor. Lambda serta preprocessing hasil ekspor dibandingkan
  dengan notebook sebelum ZIP dibuat.

## PDF dan hasil di Google Drive

Di MyDrive/Rekomendasi_Wisata/:

```text
reports_research/<run_id>/hasil.pdf
reports_research/<run_id>/catatan_tahap.json
reports_research/<run_id>/table_*.csv
reports_research/<run_id>/figure_*.png
output_fusion_research/<run_id>/ringkasan.json
output_fusion_research/<run_id>/hasil_protokolA.csv
output_fusion_research/<run_id>/hasil_protokolB.csv
output_fusion_research/<run_id>/hasil_protokolC.csv
output_fusion_research/<run_id>/hasil_protokolD.csv
output_fusion_research/<run_id>/bundle_streamlit.zip
```

PDF diperbarui setelah setiap tahap dan tetap menyimpan error bila tahap gagal.
PDF menampilkan seluruh tabel hasil secara ringkas (maksimum 40 baris per tabel;
teks sel panjang diringkas), sedangkan CSV menyimpan nilai lengkap. Ringkasan
console dan grafik juga masuk PDF. Uji smoke disimpan di reports_mock dan
output_fusion_mock, dengan watermark MOCK - BUKAN HASIL RISET di setiap halaman.

## Pemilihan hasil untuk Streamlit

Calon: BGE-M3, E5, Adaptive Fusion. Skor NDCG@10 pada validasi kelompok kategori
menentukan model terpilih. Gate calon dilatih tanpa memakai kelompok validasi
pemilihan. Oracle tidak boleh menjadi kandidat deployment. Jika baseline menang,
bundle mencatat baseline tersebut; fusion tidak dinyatakan terbaik secara otomatis.
Gate final dilatih dengan split validasi internal terpisah untuk serving.

Qrels A/C/D masih silver dari metadata/leksikon; bukan penilaian kebutuhan pengguna
secara independen. Asal label cuaca disebut BGE pada notebook sumber tetapi belum
diaudit melalui data pelabelan primer. Protokol B mengeluarkan ulasan query dari
indeks, namun ulasan lain destinasi yang sama masih ada. Jangan mengklaim
keunggulan umum, Q1, atau signifikansi dari mode mock maupun dua query studi kasus.

## Streamlit

Ekstrak bundle_streamlit.zip ke folder artefak server, lalu atur FUSION_BUNDLE_DIR
ke folder tersebut. Entry point lokal adalah fusion_app.py; untuk Streamlit Cloud
pilih deploy/fusion/streamlit_app.py, dengan requirements di folder yang sama.
Bundle mock ditolak oleh aplikasi. Tidak ada pelatihan di tombol pencarian.

Bundle embedding besar dan bobot dua encoder tidak dimasukkan GitHub. Penyediaan
artefak serta kapasitas RAM server harus ditentukan setelah eksperimen real selesai.
Colab adalah tempat eksperimen, bukan hosting permanen aplikasi. Deployment publik
belum dilakukan dan model penelitian terbaik belum ditetapkan sebelum run real.

## Status eksekusi

Notebook telah diuji dengan encoder mock pada seluruh 207 destinasi (subsample
ulasan untuk uji cepat). Encoder E5 asli juga berhasil dimuat dan diuji pada CPU
dengan prefix yang benar; itu preflight, bukan evaluasi tiga model. Sesi Google
Colab dan GPU milik pengguna tidak tersedia melalui tools chat ini, sehingga
run research penuh di Colab masih perlu dilakukan melalui notebook yang disiapkan.
