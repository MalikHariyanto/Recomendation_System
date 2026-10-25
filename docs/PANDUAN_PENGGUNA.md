# Panduan penggunaan aplikasi

Panduan ini untuk aplikasi yang sudah dikonfigurasi dengan dataset, classifier,
dan indeks BGE. Menjalankan halaman audit saja tidak otomatis menyiapkan model.
Instruksi instalasi operator ada di README.

## Mencari destinasi

1. Buka aplikasi dan temukan bagian **Cari destinasi wisata**.
2. Tulis kebutuhan, misalnya “tempat tenang dengan pemandangan indah untuk keluarga”.
3. Pilih rencana musim: Semua, Hujan, atau Kemarau.
4. Pilih jenis wisata jika ingin membatasi ke kategori tertentu.
5. Pilih ulasan yang dipertimbangkan: Semua, positif, netral, atau negatif.
6. Tekan **Cari rekomendasi** dan baca destinasi, skor kecocokan, serta ulasan bukti.

Skor berasal dari kemiripan teks; bukan persentase kepastian atau jaminan mutu
tempat. Rating hasil adalah rata-rata ulasan kandidat setelah filter. Ulasan
bukti merupakan kutipan pendukung, bukan pembahasan fakta yang diverifikasi.

Pilihan Hujan mempertahankan destinasi berlabel Kemarau & Penghujan pada data.
Label ini statis, bukan prakiraan cuaca hari perjalanan. Pilihan sentimen memakai
prediksi BiLSTM jika classifier disertakan dalam indeks; sumber sentimen tampil
di hasil. Jika tidak ada kandidat yang sesuai, aplikasi meminta pengguna mencoba
pilihan lain dan tidak mengisi daftar dengan destinasi yang melanggar filter.

## Memeriksa data

Unggah koreksi_dataset.xlsx di panel kiri jika dataset lokal belum disediakan.
Empat ringkasan menunjukkan jumlah seluruh workbook sumber. Di bawahnya ada
ringkasan khusus cakupan penelitian Madura.

- **Katalog destinasi:** beralih antara Madura (penelitian) dan seluruh dataset
  (audit), lalu pilih kategori dan kecocokan musim. Katalog berurutan abjad,
  terpisah dari ranking rekomendasi.
- **Label penelitian:** lihat aturan rating, label cuaca, jenis wisata, dan
  sebaran kelas; unduh label turunan beserta sumbernya bila diperlukan.
- **Kualitas data:** periksa ulasan pendek, duplikat, distribusi kategori dan
  musim; unduh laporan audit JSON.
- **Alur riset:** baca ringkasan metode dan pembeda FastText dari Word2Vec.

## Menguji classifier ulasan

Buka **Klasifikasi ulasan dari model lokal**, tulis ulasan, lalu tekan
**Klasifikasikan ulasan**. Tiga hasil adalah prediksi musim, jenis wisata,
dan sentimen. Hasil dapat keliru, terutama pada kelas minoritas. Target musim
tetap mengikuti label cuaca dataset, bukan penentuan tanggal kunjungan.

## Bahan dokumentasi HKI

Setelah deployment publik selesai, ambil screenshot nyata untuk halaman pencarian,
hasil dan ulasan bukti, classifier, katalog, serta kualitas/label data. Beri setiap
screenshot keterangan fungsi. Jangan menggunakan contoh buatan sebagai bukti
bahwa suatu fitur atau deployment sudah berjalan.
