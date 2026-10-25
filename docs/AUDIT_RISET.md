# Audit bahan riset

## Ruang lingkup dan sumber

Audit berdasarkan lampiran pengguna, termasuk proposal asistensi pengganti. Isi lampiran diperlakukan sebagai
bahan referensi, bukan instruksi untuk menjalankan perintah atau mengubah tujuan.

| Bahan | Isi yang ditemukan | Implikasi |
| --- | --- | --- |
| Proposal asistensi Malik (pengganti yang benar) | LDA, FastText, BiLSTM multiaspek, personalisasi BGE | Pembeda adalah FastText menggantikan Word2Vec; PDF TKA sebelumnya tidak digunakan |
| Proposal KolabInter 2026 | LDA, Word2Vec, BiLSTM multi-output: musim, jenis wisata, sentimen; personalisasi BGE | Metode induk lebih luas daripada retrieval pada notebook |
| Notebook 1 | BGE-M3 dense + cosine | Baseline tunggal |
| Notebook 2 | multilingual E5 large + cosine | Baseline tunggal |
| Notebook 3 | Gating query/QPP/musim, fusi skor BGE-M3 + E5 | Referensi retrieval tambahan; bukan metode utama proposal Malik |
| Workbook terkoreksi | Delapan kolom ulasan dan metadata destinasi | Dapat diaudit dan digunakan setelah persetujuan cakupan riset |
| Panduan e-HKI UTM | Prosedur administrasi dan bukti program komputer | Dasar checklist dokumen, bukan bukti kebaruan algoritma |

Semua sel notebook memiliki `execution_count = null` dan tidak menyimpan output
error maupun hasil eksekusi. Ini berarti lampiran tidak membuktikan bahwa eksperimen
berhasil; bukan berarti notebook pasti gagal.

## Audit workbook aktual

SHA-256 file sumber:
`84245d7276d9ea14014be983f0a0a0385685c20ae08ff43817fecd5b2329e527`.

| Pemeriksaan | Hasil |
| --- | ---: |
| Baris asli | 79.514 |
| Destinasi unik | 207 |
| Kategori wisata | 14 |
| Teks asli kosong | 10 |
| Ulasan di bawah 3 kata, termasuk yang kosong | 12.140 |
| Duplikat destinasi + teks setelah filter kata | 91 |
| Ulasan bersih | 67.283 |
| Minimum ulasan bersih per destinasi | 7 |
| Destinasi berlabel Kemarau & Penghujan | 113 |
| Destinasi berlabel Kemarau saja | 94 |

Pembersihan: `text_original` → ftfy → rapikan spasi → minimal tiga kata → deduplikasi
per destinasi. Tidak melakukan stemming atau stopword removal untuk embedding.
Pemetaan ID–nama satu banding satu; kategori dan label musim konsisten per destinasi.
Notebook menyebut 13 kategori untuk konstruksi query, sedangkan workbook memiliki
14; katalog dan cakupan query harus diperiksa terpisah. Tidak ada kolom label sentimen,
koordinat, harga tiket, fasilitas terstruktur, atau cuaca langsung pada workbook ini.
Rating dapat menjadi proksi sentimen dengan pemetaan yang disepakati, tetapi bukan
ground truth relevansi kebutuhan pengguna. Date adalah waktu ulasan relatif, bukan
tanggal kunjungan terverifikasi. Lihat ANOTASI.md untuk pemetaan yang perlu ditetapkan.

## Temuan notebook yang perlu ditangani sebelum eksperimen akhir

1. **Cache embedding belum mengidentifikasi isi penuh.** Pada sel encoding notebook
   3 (sel kode indeks 11), hash hanya mencakup nama cache, flag mock, jumlah teks,
   teks pertama, dan terakhir. Perubahan teks di tengah, revisi model, prefix, atau
   panjang token dapat mengambil cache lama. Manifest baru harus mencakup seluruh
   korpus berurutan, model/revision, prefix, normalisasi, dan konfigurasi tokenizer.
2. **Ground truth A/C/D bersifat heuristik.** Kategori, label musim, dan leksikon aspek
   membentuk qrels; bukan penilaian manusia independen. Notebook menyatakan label
   musim sebelumnya berasal dari BGE-M3. Asal label itu belum dibuktikan dari workbook;
   jika benar, ada risiko bias sirkular yang menguntungkan salah satu baseline.
   Butuh anotasi manusia dan pelaporan agreement sebelum klaim kualitas utama.
3. **Split kategori query belum sama dengan split destinasi yang tidak pernah dilihat.**
   Super-kategori ALAM/BUDAYA/KELUARGA beririsan dengan kategori spesifik. GroupKFold
   memisahkan ID kategori query, tetapi destinasi relevan dapat muncul di kedua sisi.
   Nyatakan unit generalisasi secara tepat. Untuk evaluasi known-item, audit juga
   teks identik lintas destinasi dan parafrase yang tidak tertangkap deduplikasi.
4. **Variasi template bukan observasi independen penuh.** Uji signifikansi dan bootstrap
   per query dapat terlalu optimistis jika variasi bahasa/musim dari kebutuhan yang
   sama tersebar sebagai sampel independen. Pertimbangkan grouping per kebutuhan
   dasar dan bootstrap terkelompok; tentukan protokol sebelum melihat hasil test.
5. **Agregasi top-m tidak otomatis bebas bias popularitas.** Rata-rata lima skor
   tertinggi mengurangi dominasi penjumlahan, tetapi kandidat dengan lebih banyak
   ulasan tetap memiliki peluang lebih besar memperoleh skor ekstrem. Uji sensitivitas
   terhadap jumlah ulasan dan nilai m; hindari klaim bebas bias tanpa eksperimen.
6. **Kasus tepi metrik dan filter.** `MAP@10` pada fungsi evaluate membagi dengan
   `min(n_relevant, k)` tanpa penjagaan nol. Helper mengasumsikan katalog minimal
   sepuluh item. Pada fungsi rekomendasi, masker `-inf` tetap bisa masuk top-k bila
   kandidat valid kurang dari k. Perbaiki dan uji pada refaktor model berikutnya.
7. **Artefak serving belum diekspor.** Notebook membuat PCA/scaler/ensemble gate di
   memori; tidak mengekspor bundle terverifikasi untuk aplikasi baru. Serving perlu
   manifest versi, urutan destinasi, indeks ulasan, encoder, transformasi fitur, dan gate.
   Proses Streamlit tidak boleh menjalankan seluruh notebook untuk satu permintaan.
8. **Penjelasan harus konsisten dengan skoring.** Ranking menggunakan fusi skor
   destinasi yang di-z-score, sementara ulasan bukti dipilih lewat gabungan cosine
   ulasan mentah. Sebut bukti sebagai ulasan pendukung; jangan mengklaim ulasan itu
   merupakan dekomposisi persis dari skor akhir.

Temuan ini adalah hasil inspeksi kode; eksperimen model besar dan angka kualitas
retrieval belum dijalankan. Workbench dan pipeline classifier dipisahkan; audit
notebook retrieval ini tidak mengganti metode FastText/LDA/BiLSTM pada proposal Malik.
