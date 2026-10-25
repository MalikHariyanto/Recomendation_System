# Hasil eksperimen awal pada dataset pengguna

Ini hasil satu run eksploratif, bukan klaim keunggulan signifikan atau hasil rekomendasi BGE.

## Data dan protokol

- 5.071 ulasan bersih dari 20 destinasi Madura yang masuk manifest; 2 kandidat lokasi masih pending.
- 4.058 ulasan train dan 1.013 test; GroupShuffleSplit menurut teks ternormalisasi, seed 42.
- Ketiga skenario memakai ID train/test identik; encoder, vocabulary dan bank LDA hanya fit pada train.
- Label turunan: sentimen dari rating, musim dari rekomendasi_cuaca, jenis wisata dari wisata.
- Dimensi embedding 64, hidden BiLSTM 32, maksimum 96 token, batch 32, 10 epoch.
- Bank LDA: 5 topik per kelas cuaca, 10 iterasi; FastText skip-gram, n-gram 3–6, 10 epoch.
- Tidak ada tuning berdasarkan test atau pengulangan seed. Python hash seed proses tidak di-pin; identitas bitwise lintas proses tidak diklaim.

## Hasil

| Skenario | Hamming Loss keseluruhan ↓ | Micro F1 musim | Micro F1 jenis | Micro F1 sentimen | Macro F1 sentimen |
| --- | ---: | ---: | ---: | ---: | ---: |
| fasttext | 0.2708 | 0.8026 | 0.5538 | 0.8312 | 0.3893 |
| lda | 0.3366 | 0.7710 | 0.3909 | 0.8282 | 0.3020 |
| fusion | 0.2652 | 0.8065 | 0.5656 | 0.8322 | 0.4017 |

## Pembanding sederhana dan batas hasil

Baseline yang selalu menebak kelas sentimen terbanyak di train menghasilkan Micro F1 0.8282 dan Macro F1 0.3020 pada test.
Micro F1 sentimen gabungan hanya sedikit di atas baseline tersebut. Macro F1 masih rendah; model belum dapat dianggap andal untuk semua kelas sentimen.

Gabungan mendapat Hamming Loss terendah pada run ini. Itu belum membuktikan peningkatan yang stabil atau signifikan. Evaluasi menggunakan label turunan, bukan anotasi manusia. Label cuaca/jenis konstan per destinasi dan split ulasan memungkinkan destinasi yang sama berada di train/test, sehingga hasil bukan bukti generalisasi ke destinasi baru.

## Artefak dan pemeriksaan

Artefak lokal berada di artifacts/experiments-dataset-v1/<skenario>/: classifier.pt, embedding.gensim (bila digunakan), topics.joblib (bila digunakan), manifest.json, dan test_predictions.csv. Label serta artefak tidak dimasukkan Git. Hasil tersimpan juga di summary.csv; majority_baseline.json menyediakan pembanding di atas.

Runner telah dijalankan ulang menggunakan cache dan manifest yang cocok, tanpa melatih ulang. Semua model berhasil dimuat untuk inferensi. Sumber label: dataset SHA-256 84245d7276d9ea14014be983f0a0a0385685c20ae08ff43817fecd5b2329e527.

## Retrieval BGE asli

Indeks selesai dibangun dari seluruh 5.071 ulasan cakupan, tanpa subsampling.
Model BAAI/bge-m3 revisi 5617a9f61b028005a4858fdac845db406aefb181,
normalisasi L2, maksimum 256 token, presisi BF16 pada CPU, batch 16.
Checkpoint asli berhasil diunduh dan SHA-256 dicocokkan dengan objek cache:
b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38.

Classifier gabungan memasok prediksi sentimen untuk filter. Filter cuaca dan
jenis wisata menggunakan metadata sumber. Uji query baru, pembatasan musim
hujan, ketersediaan kandidat, dan pencarian ulang berhasil dilakukan. Query religi
bersama pilihan jenis religi serta hujan menghasilkan Masjid Jamik Sumenep;
ini pemeriksaan fungsi, bukan label relevansi untuk semua permintaan pengguna.

### Known-item terbatas

Satu ulasan dipilih per destinasi (20 query, seed 42); ulasan query dikeluarkan
dari kandidat sebelum meranking. Destinasi asal menjadi satu-satunya item relevan.
Embedding query dipakai ulang dari embedding ulasan yang sama dengan model beku.

| Metrik | Hasil |
| --- | ---: |
| Precision@5 | 0,1300 |
| Recall@5 / Hit Rate@5 | 0,6500 |
| MRR@5 | 0,4708 |

Ini evaluasi in-corpus dengan ulasan lain dari destinasi sama masih di indeks,
bukan evaluasi query kebutuhan pengguna dengan anotasi manusia independen.
Sampel hanya 20 query dan tidak mendukung klaim signifikansi atau keunggulan umum.
Rincian per query tersedia lokal di artifacts/bge-madura/known_items.json.

### Biaya dan batas deployment

Pada pemeriksaan satu proses, query pertama termasuk load encoder memerlukan
sekitar 6,25 detik, query berikutnya sekitar 0,13 detik; ini contoh pengukuran,
bukan benchmark latensi yang stabil. Puncak RSS proses pemeriksaan sekitar
3.769 MiB. Target deployment harus memiliki memori memadai dan diuji kembali.
UI tidak melatih ulang model saat mencari. Deployment publik, riwayat pengguna,
SUS, dan evaluasi rekomendasi dengan human qrels belum dilakukan.
