# Konvensi label dan format anotasi

Workbook menyediakan rating, date, rekomendasi_cuaca, dan wisata. Kolom itu
merupakan bahan pelabelan, namun arti setiap target perlu dibuat eksplisit.

| Data sumber | Pemakaian yang mungkin | Batas interpretasi |
| --- | --- | --- |
| rating | Proksi sentimen: 1–2 negatif, 3 netral, 4–5 positif | Harus disepakati dan dilaporkan sebagai rating-derived, bukan anotasi teks manusia |
| date | Waktu publikasi/edit ulasan | Nilai seperti 1 hari lalu memerlukan tanggal scraping; belum tentu tanggal kunjungan |
| rekomendasi_cuaca | Kecocokan cuaca destinasi | Dua nilai Kemarau dan Kemarau & Penghujan; bukan empat kelas musim/liburan dalam proposal |
| wisata | Karakteristik destinasi | Empat belas kategori sumber perlu pemetaan eksplisit ke kategori target proposal |

File unggahan ulang memiliki SHA-256 identik dengan unggahan pertama:
84245d7276d9ea14014be983f0a0a0385685c20ae08ff43817fecd5b2329e527.
Satu sheet: Data Ulasan. Tidak ada sheet anotasi tersembunyi tambahan yang ditemukan.
Persetujuan konvensi label dan tanggal scraping masih ditanyakan; pipeline belum
mengonversi target secara diam-diam.

## Template

Jalankan python -m wisata.annotations. Template mempertahankan source_rating,
source_date, dan source_weather_suitability untuk pemeriksaan, dengan target kosong.

| Kolom | Isi |
| --- | --- |
| review_id | SHA-256 ID destinasi dan teks bersih, unik |
| text / destinasi | Ulasan dan konteks destinasi |
| source_date / source_rating / source_weather_suitability | Bahan pelabelan asli |
| season_label | kemarau / hujan / liburan_sekolah / libur_nasional |
| tourism_label | sejarah / alam / budaya / religi / lainnya |
| sentiment_label | positif / negatif / netral |
| region | madura untuk data utama, wilayah lain ditulis eksplisit |
| annotation_status | approved setelah label diverifikasi |
| annotator_code / notes | Kode penilai dan alasan/adjudikasi |

Cuaca dan liburan dapat beririsan. Target single-label mengikuti tabel proposal,
tetapi kebijakan untuk kasus beririsan perlu disepakati. Jangan memilih label
sembarang untuk memenuhi format. Jika keputusan akhirnya multilabel atau memakai
label kecocokan cuaca, perubahan ruang lingkup harus dicatat dan model disesuaikan.

Lokasi Madura perlu diverifikasi; workbook Jawa Timur tidak punya kolom wilayah.
Semua 207 destinasi tidak otomatis termasuk ruang lingkup Madura.
