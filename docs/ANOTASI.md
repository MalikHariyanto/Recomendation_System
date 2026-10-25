# Kebijakan label yang digunakan

Keputusan pengguna: jalankan sesuai dataset, sentimen memakai rating,
musim memakai rekomendasi_cuaca, dan wilayah mengikuti proposal kolaborasi.

| Target | Sumber | Aturan |
| --- | --- | --- |
| sentiment_label | rating | 1–2 negatif, 3 netral, 4–5 positif |
| season_label | rekomendasi_cuaca | Pertahankan Kemarau dan Kemarau & Penghujan sebagai dua kelas kategorikal |
| tourism_label | wisata | Pertahankan kategori asli, tanpa menggabungkan ke lima kelompok buatan |
| region | manifest wilayah | madura untuk identitas destinasi yang masuk cakupan |

Kelas gabungan Kemarau & Penghujan tidak dipecah menjadi dua observasi atau
diubah menjadi label hujan saja. Tidak membuat kelas liburan sekolah/nasional
dan tidak menghitung musim dari date. Dalam penulisan, sebut **label musim
berbasis rekomendasi_cuaca**, bukan tanggal kunjungan yang sudah diverifikasi.

## Provenance dan status

Label yang dihasilkan otomatis memakai annotation_status=derived,
sentiment_source=rating, season_source=rekomendasi_cuaca, tourism_source=wisata,
dan label_policy=dataset-labels-v1. Status derived berbeda dari approved yang
dipakai untuk anotasi manusia. Validator memeriksa provenance dan kategori label.

Kolom sumber source_rating, source_date, source_weather_suitability tetap
dipertahankan. Label bukan kebenaran sentimen teks yang diperiksa manusia:
rating tinggi dapat berdampingan dengan keluhan dalam teks.

## Wilayah Madura

Proposal KolabInter menyebut Bangkalan, Sampang, Pamekasan, dan Sumenep.
config/madura_destinations.csv berisi pemetaan ID sekaligus nama sumber, kabupaten,
status, dan dasar penetapannya. Pemetaan memakai identitas destinasi dan konteks
ulasan; dua destinasi disebut langsung dalam proposal. Ini kurasi manual, bukan
hasil geocoding atau verifikasi lokasi resmi eksternal.

Saat ini 20 destinasi included, 2 kandidat pending_location:
Bukit Salju dan Pantai Siri Kemuning. Kandidat pending tidak masuk training
atau indeks. Destinasi lain pada workbook tidak otomatis dinyatakan di luar
Madura; mereka hanya tidak termasuk daftar yang ditetapkan. Nama/ID yang
berbeda dari manifest menyebabkan error, bukan penggabungan diam-diam.

## Eksekusi

```bash
python -m wisata.labeling data/koreksi_dataset.xlsx --output artifacts/madura_labels.csv
python -m wisata.experiments artifacts/madura_labels.csv --output artifacts/experiments-dataset-v1 --epochs 10
```

File CSV dan laporan JSON label berada di artifacts, diabaikan Git.
Template anotasi manual tetap tersedia melalui wisata.annotations untuk validasi
lanjutan; tidak menjadi prasyarat ulang untuk label turunan yang telah disetujui.
