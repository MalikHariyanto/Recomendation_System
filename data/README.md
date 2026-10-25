# Dataset lokal

Letakkan `koreksi_dataset.xlsx` di folder ini untuk pengembangan lokal, atau unggah
melalui aplikasi. File dataset diabaikan Git. Dokumen proposal, identitas peneliti,
dan workbook mentah tidak disertakan dalam kontribusi kode.

Skema: `id_destinasi`, `destinasi`, `rating`, `date`, `text_original`,
`text_preprocessed`, `rekomendasi_cuaca`, `wisata`.

Kolom `date` dipertahankan sebagai data sumber; nilai tanggal relatif tidak otomatis
diubah menjadi tanggal kunjungan. `text_preprocessed` dipertahankan untuk audit,
sedangkan kandidat embedding menggunakan teks asli yang dibersihkan ringan.
