# Penjelasan proyek dengan bahasa sederhana

Bayangkan pengguna menulis: **“Cari tempat yang tenang untuk keluarga saat hujan.”**
Aplikasi perlu membaca keinginan itu, mencari destinasi yang ulasannya sesuai,
dan memeriksa kecocokan musim. Data yang sekarang sudah berguna untuk fungsi ini.

## Dua pekerjaan yang berbeda dalam proposal

**Membaca dan mengelompokkan ulasan:** tentukan sentimen, jenis wisata, dan musim.
Bagian ini memakai LDA, FastText, dan BiLSTM. Pembeda Malik adalah FastText
menggantikan Word2Vec dari penelitian induk.

**Mencari tempat sesuai keinginan:** cocokkan kalimat pengguna dengan ulasan
destinasi melalui BGE, lalu gunakan konteks untuk menyusun rekomendasi.
Notebook yang dikirim terutama membantu pekerjaan kedua ini. Menggunakan notebook
BGE saja belum menjalankan semua metode pada proposal Malik.

## Arti data yang tersedia

| Kolom | Yang bisa dibantu | Yang belum otomatis diketahui |
| --- | --- | --- |
| text_original | Membaca pengalaman pengunjung dan mencocokkan permintaan | Label jawaban benar untuk menilai model |
| rating | Petunjuk positif/netral/negatif | Sentimen teks yang sudah diperiksa manusia |
| date | Waktu ulasan relatif | Tanggal kunjungan sebenarnya |
| rekomendasi_cuaca | Kecocokan destinasi untuk kemarau atau kedua musim | Musim saat penulis berkunjung; liburan sekolah/nasional |
| wisata | Jenis destinasi | Pemetaan yang disepakati ke lima jenis pada proposal |

Bintang lima dapat dijadikan proksi positif dan bintang satu proksi negatif,
tetapi orang bisa memberi bintang lima sambil mengeluhkan parkir. Karena itu
laporan harus menyebut sentimen berasal dari rating jika itulah cara pelabelannya.

Museum dapat cocok saat kemarau **dan** hujan. Itu informasi tentang museumnya,
bukan bukti bahwa suatu ulasan ditulis setelah kunjungan musim hujan.
“1 hari lalu” perlu tanggal scraping untuk menjadi tanggal kalender; bahkan
setelah dikonversi, itu tetap waktu ulasan, belum tentu waktu kunjungan.

## Saran jalur kerja

1. Gunakan dataset yang ada untuk workbench dan pencarian destinasi.
2. Tulis terang sumber setiap label: rating, metadata destinasi, atau anotasi manusia.
3. Cocokkan cakupan lokasi: proposal menyebut Madura, workbook mencakup Jawa Timur.
4. Jika tetap mengikuti empat kategori musim/liburan pada proposal, lengkapi data
   pendukungnya. Jika memakai kecocokan cuaca saja, catat penyesuaian ruang lingkup
   dan diskusikan dengan pembimbing sebelum menganggapnya eksperimen proposal penuh.
5. Uji tiga skenario classifier, lalu integrasikan BGE ke aplikasi. Pengujian kode
   dengan contoh buatan hanya membuktikan program berjalan, bukan model sudah bagus.

Tidak perlu memaksa semua kolom ada dengan cara mengarang label. Dataset tetap
berguna; yang perlu dirapikan adalah hubungan antara pertanyaan riset dan arti data.
