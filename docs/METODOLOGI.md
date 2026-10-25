# Metode sesuai proposal asistensi Malik

Acuan utama: **PROPOSAL ASISTENSI PENELITIAN_MALIK HARIYANTO_230441100067.docx.pdf**,
posisi kontribusi halaman cetak 18 dan Bab III §3.3–3.11, halaman cetak 20–22.
PDF TKA yang sempat diberikan tidak digunakan sebagai acuan metode.

## Pembeda penelitian induk

| Komponen | Proposal kolaborasi | Proposal Malik |
| --- | --- | --- |
| Topik | LDA per musim | Tetap LDA |
| Representasi kata | Word2Vec | **FastText, subword karakter n-gram** |
| Klasifikasi | BiLSTM multi-output | Tetap BiLSTM |
| Target | Musim, jenis wisata, sentimen | Tiga aspek yang sama |
| Personalisasi | BGE dan konteks musim | BGE, preferensi/riwayat dan musim |
| Skenario | Word2Vec, LDA, gabungan | FastText, LDA, gabungan |

Notebook BGE-M3/E5/Adaptive Fusion adalah bahan retrieval tambahan.
**Adaptive Fusion bukan pembeda proposal Malik.** Keunggulan FastText pada data
ini belum dibuktikan dan tidak diasumsikan.

## Alur program

```mermaid
flowchart TD
    A[Ulasan dan metadata] --> B[Seleksi wilayah Madura dan verifikasi tanggal]
    B --> C[Pelabelan musim, jenis wisata, sentimen]
    C --> D[Split train 80 persen dan test 20 persen]
    D --> E[Normalisasi, tokenisasi, stopword, stemming]
    E --> F[LDA per musim pada train]
    E --> G[FastText pada train]
    F --> H[Distribusi topik]
    G --> I[Vektor kata subword]
    H --> J[BiLSTM tiga kepala klasifikasi]
    I --> J
    J --> K[Evaluasi Hamming Loss dan Micro F1]
    J --> L[Prediksi multiaspek]
    Q[Preferensi, riwayat, konteks musim] --> M[Retrieval BGE]
    L --> N[Integrasi konteks dan ranking destinasi]
    M --> N
    N --> O[Top-N dan evaluasi precision/recall]
```

## Keputusan implementasi eksplisit

- Jalur audit/retrieval menggunakan teks asli dengan pembersihan ringan. Jalur
  classifier terpisah: lowercase, tokenisasi huruf Unicode, stopword Sastrawi,
  stemming. Negasi tidak/bukan/belum/jangan/kurang dipertahankan untuk sentimen.
- Target saat ini satu kategori per aspek sesuai tabel proposal. Cuaca dan liburan
  dapat beririsan; kebijakan adjudikasi harus ditetapkan. Jika memilih multilabel,
  ubah head/loss/metrik sebelum eksperimen.
- Split 80/20 dikelompokkan menurut teks ternormalisasi agar duplikat identik
  tidak tersebar di train/test. Seed dan review_id split disimpan. Ini belum
  menjamin generalisasi ke destinasi baru atau masa depan.
- Vocabulary, LDA, FastText/Word2Vec hanya fit pada train. Epoch ditetapkan
  sebelum test; tidak ada pemilihan model berdasarkan skor test.
- Bank LDA dilatih terpisah per musim pada train. Saat inferensi **semua bank
  ditransformasi dan dikonkatenasi**, tanpa memilih bank menggunakan label target.
  Memilih bank dari musim jawaban test akan membocorkan target. Proposal belum
  merinci langkah ini; concatenation bank adalah keputusan implementasi.
- FastText saja: urutan vektor kata → BiLSTM. LDA saja: distribusi topik sebagai
  urutan **satu langkah** → BiLSTM. Gabungan: topik dokumen dikonkatenasi ke tiap
  vektor kata → BiLSTM. LDA satu langkah bukan urutan token buatan.
- Tiga kepala categorical softmax; loss adalah jumlah cross-entropy tiga aspek.
  FastText memberi vektor OOV melalui subword; Word2Vec pembanding memakai nol
  untuk OOV dengan seed/dimensi/split yang sama.
- Pada target single-label, Micro F1 per aspek setara accuracy. Hamming Loss
  per aspek adalah proporsi keputusan salah; overall merata-ratakan semua
  keputusan pada tiga aspek. Ini dibedakan dari hamming pada matriks multi-hot.
- Minimal 20 baris approved pada CLI adalah batas teknis, bukan ukuran sampel
  yang dinyatakan cukup secara ilmiah.

## Eksperimen setelah label nyata tersedia

```bash
python -m pip install -r requirements-model.txt
python -m wisata.annotations data/koreksi_dataset.xlsx artifacts/anotasi.csv --limit 200
# Tetapkan konvensi label, verifikasi wilayah, lalu isi dan adjudikasi terlebih dahulu.
python -m wisata.train artifacts/anotasi.csv --scenario fasttext --output artifacts/fasttext
python -m wisata.train artifacts/anotasi.csv --scenario lda --output artifacts/lda
python -m wisata.train artifacts/anotasi.csv --scenario fusion --output artifacts/fusion
python -m wisata.train artifacts/anotasi.csv --scenario fusion --embedding word2vec --output artifacts/word2vec-lda
```

Default region adalah madura. --region all hanya untuk eksperimen tambahan yang
diberi cakupan berbeda. Folder output harus baru/kosong. Artefak: embedding,
bank topik, classifier, konfigurasi, hash label, metrik, ID split.
Muat hanya artefak tepercaya buatan pipeline sendiri; gensim/joblib bukan format
aman untuk unggahan model bebas.

## Batas implementasi

Pipeline classifier tersedia untuk pengujian teknis dengan data sintetis.
Konvensi label data nyata sedang diklarifikasi. Retrieval BGE, integrasi prediksi
ke ranking, riwayat pengguna, ground truth rekomendasi independen, serta deployment
rekomendasi akhir belum diimplementasikan. Notebook tidak memuat hasil eksekusi,
indeks, atau model terlatih yang dapat langsung dijadikan artefak serving.
