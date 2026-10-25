# Rujukan dan status pemeriksaan

Tidak semua daftar pustaka dalam notebook telah diverifikasi. Paper FastText dan
BGE-M3 berhasil diunduh dari arXiv serta LDA dari JMLR. Bagian metode yang relevan
telah diperiksa; status sumber lainnya dibedakan di bawah.
Jangan mengartikan catatan ini sebagai systematic literature review yang selesai.

## Paper yang bagian metodenya diperiksa

- Blei, Ng, dan Jordan (2003), Latent Dirichlet Allocation,
  https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf, §2–3. Dokumen
  direpresentasikan sebagai campuran distribusi topik dengan asumsi bag-of-words;
  fitur classifier adalah distribusi topik, bukan label sentimen. Implementasi
  memakai CountVectorizer dan LDA scikit-learn. Bank cuaca dan concatenation
  seluruh bank adalah rancangan proyek, bukan metode musim/BiLSTM pada paper asli.
- Bojanowski et al. (2017), Enriching Word Vectors with Subword Information,
  https://arxiv.org/abs/1607.04606, §3.1–3.2 dan §4.3. Skip-gram dikembangkan
  dengan representasi kata melalui n-gram karakter 3–6, termasuk boundary
  karakter dan kata utuh; mendukung kata yang tidak ada pada vocabulary.
  Implementasi proyek menggunakan Gensim, dimensi 64, bucket 50.000, bukan
  replikasi hyperparameter paper (dimensi 300 dan bucket dua juta).
- Chen et al. (2024), BGE M3-Embedding, https://arxiv.org/abs/2402.03216, §3.1
  dan arsitektur retrieval: dense memakai hidden state CLS ternormalisasi;
  sparse dan multi-vector adalah mode lain. Proyek memilih dense saja.
  Paper mendukung panjang hingga 8192, proyek membatasi 256 untuk biaya CPU.

Tidak memindahkan nilai benchmark paper menjadi hasil pada dataset proyek.

## Sumber resmi yang berhasil diperiksa

0. [fastText, repositori resmi](https://github.com/facebookresearch/fastText), bagian
   Learning word vectors dan Obtaining word vectors for out-of-vocabulary words.
   Mendokumentasikan subword karakter n-gram default 3–6 dan vektor OOV.
   Ini landasan representasi kata pada metode Malik, bukan classifier fastText
   supervised yang menggantikan BiLSTM. Paper acuan proposal:
   [Bojanowski et al. (2017), Enriching Word Vectors with Subword Information](https://doi.org/10.1162/tacl_a_00051).
   Kode proyek memakai implementasi FastText dari Gensim, bukan binding resmi C++.
1. [FlagEmbedding, dokumentasi BGE-M3](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3).
   Mendukung dense, sparse, dan multi-vector retrieval, embedding dense 1024 dimensi,
   hingga 8192 token, dan query tanpa instruksi tambahan. Notebook memilih **dense**
   dan batas 256 token; jangan menyebut implementasinya hybrid dense+sparse+ColBERT.
   Paper terkait: [Chen et al., arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
2. [Microsoft UniLM, E5](https://github.com/microsoft/unilm/tree/master/e5), khususnya
   [model_config.py](https://github.com/microsoft/unilm/blob/master/e5/model_config.py)
   dan [mteb_beir_eval.py](https://github.com/microsoft/unilm/blob/master/e5/mteb_beir_eval.py).
   Untuk multilingual-e5-large, kode resmi memakai average pooling, prefix
   `query: `/`passage: `, dan normalisasi L2. Dimensi yang didokumentasikan: 1024.
   Paper terkait: [Wang et al., arXiv:2402.05672](https://arxiv.org/abs/2402.05672).

Sumber resmi di atas dibaca langsung melalui raw.githubusercontent.com. Klaim
benchmark mereka tidak dipindahkan menjadi hasil pada dataset Jawa Timur.

## Rujukan notebook yang perlu dibaca dan diverifikasi berikutnya

Rujukan utama proposal Malik tambahan: Blei et al. (2003), Latent Dirichlet
Allocation, https://www.jmlr.org/papers/v3/blei03a.html, dan Schuster & Paliwal
(1997), Bidirectional Recurrent Neural Networks, https://doi.org/10.1109/78.650093.
Paper LDA telah diperiksa seperti catatan di atas; paper BiRNN belum diperiksa
langsung di sesi ini. Tidak diklaim telah direplikasi persis.
Studi FastText lokal dalam proposal (RESTI DOI 10.29207/resti.v6i4.4187 dan
10.29207/resti.v6i5.4429) serta PeerJ DOI 10.7717/peerj-cs.422 juga perlu dibaca
langsung sebelum memindahkan klaim atau angka mereka ke laporan.

| Rujukan | Peran yang diharapkan | Status |
| --- | --- | --- |
| [POINTREC, Afzali et al. (2021)](https://doi.org/10.1145/3404835.3463243) | Permintaan wisata naratif dan anotasi relevansi | Diidentifikasi dari notebook; full text belum diperiksa |
| [TourismQA, Contractor et al. (2021)](https://doi.org/10.1145/3459637.3482320) | Representasi POI lewat ulasan dan query forum | Diidentifikasi dari notebook; full text belum diperiksa |
| [LAMB, arXiv:2401.02187](https://arxiv.org/abs/2401.02187) | Bi-encoder untuk tourism QA | Akses full text ditolak proxy |
| [SynthTRIPs (2025)](https://doi.org/10.1145/3726302.3730321) | Persona dan query sintetis bertingkat | Metadata dari notebook; belum diverifikasi langsung |
| [ListNet, Cao et al. (2007)](https://doi.org/10.1145/1273496.1273513) | Landasan learning to rank listwise | Sumber Microsoft yang dicoba ditolak proxy; formulasi notebook perlu dibedakan dari asli |
| [RRF, Cormack et al. (2009)](https://doi.org/10.1145/1571941.1572114) | Baseline fusi ranking | DOI dari notebook; full text belum diperiksa |
| [Järvelin & Kekäläinen (2002)](https://doi.org/10.1145/582415.582418) | Evaluasi graded relevance/NDCG | DOI dari notebook; full text belum diperiksa |

Query terinspirasi suatu paper bukan dataset resmi paper tersebut. Protokol A/C/D
notebook tetap koleksi lokal dengan qrels heuristik. Ketersediaan skripsi Prasetyo,
paper lokal tahun 2026, identitas penulis, dan hasil pembanding belum diverifikasi;
jangan mengutip angka atau menyatakan keunggulan atas studi tersebut dahulu.

## Kebutuhan akses

Draft environment menambahkan `arxiv.org`, `huggingface.co`, `www.microsoft.com`,
dan `krisztianbalog.com` ke daftar domain khusus, tanpa mengganti preset package
manager. Perubahan draft belum berarti akses runtime aktif. Perlu review/save
dan publish melalui pengaturan environment, lalu ulangi pengambilan sumber.
Domain tambahan us.aws.cdn.hf.co (redirect bobot BGE), id.wikipedia.org
(pemeriksaan identitas destinasi), dan www.jmlr.org (paper LDA) sudah disimpan
di draft. Akses bobot BGE serta paper LDA akhirnya berhasil melalui library/HTTPS resmi.
Penyimpanan draft tetap dibedakan dari penerapan atau publication environment.
