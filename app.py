"""Workbench Streamlit: audit data riset sebelum integrasi model usulan."""

import json
import os
from pathlib import Path

import streamlit as st

from wisata.data import DatasetError, audit_payload, prepare_data, read_excel


def main() -> None:
    st.set_page_config(page_title="Riset Wisata · Pemeriksaan Data", page_icon="🧭", layout="wide")
    st.title("🧭 Riset Rekomendasi Wisata Jawa Timur")
    st.caption("Workbench persiapan data untuk aplikasi rekomendasi sadar musim.")
    st.info("Metode acuan: LDA + FastText + BiLSTM multiaspek, lalu BGE untuk rekomendasi. "
            "Workbench ini belum menghasilkan rekomendasi; training riset memerlukan label target yang disepakati.")

    with st.expander("Klasifikasi ulasan dari model lokal"):
        model_dir = os.environ.get("WISATA_MODEL_DIR")
        if not model_dir:
            st.caption("Belum ada artefak model riset. Tetapkan label dan jalankan training, lalu atur WISATA_MODEL_DIR.")
        else:
            review = st.text_area("Ulasan yang akan diklasifikasikan")
            if st.button("Klasifikasikan ulasan"):
                try:
                    from wisata.inference import Classifier
                    classifier = Classifier(Path(model_dir))
                    st.write(classifier.predict(review))
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Model belum dapat dipakai: {exc}")

    with st.sidebar:
        st.header("Dataset ulasan")
        upload = st.file_uploader("Unggah koreksi_dataset.xlsx", type=["xlsx"])
        st.caption("Gunakan data yang diizinkan tim riset. Ulasan tidak disimpan ke repository.")
        st.markdown("**Alur persiapan**\n\n1. Validasi skema dan label\n2. Bersihkan teks asli\n"
                    "3. Periksa katalog dan distribusi\n4. Unduh ringkasan audit")

    content = None
    if upload is not None:
        content = upload.getvalue()
    elif os.environ.get("WISATA_DATA_PATH"):
        try:
            content = Path(os.environ["WISATA_DATA_PATH"]).read_bytes()
        except OSError:
            st.error("Dataset lokal tidak dapat dibaca. Periksa WISATA_DATA_PATH atau unggah file.")
            return
    if content is None:
        st.markdown("Unggah dataset melalui panel kiri untuk melihat statistik dan katalog destinasi.")
        st.markdown("Metode asal, rancangan Adaptive Fusion, serta temuan audit ada di "
                    "`docs/METODOLOGI.md` dan `docs/AUDIT_RISET.md`.")
        return

    # Data pengguna tetap di sesi ini; tidak memakai cache global lintas pengguna.
    if st.session_state.get("dataset_content") != content:
        try:
            with st.spinner("Memvalidasi dan menyiapkan ulasan…"):
                prepared = prepare_data(read_excel(content))
            st.session_state["prepared"] = prepared
            st.session_state["dataset_content"] = content
        except DatasetError as exc:
            st.error(str(exc))
            return
    prepared = st.session_state["prepared"]
    audit = prepared.audit
    report = audit_payload(content, prepared)
    cols = st.columns(4)
    for column, label, value in zip(cols, ("Ulasan asli", "Ulasan bersih", "Destinasi", "Kategori"),
                                     (audit.original_rows, audit.clean_reviews,
                                      audit.clean_destinations, audit.categories)):
        column.metric(label, f"{value:,}")

    tab_catalog, tab_audit, tab_method = st.tabs(["Katalog destinasi", "Kualitas data", "Alur riset"])
    with tab_catalog:
        category = st.selectbox("Kategori wisata", ["Semua"] + sorted(prepared.catalog["wisata"].unique()))
        season = st.selectbox("Kecocokan musim berdasarkan label dataset", ["Semua", "Hujan", "Kemarau"])
        filtered = prepared.catalog.copy()
        if category != "Semua":
            filtered = filtered.loc[filtered["wisata"].eq(category)]
        if season != "Semua":
            filtered = filtered.loc[filtered["cocok_hujan" if season == "Hujan" else "cocok_kemarau"]]
        st.caption(f"{len(filtered)} destinasi · urut abjad, bukan ranking rekomendasi. "
                   "Rating rata-rata dihitung dari ulasan yang lolos pembersihan.")
        st.dataframe(filtered[["destinasi", "wisata", "musim", "n_ulasan", "rating_mean"]],
                     hide_index=True, width="stretch")
        st.caption("Label musim bersifat statis; aplikasi belum menggunakan prakiraan cuaca langsung.")
    with tab_audit:
        st.write({"Ulasan kosong (bagian dari ulasan pendek)": audit.empty_reviews,
                  "Ulasan kurang dari 3 kata dibuang": audit.short_reviews_removed,
                  "Duplikat destinasi + teks dibuang": audit.duplicate_reviews_removed,
                  "Destinasi hilang setelah pembersihan": audit.original_destinations - audit.clean_destinations})
        left, right = st.columns(2)
        with left:
            st.subheader("Destinasi per kategori")
            st.bar_chart(prepared.catalog["wisata"].value_counts())
        with right:
            st.subheader("Destinasi per label musim")
            st.bar_chart(prepared.catalog["musim"].value_counts())
        st.download_button("Unduh ringkasan audit (JSON)", json.dumps(report, ensure_ascii=False, indent=2),
                           file_name="audit_dataset.json", mime="application/json")
    with tab_method:
        st.markdown("""
**Persiapan yang sudah tersedia:** validasi delapan kolom, pemeriksaan pemetaan ID,
pembersihan `text_original` dengan ftfy dan perapian spasi, penyaringan minimal
tiga kata, penghapusan duplikat per destinasi, serta pembentukan katalog.

**Metode proposal Malik:** seleksi data Madura dan pelabelan → normalisasi, tokenisasi,
stopword removal, stemming → LDA per kelompok musim dan FastText subword → BiLSTM
dengan tiga output (musim, jenis wisata, sentimen) → Hamming Loss dan Micro F1 →
personalisasi BGE dengan konteks musim.

**Pembeda penelitian induk:** FastText menggantikan Word2Vec. Tiga skenario:
FastText saja, LDA saja, dan gabungan. Adaptive Fusion BGE-M3/E5 pada notebook
adalah bahan tambahan, bukan metode utama proposal Malik.

**Pelabelan:** rating dapat dipakai sebagai proksi sentimen dengan aturan eksplisit.
Label kecocokan cuaca destinasi perlu dibedakan dari musim kunjungan dan liburan.
Proses training dan evaluasi dijalankan terpisah dari interaksi Streamlit.
""")


if __name__ == "__main__":
    main()
