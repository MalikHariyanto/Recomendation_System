"""Workbench Streamlit: audit data riset sebelum integrasi model usulan."""

import json
import os
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from wisata.data import DatasetError, audit_payload, prepare_data, read_excel
from wisata.labeling import DEFAULT_SCOPE, derive_labels, scoped_reviews


@st.cache_resource
def load_index(path: str, manifest_sha: str):
    from wisata.retrieval import ReviewIndex
    return ReviewIndex(Path(path))


def render_search():
    st.subheader("Cari destinasi wisata")
    directory = os.environ.get("WISATA_INDEX_DIR")
    if not directory:
        st.info("Pencarian menunggu indeks BGE selesai disiapkan. Pemeriksaan data dan label tersedia di bawah.")
        return
    try:
        manifest = (Path(directory) / "manifest.json").read_bytes()
        index = load_index(directory, sha256(manifest).hexdigest())
        categories = sorted(index.reviews["tourism_label"].unique())
    except (ImportError, OSError, ValueError, RuntimeError):
        st.error("Indeks rekomendasi belum dapat dibaca. Periksa konfigurasi artefak.")
        return
    with st.form("search"):
        query = st.text_input("Ceritakan wisata yang lo cari", placeholder="Tempat tenang untuk keluarga")
        season = st.selectbox("Rencana musim perjalanan", ["Semua", "Hujan", "Kemarau"], key="search_season")
        category = st.selectbox("Jenis wisata", ["Semua"] + categories, key="search_category")
        sentiment = st.selectbox("Ulasan yang dipertimbangkan", ["Semua", "positif", "netral", "negatif"], key="search_sentiment")
        submit = st.form_submit_button("Cari rekomendasi")
    if submit:
        if not query.strip():
            st.warning("Isi kebutuhan wisata terlebih dahulu.")
            return
        try:
            with st.spinner("Mencocokkan kebutuhan dengan ulasan destinasi…"):
                result = index.search(query, season=season, category=category, sentiment=sentiment)
            if result.empty:
                st.warning("Belum ada destinasi yang memenuhi pilihan tersebut. Coba pilihan musim atau ulasan lain.")
            else:
                st.dataframe(result, hide_index=True, width="stretch")
                st.caption("Skor menunjukkan kecocokan teks, bukan persentase kepastian. "
                           "Musim mengikuti rekomendasi_cuaca dataset. Ulasan bukti adalah kutipan pendukung.")
        except (ImportError, OSError, ValueError, RuntimeError):
            st.error("Indeks atau encoder BGE belum dapat dimuat. Periksa artefak dan akses unduhan model di environment.")


def main() -> None:
    st.set_page_config(page_title="Riset Wisata · Pemeriksaan Data", page_icon="🧭", layout="wide")
    st.title("🧭 Rekomendasi Wisata Madura")
    st.caption("Workbench persiapan data untuk aplikasi rekomendasi sadar musim.")
    st.caption("Metode: LDA + FastText + BiLSTM, dilanjutkan pencarian BGE. "
               "Sentimen dari rating; label musim mengikuti rekomendasi_cuaca.")
    render_search()

    with st.expander("Klasifikasi ulasan dari model lokal"):
        model_dir = os.environ.get("WISATA_MODEL_DIR")
        if not model_dir:
            st.caption("Artefak classifier belum dikonfigurasi untuk sesi ini.")
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
        st.markdown("Metode asal, metode FastText, serta temuan audit ada di "
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

    scope = pd.read_csv(os.environ.get("WISATA_SCOPE_PATH", str(DEFAULT_SCOPE)))
    try:
        research = scoped_reviews(prepared.reviews, scope)
        labels = derive_labels(prepared.reviews, scope) if not research.empty else None
    except DatasetError as exc:
        st.error(str(exc))
        return
    st.write(f"Cakupan penelitian Madura: **{research['destinasi'].nunique()} destinasi**, "
             f"**{len(research):,} ulasan bersih**. Ringkasan di atas adalah seluruh workbook sumber.")
    tab_catalog, tab_labels, tab_audit, tab_method = st.tabs(["Katalog destinasi", "Label penelitian", "Kualitas data", "Alur riset"])
    with tab_catalog:
        region = st.radio("Cakupan katalog", ["Madura (penelitian)", "Seluruh dataset (audit)"], horizontal=True)
        category = st.selectbox("Kategori wisata", ["Semua"] + sorted(prepared.catalog["wisata"].unique()), key="catalog_category")
        season = st.selectbox("Kecocokan musim berdasarkan label dataset", ["Semua", "Hujan", "Kemarau"], key="catalog_season")
        filtered = prepared.catalog.copy()
        if region == "Madura (penelitian)":
            filtered = filtered.loc[filtered["destinasi"].isin(research["destinasi"])]
        if category != "Semua":
            filtered = filtered.loc[filtered["wisata"].eq(category)]
        if season != "Semua":
            filtered = filtered.loc[filtered["cocok_hujan" if season == "Hujan" else "cocok_kemarau"]]
        st.caption(f"{len(filtered)} destinasi · urut abjad, bukan ranking rekomendasi. "
                   "Rating rata-rata dihitung dari ulasan yang lolos pembersihan.")
        st.dataframe(filtered[["destinasi", "wisata", "musim", "n_ulasan", "rating_mean"]],
                     hide_index=True, width="stretch")
        st.caption("Label musim bersifat statis; aplikasi belum menggunakan prakiraan cuaca langsung.")
    with tab_labels:
        st.markdown("**Sentimen:** rating 1–2 negatif, 3 netral, 4–5 positif. "
                    "**Musim:** nilai asli rekomendasi_cuaca, termasuk Kemarau & Penghujan. "
                    "**Jenis wisata:** kategori asli wisata. Semua ini label turunan metadata, bukan anotasi teks manusia.")
        if labels is not None:
            for column, title in [("sentiment_label", "Sentimen"), ("season_label", "Musim"), ("tourism_label", "Jenis wisata")]:
                st.write(title)
                st.bar_chart(labels[column].value_counts())
            st.download_button("Unduh label penelitian (CSV)", labels.to_csv(index=False).encode("utf-8"),
                               file_name="madura_labels.csv", mime="text/csv")
        pending = scope.loc[scope["status"].eq("pending_location"), ["destinasi", "basis"]]
        if not pending.empty:
            st.caption("Destinasi berikut belum dimasukkan karena identitas/lokasi perlu diverifikasi:")
            st.dataframe(pending, hide_index=True)
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

**Pelabelan yang ditetapkan:** rating 1–2 negatif, 3 netral, 4–5 positif; musim mengikuti
rekomendasi_cuaca dan jenis wisata mengikuti wisata. Tidak membuat kelas liburan
yang tidak tersedia pada dataset. Cakupan wilayah mengikuti Madura pada proposal kolaborasi.
Proses training dan evaluasi dijalankan terpisah dari interaksi Streamlit.
""")


if __name__ == "__main__":
    main()
