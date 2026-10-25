"""Entrypoint tugas Malik: hasil terpilih dari notebook Adaptive Fusion."""
from hashlib import sha256
import os
from pathlib import Path

import streamlit as st


@st.cache_resource
def load_bundle(path,digest):
    from wisata.fusion_serving import FusionService
    return FusionService(Path(path))


def main():
    st.set_page_config(page_title='Rekomendasi Wisata · Adaptive Fusion',page_icon='🧭',layout='wide')
    st.title('🧭 Rekomendasi Wisata Jawa Timur')
    st.caption('Tugas Malik · BGE-M3, E5, dan Adaptive Fusion')
    path=os.environ.get('FUSION_BUNDLE_DIR')
    if not path:
        st.info('Hasil eksperimen belum dipasang. Selesaikan notebook Colab, lalu pasang bundle model terpilih.')
        return
    try:
        manifest=(Path(path)/'manifest.json').read_bytes()
        service=load_bundle(path,sha256(manifest).hexdigest())
    except (ValueError,OSError) as exc:
        st.error(str(exc));return
    st.write('Model terpilih melalui validasi:',service.meta['selected_method'])
    with st.form('fusion_query'):
        text=st.text_input('Wisata seperti apa yang kamu cari?',placeholder='Wisata alam yang sejuk dan tenang')
        season=st.selectbox('Musim',['Tidak ditentukan','Hujan','Kemarau'])
        k=st.slider('Jumlah rekomendasi',1,20,10)
        strict=st.checkbox('Batasi ke destinasi yang cocok dengan musim',value=True)
        submit=st.form_submit_button('Cari rekomendasi')
    if submit:
        try:
            with st.spinner('Mencari destinasi…'):
                result,detail=service.search(text,{'Tidak ditentukan':'none','Hujan':'hujan','Kemarau':'kemarau'}[season],k,strict)
            if result.empty:st.warning('Tidak ada destinasi yang memenuhi pilihan.');return
            st.dataframe(result,hide_index=True,width='stretch')
            if detail['lambda_bge'] is not None:
                st.caption(f"Bobot adaptif BGE: {detail['lambda_bge']:.3f}; E5: {1-detail['lambda_bge']:.3f}.")
            st.caption('Skor adalah kecocokan teks, bukan probabilitas. Label cuaca bersifat statis dari dataset.')
        except (ValueError,OSError,RuntimeError) as exc:st.error(str(exc))


if __name__=='__main__':main()
