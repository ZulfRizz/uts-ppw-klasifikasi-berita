import streamlit as st
import trafilatura
import joblib

from preprocessing import clean_for_model

MODEL_PATH = "models/best_model_tfidf.joblib"

st.set_page_config(page_title="Klasifikasi Berita", page_icon="📰")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def crawl_article(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    return trafilatura.extract(downloaded)


model = load_model()

st.title("📰 Klasifikasi Berita: Sport vs Finance")
st.caption("Masukkan URL artikel detik.com, sistem akan mengambil isinya "
           "secara otomatis lalu mengklasifikasikannya.")

url = st.text_input("URL artikel", placeholder="https://sport.detik.com/...")
run = st.button("Klasifikasikan", type="primary")

if run:
    if not url.strip():
        st.warning("Masukkan URL terlebih dahulu.")
        st.stop()

    with st.spinner("Mengambil isi artikel..."):
        article_text = crawl_article(url.strip())

    if not article_text:
        st.error(
            "Gagal mengambil isi artikel dari URL ini. Kemungkinan "
            "halaman tidak bisa diakses, atau strukturnya tidak "
            "dikenali trafilatura. Coba URL lain."
        )
        st.stop()

    with st.spinner("Membersihkan teks..."):
        cleaned = clean_for_model(article_text)

    if not cleaned.strip():
        st.error("Teks hasil crawling kosong setelah dibersihkan. "
                 "Kemungkinan halaman bukan artikel berita utuh.")
        st.stop()

    with st.spinner("Mengklasifikasikan..."):
        pred_label = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]
        classes = model.classes_

    st.subheader(f"Hasil: **{pred_label.upper()}**")

    cols = st.columns(len(classes))
    for col, cls, p in zip(cols, classes, proba):
        col.metric(cls, f"{p * 100:.1f}%")

    with st.expander("Lihat teks hasil crawling (sebelum dibersihkan)"):
        st.write(article_text)

    with st.expander("Lihat token hasil pembersihan (dikirim ke model)"):
        st.code(cleaned)
