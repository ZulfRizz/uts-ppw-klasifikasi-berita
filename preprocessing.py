"""Modul pembersihan teks, disalin dari notebook preprocessing supaya
logikanya identik antara pelatihan model dan aplikasi deploy."""

import re
import unicodedata
from functools import lru_cache

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Stemmer.Stemmer import Stemmer
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from indoNLP.preprocessing import replace_slang

# --- setup sekali di awal (dipakai berulang, jadi di luar fungsi) ---
_id_stopwords = set(StopWordRemoverFactory().get_stop_words())
_stemmer = Stemmer(ArrayDictionary(StemmerFactory().get_words()))

_byline_re = r"\(\w{2,6}(?:/\w{2,6})+\)"
_boiler_re = rf"(?:{_byline_re}\s*)?Anda menyukai artikel ini\s+Artikel disimpan\s*$"
_url_pattern = re.compile(r"(?:https?://|www\.|pic\.twitter\.com/)\S+")
_alnum_pattern = re.compile(r"\b(?=[A-Za-z]*\d)[A-Za-z0-9][A-Za-z0-9.*-]*\b")
_currency_pattern = re.compile(
    r"\bRp\d[\d.,]*\b|\bUS\$\d[\d.,]*\b|\b\$\d[\d.,]*\b|\bRp\b|\bUS\$\b")
_roman_pattern = re.compile(
    r"\b(?:iii|iv|vi{0,3}|i[ivx]{0,2}|v|x{1,3}|xi{0,3})\b", re.IGNORECASE)

# Kata yang terbukti rawan salah ubah oleh kamus slang pada dataset latihan.
# Kalau nanti menemukan kata lain yang salah ubah, tambahkan di sini.
EXCLUDED_SLANG = {"minta", "pala", "fans"}


def _strip_boilerplate(text: str) -> str:
    text = re.sub(r"^Foto\s+(?:Sport|Finance)\s+", "", text)
    return re.sub(_boiler_re, "", text).strip()


def _normalize_letters(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return text.replace("\u2011", "-").replace("\u2010", "-")


def _protect_formats(text: str) -> str:
    text = _url_pattern.sub(" ", text)
    text = _currency_pattern.sub(" ", text)
    text = _alnum_pattern.sub(" ", text)
    text = _roman_pattern.sub(" ", text)
    return text


def _mid_sentence(text: str, pos: int) -> bool:
    i = pos - 1
    while i >= 0 and text[i] in ' \n\t"\'":(':
        i -= 1
    return i >= 0 and text[i] not in '.!?;:(\u2013\u2014'


def _guarded_slang_replace(text: str) -> str:
    changes = []
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group(0)
        repl = replace_slang(w.lower())
        if repl == w.lower() or w.lower() in EXCLUDED_SLANG:
            continue
        before = text[max(0, m.start() - 6):m.start()]
        after = text[m.end():m.end() + 6]
        near_digit = bool(re.search(r"\d", before + after))
        looks_name = (w.isupper() or any(c.isupper() for c in w[1:])
                      or (w[:1].isupper() and w[1:].islower()
                          and _mid_sentence(text, m.start())))
        two_letter = len(w) <= 2 and w.lower() not in _id_stopwords
        if not (near_digit or looks_name or two_letter):
            changes.append((m.start(), m.end(), repl))
    out = list(text)
    for s, e, r in reversed(changes):
        out[s:e] = list(r)
    return "".join(out)


@lru_cache(maxsize=None)
def _stem_word(token: str) -> str:
    return _stemmer.stem(token)


def clean_and_tokenize(raw_text: str) -> list[str]:
    """Satu artikel mentah -> daftar token bersih siap masuk ke model."""
    text = _strip_boilerplate(raw_text)
    text = _normalize_letters(_protect_formats(text))

    # Kumpulkan kandidat nama diri: kata berkapital di tengah kalimat.
    # Untuk satu artikel tunggal, cukup satu kemunculan sudah dianggap
    # nama (tanpa ambang jumlah seperti versi korpus di notebook).
    protected = set()
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group(0)
        if len(w) >= 3 and w[0].isupper() and _mid_sentence(text, m.start()):
            protected.add(w.lower())

    text = _guarded_slang_replace(text)
    text = re.sub(r"[^A-Za-z\s-]", " ", text)
    text = re.sub(r"-+", " ", text)
    tokens = [w for w in text.split() if len(w) >= 2]
    tokens = [w for w in tokens
              if not (len(w) == 2 and w.lower() not in _id_stopwords)]
    tokens = [w for w in tokens if w.lower() not in _id_stopwords]

    stemmed = []
    for w in tokens:
        lw = w.lower()
        stemmed.append(lw if lw in protected else _stem_word(lw))
    return stemmed


def clean_for_model(raw_text: str) -> str:
    """Keluaran siap pakai: satu string token dipisah spasi, sama persis
    format yang dipakai saat melatih pipeline TF-IDF + Naive Bayes."""
    return " ".join(clean_and_tokenize(raw_text))
