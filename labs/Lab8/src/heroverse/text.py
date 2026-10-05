"""Codificación de texto: bag of words y TF-IDF, ambos dispersos."""

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorizar_tfidf(
    textos: list[str], min_df: int = 2, max_features: int | None = 5000
) -> tuple[TfidfVectorizer, csr_matrix]:
    vectorizador = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_features=max_features,
    )
    matriz = vectorizador.fit_transform(textos)
    return vectorizador, matriz


def vectorizar_bolsa(
    textos: list[str], min_df: int = 2, max_features: int | None = 5000
) -> tuple[CountVectorizer, csr_matrix]:
    contador = CountVectorizer(
        stop_words="english",
        min_df=min_df,
        max_features=max_features,
    )
    matriz = contador.fit_transform(textos)
    return contador, matriz
