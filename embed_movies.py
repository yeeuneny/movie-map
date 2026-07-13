from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


BASE_DIR = Path(__file__).parent

INPUT_PATH = BASE_DIR / "movies.csv"
OUTPUT_PATH = BASE_DIR / "movies_pca.csv"


def clean_text_column(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {INPUT_PATH.resolve()}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"Title", "Genres", "Plot"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "필수 열이 없습니다: "
            + ", ".join(sorted(missing_columns))
        )

    genres = (
        clean_text_column(df["Genres"])
        .str.replace(",", " ", regex=False)
    )

    plots = clean_text_column(df["Plot"])

    # 장르를 두 번 사용해 약간 더 강조
    embedding_text = (
        genres
        + " "
        + genres
        + " "
        + plots
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=1000,
        min_df=1,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(
        embedding_text
    )

    # 각 영화 벡터의 길이 차이 완화
    tfidf_matrix = normalize(
        tfidf_matrix,
        norm="l2"
    )

    if tfidf_matrix.shape[0] < 2:
        raise ValueError(
            "PCA를 수행하려면 영화가 최소 2개 필요합니다."
        )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    coordinates = pca.fit_transform(
        tfidf_matrix.toarray()
    )

    df["PCA1"] = coordinates[:, 0]
    df["PCA2"] = coordinates[:, 1]

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"저장 완료: {OUTPUT_PATH.resolve()}")
    print(
        "설명 분산 비율:",
        pca.explained_variance_ratio_
    )


if __name__ == "__main__":
    main()