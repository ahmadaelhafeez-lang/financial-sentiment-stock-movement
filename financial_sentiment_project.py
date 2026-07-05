#!/usr/bin/env python
# coding: utf-8

# # Version 1 and Version 2 - Time-Aware Sentiment-Enhanced Stock Movement Prediction

# This script integrates and consolidates the functionalities from both Version 1 and Version 2 of the project, allowing for comprehensive data processing, sentiment analysis, market feature engineering, and model evaluation.


import os
import re
import zipfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =======================================================
# GENERAL SETTINGS
# =======================================================

ZIP_PATH = "raw data.zip"
# Use a local working directory if not using Google Drive
WORK_DIR_V1 = Path("financial_sentiment_project_v1")
WORK_DIR_V2 = Path("financial_sentiment_project_v2")

RAW_DIR_V1 = WORK_DIR_V1 / "raw"
OUT_DIR_V1 = WORK_DIR_V1 / "outputs"
RAW_DIR_V1.mkdir(parents=True, exist_ok=True)
OUT_DIR_V1.mkdir(parents=True, exist_ok=True)

RAW_DIR_V2 = WORK_DIR_V2 / "raw"
OUT_DIR_V2 = WORK_DIR_V2 / "outputs"
RAW_DIR_V2.mkdir(parents=True, exist_ok=True)
OUT_DIR_V2.mkdir(parents=True, exist_ok=True)

# FinBERT Settings
RUN_FINBERT = True  # Set to False to skip FinBERT processing
MAX_TWEETS_FOR_FINBERT = None # Use None for all tweets. For fast test, use an integer like 10000 first.
FINBERT_BATCH_SIZE = 64

# Time-series cross-validation settings
N_FOLDS = 5
MIN_TRAIN_DATES_RATIO = 0.50

# Target definition (for classification)
# y = 1 if next-day return > TARGET_THRESHOLD
# y = 0 if next-day return < -TARGET_THRESHOLD
# Rows with movements within +/-TARGET_THRESHOLD are removed (neutral/noisy)
TARGET_THRESHOLD = 0.005

print(f"Version 1 outputs will be saved to: {OUT_DIR_V1}")
print(f"Version 2 outputs will be saved to: {OUT_DIR_V2}")

# =======================================================
# HELPER FUNCTIONS
# =======================================================

def ensure_zip_exists(zip_path: str = ZIP_PATH) -> str:
    if os.path.exists(zip_path):
        print(f"Found zip file: {zip_path}")
        return zip_path
    print("Please ensure 'raw data.zip' is in the same directory as the script or provide its path.")
    raise FileNotFoundError("raw data.zip not found. Please provide the zip file.")

def extract_zip(zip_path: str, raw_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"Extracted to: {raw_dir}")

def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]

def read_financial_phrasebank(path: Path) -> pd.DataFrame:
    for enc in ["ISO-8859-1", "latin-1", "utf-8"]:
        try:
            df = pd.read_csv(path, encoding=enc, header=None, names=["sentiment", "text"])
            df = df.dropna(subset=["sentiment", "text"]).drop_duplicates()
            df["sentiment"] = df["sentiment"].str.lower().str.strip()
            df["source"] = "financial_phrasebank"
            print(f"Loaded Financial PhraseBank: {df.shape}, encoding={{enc}}")
            return df
        except Exception:
            continue
    raise ValueError(f"Could not read {path}")

def load_raw_data(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tweets_path = find_file(raw_dir, "stock_tweets.csv")
    prices_path = find_file(raw_dir, "stock_yfinance_data.csv")
    phrase_path = find_file(raw_dir, "all-data.csv")

    tweets = pd.read_csv(tweets_path)
    prices = pd.read_csv(prices_path)
    phrasebank = read_financial_phrasebank(phrase_path)

    print("Tweets:", tweets.shape, tweets.columns.tolist())
    print("Prices:", prices.shape, prices.columns.tolist())
    print("PhraseBank:", phrasebank.shape, phrasebank.columns.tolist())
    return tweets, prices, phrasebank

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\$([A-Za-z]+)", r"", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\.\,\-\%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_vader_textblob(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    out = df.copy()

    vader_scores = out[text_col].fillna("").apply(analyzer.polarity_scores)
    out["vader_negative"] = vader_scores.apply(lambda d: d["neg"])
    out["vader_neutral"] = vader_scores.apply(lambda d: d["neu"])
    out["vader_positive"] = vader_scores.apply(lambda d: d["pos"])
    out["vader_compound"] = vader_scores.apply(lambda d: d["compound"])

    out["textblob_polarity"] = out[text_col].fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)
    out["textblob_subjectivity"] = out[text_col].fillna("").apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return out

def add_finbert_scores(df: pd.DataFrame, text_col: str = "clean_text", limit: int = None) -> pd.DataFrame:
    # Check for cached FinBERT scores to avoid re-running if possible
    cache_path = OUT_DIR_V2 / "tweet_finbert_scores_cache.csv"
    if cache_path.exists():
        print(f"Loading cached FinBERT scores from {cache_path}")
        return pd.read_csv(cache_path)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch

    out = df.copy()
    out["finbert_positive"] = np.nan
    out["finbert_negative"] = np.nan
    out["finbert_neutral"] = np.nan
    out["finbert_score"] = np.nan

    idx = out.index if limit is None else out.index[:limit]
    texts = out.loc[idx, text_col].fillna("").astype(str).str.slice(0, 512).tolist()

    model_name = "ProsusAI/finbert"
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading {model_name}. GPU device={{device}}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=128,
        device=device,
    )

    pos, neg, neu = [], [], []

    for start in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = texts[start:start + FINBERT_BATCH_SIZE]
        preds = clf(batch)

        for item in preds:
            d = {p["label"].lower(): p["score"] for p in item}
            pos.append(d.get("positive", 0.0))
            neg.append(d.get("negative", 0.0))
            neu.append(d.get("neutral", 0.0))

        if start % (FINBERT_BATCH_SIZE * 25) == 0:
            print(f"FinBERT processed {min(start + FINBERT_BATCH_SIZE, len(texts))}/{{len(texts)}}")

    out.loc[idx, "finbert_positive"] = pos
    out.loc[idx, "finbert_negative"] = neg
    out.loc[idx, "finbert_neutral"] = neu
    out.loc[idx, "finbert_score"] = out.loc[idx, "finbert_positive"] - out.loc[idx, "finbert_negative"]

    # Fill remaining NaNs (if limit was used and not all rows processed) with 0.0
    out[["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score"]] =         out[["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score"]].fillna(0.0)

    out.to_csv(cache_path, index=False) # Cache the scores for future runs
    print(f"Saved FinBERT cache to {cache_path}")

    return out

def prepare_tweets(tweets: pd.DataFrame, run_finbert: bool = False, finbert_limit: int = None) -> pd.DataFrame:
    df = tweets.copy()
    df = df.rename(columns={
        "Stock Name": "ticker",
        "Company Name": "company",
        "Tweet": "text"
    })
    df["timestamp"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["clean_text"] = df["text"].apply(clean_text)

    df = df.dropna(subset=["timestamp", "ticker", "clean_text"])
    df = df[df["clean_text"].str.len() >= 5]
    df = df.drop_duplicates(subset=["date", "ticker", "clean_text"])

    print("Tweets after cleaning:", df.shape)

    df = add_vader_textblob(df, "clean_text")

    if run_finbert:
        df = add_finbert_scores(df, "clean_text", limit=finbert_limit)
    else:
        for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score"]:
            df[col] = 0.0

    return df

def aggregate_daily_sentiment(tweet_df: pd.DataFrame) -> pd.DataFrame:
    sentiment_cols = [
        "vader_negative", "vader_neutral", "vader_positive", "vader_compound",
        "textblob_polarity", "textblob_subjectivity",
        "finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score",
    ]

    grouped = tweet_df.groupby(["ticker", "date"])

    mean_df = grouped[sentiment_cols].mean().reset_index()
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in sentiment_cols})

    std_df = grouped[sentiment_cols].std().reset_index()
    std_df = std_df.rename(columns={c: f"{c}_std" for c in sentiment_cols})

    count_df = grouped.size().reset_index(name="tweet_count")

    daily = mean_df.merge(std_df, on=["ticker", "date"], how="left")
    daily = daily.merge(count_df, on=["ticker", "date"], how="left")
    daily = daily.fillna(0.0)

    daily["tweet_count_log"] = np.log1p(daily["tweet_count"])
    daily["finbert_strength"] = daily["finbert_positive_mean"] - daily["finbert_negative_mean"]
    daily["vader_strength"] = daily["vader_compound_mean"].abs()
    daily["textblob_strength"] = daily["textblob_polarity_mean"].abs()

    print("Daily sentiment records:", daily.shape)

    return daily

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan) # Avoid division by zero
    return 100 - (100 / (1 + rs))

def prepare_market_data(prices: pd.DataFrame, target_threshold: float = 0.005) -> pd.DataFrame:
    df = prices.copy()
    df = df.rename(columns={
        "Stock Name": "ticker",
        "Date": "date"
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    df = df.dropna(subset=["date", "ticker", "Close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["daily_return"] = df.groupby("ticker")["Close"].pct_change()
    df["log_return"] = df.groupby("ticker")["Close"].transform(lambda s: np.log(s / s.shift(1)))
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    df["volume_change"] = df.groupby("ticker")["Volume"].pct_change()

    df["volatility_5"] = df.groupby("ticker")["daily_return"].transform(lambda s: s.rolling(5).std())
    df["volatility_10"] = df.groupby("ticker")["daily_return"].transform(lambda s: s.rolling(10).std())

    df["ma_5"] = df.groupby("ticker")["Close"].transform(lambda s: s.rolling(5).mean())
    df["ma_10"] = df.groupby("ticker")["Close"].transform(lambda s: s.rolling(10).mean())
    df["ma_20"] = df.groupby("ticker")["Close"].transform(lambda s: s.rolling(20).mean())

    df["ma_ratio_5"] = df["Close"] / df["ma_5"] - 1
    df["ma_ratio_10"] = df["Close"] / df["ma_10"] - 1
    df["ma_ratio_20"] = df["Close"] / df["ma_20"] - 1

    df["momentum_5"] = df.groupby("ticker")["Close"].pct_change(5)
    df["momentum_10"] = df.groupby("ticker")["Close"].pct_change(10)

    df["rsi_14"] = df.groupby("ticker")["Close"].transform(calculate_rsi)

    df["ema_12"] = df.groupby("ticker")["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["ema_26"] = df.groupby("ticker")["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df.groupby("ticker")["Close"].transform(lambda s: s.rolling(20).mean())
    df["bb_std"] = df.groupby("ticker")["Close"].transform(lambda s: s.rolling(20).std())
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    market_return = df.groupby("date")["daily_return"].mean().rename("market_return")
    df = df.merge(market_return, on="date", how="left")

    df["next_return"] = df.groupby("ticker")["daily_return"].shift(-1)

    # Define classification target (0 for down, 1 for up, NaN for neutral/noisy)
    df["target_direction"] = np.nan
    df.loc[df["next_return"] > target_threshold, "target_direction"] = 1
    df.loc[df["next_return"] < -target_threshold, "target_direction"] = 0

    return df

def merge_features(market_df: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    df = market_df.merge(daily_sent, on=["ticker", "date"], how="left")

    sentiment_like_cols = [
        c for c in df.columns
        if c.startswith("vader_")
        or c.startswith("textblob_")
        or c.startswith("finbert_")
        or c in ["tweet_count", "tweet_count_log", "finbert_strength", "vader_strength", "textblob_strength"]
    ]

    for col in sentiment_like_cols:
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    lag_cols = [
        "daily_return", "log_return", "volume_change",
        "volatility_5", "volatility_10",
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
        "momentum_5", "momentum_10",
        "rsi_14", "macd", "macd_hist",
        "bb_width", "bb_position",
        "market_return",
        "tweet_count_log",
        "vader_compound_mean",
        "textblob_polarity_mean",
        "finbert_score_mean",
        "finbert_strength"
    ]

    lag_cols = [c for c in lag_cols if c in df.columns]

    for col in lag_cols:
        for lag in [1, 2, 3]:
            df[f"{col}_lag_{lag}"] = df.groupby("ticker")[col].shift(lag)

    # Event-day feature: high tweet-volume day within each ticker
    df["tweet_count_q75"] = df.groupby("ticker")["tweet_count"].transform(lambda s: s.quantile(0.75))
    df["event_day"] = (df["tweet_count"] >= df["tweet_count_q75"]).astype(int)

    # One-hot ticker encoding
    ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker", drop_first=True)
    df = pd.concat([df, ticker_dummies], axis=1)

    df = df.replace([np.inf, -np.inf], np.nan)

    required_for_target = [
        "target_direction", # This column comes from prepare_market_data
        "daily_return",
        "volatility_5",
        "ma_ratio_10",
        "rsi_14",
        "macd",
        "bb_width"
    ]
    # Drop rows where critical columns have NaNs after lagging/merging
    df = df.dropna(subset=required_for_target)

    # Remove neutral/noisy target rows (already handled in prepare_market_data but re-check after merge)
    df = df.dropna(subset=["target_direction"])
    df["target_direction"] = df["target_direction"].astype(int)

    print("Final modeling dataset:", df.shape)
    print("Positive strong-movement ratio:", df["target_direction"].mean())

    return df

def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    ticker_cols = [c for c in df.columns if c.startswith("ticker_")]

    technical_cols = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "daily_return", "log_return", "intraday_range", "volume_change",
        "volatility_5", "volatility_10",
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
        "momentum_5", "momentum_10",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_position", "market_return",
        "daily_return_lag_1", "daily_return_lag_2", "daily_return_lag_3",
        "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
        "volume_change_lag_1", "volume_change_lag_2", "volume_change_lag_3",
        "volatility_5_lag_1", "volatility_5_lag_2", "volatility_5_lag_3",
        "rsi_14_lag_1", "rsi_14_lag_2", "rsi_14_lag_3",
        "macd_lag_1", "macd_lag_2", "macd_lag_3",
        "bb_width_lag_1", "bb_width_lag_2", "bb_width_lag_3",
        "market_return_lag_1", "market_return_lag_2", "market_return_lag_3",
        "event_day"
    ] + ticker_cols

    vader_cols = [
        c for c in df.columns
        if c.startswith("vader_")
    ] + [
        "tweet_count", "tweet_count_log",
        "tweet_count_log_lag_1", "tweet_count_log_lag_2", "tweet_count_log_lag_3"
    ]

    textblob_cols = [
        c for c in df.columns
        if c.startswith("textblob_")
    ]

    finbert_cols = [
        c for c in df.columns
        if c.startswith("finbert_")
    ]

    def keep(cols: List[str]) -> List[str]:
        return [c for c in cols if c in df.columns]

    return {
        "technical_only": keep(technical_cols),
        "sentiment_only_vader_textblob": keep(vader_cols + textblob_cols),
        "sentiment_only_finbert": keep(finbert_cols),
        "technical_plus_vader": keep(technical_cols + vader_cols),
        "technical_plus_textblob": keep(technical_cols + textblob_cols),
        "technical_plus_finbert": keep(technical_cols + finbert_cols),
        "technical_plus_vader_textblob": keep(technical_cols + vader_cols + textblob_cols),
        "full_all_features": keep(technical_cols + vader_cols + textblob_cols + finbert_cols),
    }

def make_date_splits(df: pd.DataFrame, n_folds: int = N_FOLDS, min_train_ratio: float = MIN_TRAIN_DATES_RATIO) -> List[Tuple]:
    dates = np.array(sorted(df["date"].unique()))
    n_dates = len(dates)

    min_train = int(n_dates * min_train_ratio)
    remaining = n_dates - min_train
    fold_size = max(1, remaining // n_folds)

    splits = []

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = n_dates if fold == n_folds - 1 else min(n_dates, test_start + fold_size)

        if test_start >= n_dates or test_end <= test_start:
            continue

        train_dates = set(dates[:train_end])
        test_dates = set(dates[test_start:test_end])

        train_idx = df.index[df["date"].isin(train_dates)].to_numpy()
        test_idx = df.index[df["date"].isin(test_dates)].to_numpy()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append(
            (
                train_idx,
                test_idx,
                dates[0],
                dates[train_end - 1],
                dates[test_start],
                dates[test_end - 1]
            )
        )

    return splits

# =======================================================
# MODELING FUNCTIONS (CLASSIFICATION & REGRESSION)
# =======================================================

def classification_models() -> Dict[str, Pipeline]:
    return {
        "naive_most_frequent": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "svc_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }

def regression_models() -> Dict[str, Pipeline]:
    return {
        "naive_mean": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean"))
        ]),
        "random_forest_reg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "xgboost_reg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "svr_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.001)),
        ]),
    }

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan

def evaluate_classification(df: pd.DataFrame, feature_sets: Dict[str, List[str]], out_dir: Path) -> pd.DataFrame:
    splits = make_date_splits(df)
    records = []

    print(f"Number of time-aware folds: {len(splits)}")

    for fs_name, features in feature_sets.items():
        if len(features) == 0:
            continue

        X = df[features].astype(float)
        y = df["target_direction"].astype(int)

        print(f"
Running feature set: {fs_name}, features={{len(features)}}")

        for model_name, base_model in classification_models().items():
            for fold, split in enumerate(splits, start=1):
                train_idx, test_idx, train_start, train_end, test_start, test_end = split

                model = clone(base_model)

                X_train = X.loc[train_idx]
                X_test = X.loc[test_idx]
                y_train = y.loc[train_idx]
                y_test = y.loc[test_idx]

                # XGBoost class imbalance handling (if model is XGBoost, set parameter on the 'model' step of the pipeline)
                if model_name == "xgboost":
                    pos = int((y_train == 1).sum())
                    neg = int((y_train == 0).sum())
                    scale_pos_weight = neg / max(pos, 1)
                    model.set_params(model__scale_pos_weight=scale_pos_weight)
                # LogisticRegression and SVC handle class_weight directly in their constructor within the pipeline

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred

                records.append({
                    "feature_set": fs_name,
                    "model": model_name,
                    "fold": fold,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "auc": safe_auc(y_test, y_prob),
                    "mcc": matthews_corrcoef(y_test, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
                })

    return pd.DataFrame(records)

def evaluate_regression(df: pd.DataFrame, feature_sets: Dict[str, List[str]], out_dir: Path) -> pd.DataFrame:
    splits = make_date_splits(df)
    records = []
    y_col = "next_return"

    print(f"Number of time-aware folds: {len(splits)}")

    for fs_name, features in feature_sets.items():
        if len(features) == 0:
            continue

        X = df[features].astype(float)
        y = df[y_col].astype(float)

        print(f"
Running feature set: {fs_name}, features={{len(features)}}")

        for model_name, base_model in regression_models().items():
            for fold, split in enumerate(splits, start=1):
                train_idx, test_idx, train_start, train_end, test_start, test_end = split

                model = clone(base_model)
                X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                y_train, y_test = y.loc[train_idx], y.loc[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # For regression, we can also evaluate directional accuracy
                pred_dir = (y_pred > 0).astype(int)
                true_dir = (y_test > 0).astype(int)

                records.append({
                    "feature_set": fs_name,
                    "model": model_name,
                    "fold": fold,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "r2": r2_score(y_test, y_pred),
                    "directional_accuracy": accuracy_score(true_dir, pred_dir),
                })

    return pd.DataFrame(records)

def summarize_results(results: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    summary = (
        results
        .groupby(["feature_set", "model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(c) for c in col if c])
        for col in summary.columns
    ]

    summary = summary.sort_values(
        [f"{metrics[0]}_mean", "auc_mean"] if "auc" in metrics else f"{metrics[0]}_mean",
        ascending=False
    )

    return summary

def plot_top_results(summary: pd.DataFrame, out_dir: Path, title_prefix: str, metric: str = "f1_mean", top_n: int = 20) -> None:
    plot_df = summary.sort_values(metric, ascending=False).head(top_n)
    labels = plot_df["feature_set"] + " | " + plot_df["model"]

    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], plot_df[metric][::-1])
    plt.xlabel(metric.replace("_", " "))
    plt.title(f"{title_prefix} Top Classification Configurations")
    plt.tight_layout()

    path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_top_classification_results.png"
    plt.savefig(path, dpi=300)
    plt.show()

    print(f"Saved: {path}")

def plot_ablation_xgboost(summary: pd.DataFrame, out_dir: Path, title_prefix: str, metric: str = "f1_mean") -> None:
    plot_df = summary[summary["model"] == "xgboost"].copy()

    if plot_df.empty:
        print(f"No XGBoost results to plot for {title_prefix} ablation study.")
        return

    plot_df = plot_df.sort_values(metric, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature_set"], plot_df[metric])
    plt.xlabel(metric.replace("_", " "))
    plt.title(f"{title_prefix} Ablation Study with XGBoost")
    plt.tight_layout()

    path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_xgboost_ablation.png"
    plt.savefig(path, dpi=300)
    plt.show()

    print(f"Saved: {path}")

def run_shap_xgboost(df: pd.DataFrame, features: List[str], out_dir: Path, title_prefix: str) -> None:
    try:
        import shap

        splits = make_date_splits(df)
        if not splits:
            print("SHAP skipped: Not enough data for time-aware splits.")
            return

        # Use the last fold for SHAP explanation
        train_idx, test_idx, *_ = splits[-1]

        X_train = df.loc[train_idx, features].astype(float)
        y_train = df.loc[train_idx, "target_direction"].astype(int)
        X_test = df.loc[test_idx, features].astype(float)

        # Impute missing values before training XGBoost and SHAP
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=features,
            index=X_train.index
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=features,
            index=X_test.index
        )

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = neg / max(pos, 1)

        model = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )

        model.fit(X_train_imp, y_train)

        # Sample X_test for SHAP to avoid excessive computation
        sample = X_test_imp.sample(min(1000, len(X_test_imp)), random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False, max_display=25)
        plt.tight_layout()

        path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_shap_xgboost_full_features.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved SHAP figure to: {path}")

        # Also save feature importance table
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": importance
        }).sort_values("mean_abs_shap", ascending=False)

        importance_path = out_dir / f"{title_prefix.lower().replace(' ', '_')}_shap_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        print(f"
Top 30 SHAP features for {title_prefix}:")
        print(importance_df.head(30).to_string(index=False))
        print(f"Saved SHAP importance table to: {importance_path}")

    except Exception as e:
        print(f"SHAP skipped for {title_prefix} because: {e}")

# =======================================================
# MAIN EXECUTION FUNCTIONS FOR EACH VERSION
# =======================================================

def run_version1_analysis():
    print("
=======================================================")
    print("Starting Version 1 Analysis")
    print("=======================================================")

    zip_path = ensure_zip_exists(ZIP_PATH)
    extract_zip(zip_path, RAW_DIR_V1)

    tweets_raw, prices_raw, phrasebank = load_raw_data(RAW_DIR_V1)

    dataset_facts = {
        "tweets_rows_raw": len(tweets_raw),
        "tweets_unique_tickers": tweets_raw["Stock Name"].nunique(),
        "tweets_date_min": str(pd.to_datetime(tweets_raw["Date"], utc=True).min()),
        "tweets_date_max": str(pd.to_datetime(tweets_raw["Date"], utc=True).max()),
        "price_rows_raw": len(prices_raw),
        "price_unique_tickers": prices_raw["Stock Name"].nunique(),
        "price_date_min": str(prices_raw["Date"].min()),
        "price_date_max": str(prices_raw["Date"].max()),
        "financial_phrasebank_rows": len(phrasebank),
    }

    print("
Preparing tweet sentiment features...")
    tweet_df = prepare_tweets(tweets_raw, run_finbert=RUN_FINBERT, finbert_limit=MAX_TWEETS_FOR_FINBERT)
    daily_sent = aggregate_daily_sentiment(tweet_df)

    print("
Preparing market features...")
    market_df = prepare_market_data(prices_raw, target_threshold=TARGET_THRESHOLD)
    final_df = merge_features(market_df, daily_sent)

    dataset_facts.update({
        "tweets_rows_after_cleaning": len(tweet_df),
        "daily_sentiment_rows": len(daily_sent),
        "final_merged_stock_day_rows": len(final_df),
        "final_date_min": str(final_df["date"].min()),
        "final_date_max": str(final_df["date"].max()),
        "final_unique_tickers": final_df["ticker"].nunique(),
        "positive_direction_ratio": float(final_df["target_direction"].mean()),
    })

    facts_df = pd.DataFrame(list(dataset_facts.items()), columns=["item", "value"])
    facts_df.to_csv(OUT_DIR_V1 / "version1_dataset_facts.csv", index=False)

    final_df.to_csv(OUT_DIR_V1 / "version1_final_integrated_dataset.csv", index=False)
    daily_sent.to_csv(OUT_DIR_V1 / "version1_daily_sentiment_features.csv", index=False)

    print("
Dataset facts:")
    print(facts_df.to_string(index=False))

    feature_sets = get_feature_sets(final_df)

    print("
Feature sets:")
    for k, v in feature_sets.items():
        print(f"{k}: {len(v)} features")

    print("
Running classification experiments...")
    cls_results = evaluate_classification(final_df, feature_sets, OUT_DIR_V1)
    cls_results.to_csv(OUT_DIR_V1 / "version1_classification_fold_results.csv", index=False)

    cls_summary = summarize_results(
        cls_results,
        ["f1", "accuracy", "precision", "recall", "auc", "mcc"]
    )
    cls_summary.to_csv(OUT_DIR_V1 / "version1_classification_summary_mean_std.csv", index=False)

    print("
Top classification results (Version 1):")
    print(cls_summary.head(20).to_string(index=False))

    print("
Running regression experiments...")
    reg_results = evaluate_regression(final_df, feature_sets, OUT_DIR_V1)
    reg_results.to_csv(OUT_DIR_V1 / "version1_regression_fold_results.csv", index=False)

    reg_summary = summarize_results(
        reg_results,
        ["directional_accuracy", "mae", "rmse", "r2"]
    )
    reg_summary.to_csv(OUT_DIR_V1 / "version1_regression_summary_mean_std.csv", index=False)

    print("
Top regression results (Version 1):")
    print(reg_summary.head(20).to_string(index=False))

    plot_top_results(cls_summary, OUT_DIR_V1, "Version 1", metric="f1_mean", top_n=15)
    plot_ablation_xgboost(cls_summary, OUT_DIR_V1, "Version 1", metric="f1_mean")

    if "full_all_features" in feature_sets:
        print("
Running SHAP for full XGBoost model (Version 1)...")
        run_shap_xgboost(final_df, feature_sets["full_all_features"], OUT_DIR_V1, "Version 1")

    print("
All Version 1 outputs saved in:")
    print(OUT_DIR_V1)
    print("
Main files for manuscript (Version 1):")
    print("- version1_dataset_facts.csv")
    print("- version1_classification_summary_mean_std.csv")
    print("- version1_regression_summary_mean_std.csv")
    print("- version1_classification_fold_results.csv")
    print("- version1_regression_fold_results.csv")
    print("- version1_ablation_xgboost.png")
    print("- version1_shap_xgboost_full_features.png")

def run_version2_analysis():
    print("
=======================================================")
    print("Starting Version 2 Analysis")
    print("=======================================================")

    zip_path = ensure_zip_exists(ZIP_PATH)
    extract_zip(zip_path, RAW_DIR_V2)

    tweets_raw, prices_raw, phrasebank = load_raw_data(RAW_DIR_V2)

    print("
Preparing tweets and sentiment features...")
    tweets = prepare_tweets(tweets_raw, run_finbert=RUN_FINBERT, finbert_limit=MAX_TWEETS_FOR_FINBERT)

    print("
Aggregating daily sentiment...")
    daily_sent = aggregate_daily_sentiment(tweets)

    print("
Preparing market and technical features...")
    market = prepare_market_data(prices_raw, target_threshold=TARGET_THRESHOLD)

    print("
Merging final dataset...")
    final_df = merge_features(market, daily_sent)

    final_df.to_csv(OUT_DIR_V2 / "version2_final_integrated_dataset.csv", index=False)

    dataset_facts = {
        "tweets_rows_raw": len(tweets_raw),
        "tweets_rows_after_cleaning": len(tweets),
        "tweets_unique_tickers": tweets_raw["Stock Name"].nunique(),
        "price_rows_raw": len(prices_raw),
        "price_unique_tickers": prices_raw["Stock Name"].nunique(),
        "financial_phrasebank_rows": len(phrasebank),
        "daily_sentiment_rows": len(daily_sent),
        "final_modeling_rows_after_threshold": len(final_df),
        "final_unique_tickers": final_df["ticker"].nunique(),
        "final_date_min": final_df["date"].min(),
        "final_date_max": final_df["date"].max(),
        "target_threshold": TARGET_THRESHOLD,
        "positive_class_ratio": final_df["target_direction"].mean(),
        "RUN_FINBERT": RUN_FINBERT,
        "FINBERT_LIMIT": MAX_TWEETS_FOR_FINBERT
    }

    facts_df = pd.DataFrame(
        list(dataset_facts.items()),
        columns=["item", "value"]
    )
    facts_df.to_csv(OUT_DIR_V2 / "version2_dataset_facts.csv", index=False)

    print("
Dataset facts:")
    print(facts_df.to_string(index=False))

    feature_sets = get_feature_sets(final_df)

    print("
Feature sets:")
    for name, cols in feature_sets.items():
        print(name, ":", len(cols), "features")

    print("
Running Version 2 classification experiments...")
    results = evaluate_classification(final_df, feature_sets, OUT_DIR_V2)

    results.to_csv(OUT_DIR_V2 / "version2_classification_fold_results.csv", index=False)

    summary = summarize_results(results, ["f1", "accuracy", "precision", "recall", "auc", "mcc"])
    summary.to_csv(OUT_DIR_V2 / "version2_classification_summary_mean_std.csv", index=False)

    print("
Top Version 2 results:")
    print(summary.head(30).to_string(index=False))

    plot_top_results(summary, OUT_DIR_V2, "Version 2", metric="f1_mean", top_n=20)
    plot_ablation_xgboost(summary, OUT_DIR_V2, "Version 2", metric="f1_mean")

    if "full_all_features" in feature_sets:
        print("
Running SHAP for full XGBoost model (Version 2)...")
        run_shap_xgboost(final_df, feature_sets["full_all_features"], OUT_DIR_V2, "Version 2")

    print("
All Version 2 outputs saved in:")
    print(OUT_DIR_V2)

    print("
Main files for manuscript (Version 2):")
    print("- version2_dataset_facts.csv")
    print("- version2_classification_summary_mean_std.csv")
    print("- version2_classification_fold_results.csv")
    print("- version2_top_classification_results.png")
    print("- version2_xgboost_ablation.png")
    print("- version2_shap_xgboost_full_features.png")


# =======================================================
# MAIN ENTRY POINT
# =======================================================

if __name__ == "__main__":
    # Run Version 1 analysis
    # run_version1_analysis()

    # Run Version 2 analysis
    run_version2_analysis()

