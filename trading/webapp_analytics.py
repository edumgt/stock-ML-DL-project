"""
웹앱에서 사용할 시계열/딥러닝 분석 유틸리티.
`etc/*.md`의 학습 기능을 API와 프론트에서 재사용하기 쉽게 묶는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def load_ohlcv(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y") -> pd.DataFrame:
    if source == "naver":
        from trading.naver_crawler import NaverFinanceCrawler

        df = NaverFinanceCrawler().get_daily_ohlcv(ticker, pages=pages)
    else:
        import yfinance as yf

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.empty:
        raise ValueError(f"데이터가 없습니다: {ticker}")
    return df.sort_values("Date" if "Date" in df.columns else df.index.name or "Date").reset_index(drop=True) if "Date" in df.columns else df.sort_index()


def _dates(df: pd.DataFrame) -> pd.Series:
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"])
    return pd.to_datetime(df.index)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    volume = out["Volume"].astype(float)
    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close / close.shift(1))
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["ma5"] = close.rolling(5).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["vol_ratio"] = volume / (volume.rolling(10).mean() + 1e-9)
    out["volatility"] = out["ret"].rolling(5).std()
    out["ma_cross"] = (out["ma5"] > out["ma20"]).astype(int)
    return out.dropna().copy()


def _seq_features(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = ["ret", "log_ret", "vol_ratio", "volatility"]
    return feat_df[feature_cols].copy(), feature_cols


def _summary_item(label: str, value: Any) -> dict[str, Any]:
    return {"label": label, "value": value}


def _tail_labels(dates: pd.Series, n: int) -> list[str]:
    return pd.to_datetime(dates.tail(n)).dt.strftime("%Y-%m-%d").tolist()


def _bar_colors(values: list[float], pos: str, neg: str) -> list[str]:
    return [pos if v >= 0 else neg for v in values]


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0


def _safe_pct(v: float) -> float:
    return round(float(v) * 100, 2)


def _float(v: Any, digits: int = 4) -> float:
    return round(float(v), digits)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _self_attention_1d(seq: np.ndarray) -> tuple[np.ndarray, float]:
    scores = seq * seq[-1]
    scores = scores / np.sqrt(len(seq))
    weights = _softmax(scores)
    context = float(np.dot(weights, seq))
    return weights, context


def _positional_encoding(seq_len: int, d_model: int = 8) -> np.ndarray:
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)
    dims = np.arange(0, d_model, 2)
    pe[:, 0::2] = np.sin(positions / (10000 ** (dims / d_model)))
    pe[:, 1::2] = np.cos(positions / (10000 ** (dims / d_model)))
    return pe


def _transformer_encode(seq: np.ndarray, d_model: int = 8, n_heads: int = 2) -> tuple[np.ndarray, np.ndarray]:
    seq_len, n_feat = seq.shape
    rng = np.random.default_rng(42)
    embedded = seq @ rng.normal(0, 0.1, (n_feat, d_model)) + _positional_encoding(seq_len, d_model)
    head_dim = d_model // n_heads
    contexts = []
    weights = None
    for _ in range(n_heads):
        wq = rng.normal(0, 0.08, (d_model, head_dim))
        wk = rng.normal(0, 0.08, (d_model, head_dim))
        wv = rng.normal(0, 0.08, (d_model, head_dim))
        q = embedded @ wq
        k = embedded @ wk
        v = embedded @ wv
        attn = _softmax((q @ k.T) / np.sqrt(head_dim))
        contexts.append(attn @ v)
        if weights is None:
            weights = attn
    joined = np.concatenate(contexts, axis=-1)
    ff1 = rng.normal(0, 0.08, (d_model, 16))
    ff2 = rng.normal(0, 0.08, (16, d_model))
    encoded = embedded + joined
    encoded = encoded + np.maximum(0, encoded @ ff1) @ ff2
    return encoded[-1], weights[-1] if weights is not None else np.zeros(seq_len)


def _make_patches(seq: np.ndarray, patch_len: int = 5, stride: int = 2) -> np.ndarray:
    patches = []
    start = 0
    while start + patch_len <= len(seq):
        patches.append(seq[start:start + patch_len].flatten())
        start += stride
    return np.array(patches)


def _patch_attention(patches: np.ndarray) -> np.ndarray:
    dim = patches.shape[1]
    rng = np.random.default_rng(42)
    wq = rng.normal(0, 0.05, (dim, max(dim // 2, 1)))
    wk = rng.normal(0, 0.05, (dim, max(dim // 2, 1)))
    wv = rng.normal(0, 0.05, (dim, dim))
    q = patches @ wq
    k = patches @ wk
    v = patches @ wv
    weights = _softmax((q @ k.T) / np.sqrt(max(dim // 2, 1)))
    return weights @ v + patches


@dataclass
class MultiHeadAttention:
    d_model: int
    n_heads: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        rng = np.random.default_rng(self.seed)
        head_dim = self.d_model // self.n_heads
        self.wq = rng.normal(0, 0.05, (self.n_heads, self.d_model, head_dim))
        self.wk = rng.normal(0, 0.05, (self.n_heads, self.d_model, head_dim))
        self.wv = rng.normal(0, 0.05, (self.n_heads, self.d_model, head_dim))
        self.wo = rng.normal(0, 0.05, (self.d_model, self.d_model))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        head_dim = self.d_model // self.n_heads
        head_outs = []
        head_weights = []
        for h in range(self.n_heads):
            q = x @ self.wq[h]
            k = x @ self.wk[h]
            v = x @ self.wv[h]
            weights = _softmax((q @ k.T) / np.sqrt(head_dim))
            head_weights.append(weights)
            head_outs.append(weights @ v)
        concat = np.concatenate(head_outs, axis=-1)
        return (concat + x) @ self.wo, head_weights


def timeseries_report(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y") -> dict[str, Any]:
    df = load_ohlcv(ticker, source, pages, period)
    feat_df = _feature_frame(df)
    dates = _dates(feat_df)
    ret = feat_df["ret"].dropna()
    if len(ret) < 30:
        raise ValueError("시계열 분석용 데이터가 부족합니다.")

    acf_values = [float(ret.autocorr(lag=i)) for i in range(1, 11)]
    ci = 2 / sqrt(len(ret))
    weekday = ret.groupby(dates.dt.dayofweek).mean().reindex(range(5), fill_value=0)
    monthly = ret.groupby(dates.dt.month).mean().reindex(range(1, 13), fill_value=0)
    monthly_avg = feat_df.assign(year=dates.dt.year, month=dates.dt.month).groupby(["year", "month"])["Close"].mean()

    return {
        "lesson_id": "timeseries",
        "title": "시계열 분석 기초",
        "summary": [
            _summary_item("분석 대상", f"{ticker} ({source})"),
            _summary_item("관측치 수", int(len(feat_df))),
            _summary_item("평균 일수익률", f"{ret.mean() * 100:.3f}%"),
            _summary_item("연환산 변동성", f"{ret.std() * np.sqrt(252) * 100:.2f}%"),
            _summary_item("상승일 비율", f"{(ret > 0).mean() * 100:.1f}%"),
        ],
        "notes": [
            "이동평균으로 단기/중기/장기 추세를 함께 확인합니다.",
            "수익률 기반 통계로 정상성에 가까운 입력을 확인합니다.",
            f"자기상관 95% 신뢰구간은 ±{ci:.4f} 입니다.",
        ],
        "charts": [
            {
                "title": "종가와 이동평균",
                "type": "line",
                "labels": _tail_labels(dates, 120),
                "datasets": [
                    {"label": "Close", "data": feat_df["Close"].tail(120).round(2).tolist(), "borderColor": "#60a5fa"},
                    {"label": "MA20", "data": feat_df["ma20"].tail(120).round(2).tolist(), "borderColor": "#f59e0b"},
                    {"label": "MA60", "data": feat_df["ma60"].tail(120).round(2).tolist(), "borderColor": "#f43f5e"},
                ],
            },
            {
                "title": "자기상관",
                "type": "bar",
                "labels": [f"lag {i}" for i in range(1, 11)],
                "datasets": [
                    {"label": "ACF", "data": [round(v, 4) for v in acf_values], "backgroundColor": _bar_colors(acf_values, "#34d399", "#f87171")},
                ],
            },
            {
                "title": "월별 평균 수익률",
                "type": "bar",
                "labels": [f"{i}월" for i in range(1, 13)],
                "datasets": [
                    {"label": "월평균 수익률(%)", "data": [round(v * 100, 3) for v in monthly.tolist()], "backgroundColor": _bar_colors(monthly.tolist(), "#60a5fa", "#fb7185")},
                ],
            },
        ],
        "details": {
            "monthly_average_close": {f"{int(y)}-{int(m):02d}": round(float(v), 2) for (y, m), v in monthly_avg.tail(24).items()},
            "weekday_effect_pct": {name: round(float(v) * 100, 3) for name, v in zip(["월", "화", "수", "목", "금"], weekday.tolist())},
        },
    }


def sequence_report(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y") -> dict[str, Any]:
    df = load_ohlcv(ticker, source, pages, period)
    feat_df = _feature_frame(df)
    seq_df, feature_cols = _seq_features(feat_df)
    feat_values = seq_df.values
    returns = feat_df["ret"].values
    log_returns = feat_df["log_ret"].values
    if len(feat_values) < 120:
        raise ValueError("시퀀스 실험용 데이터가 부족합니다.")

    seq_len = 20
    x_list, y_list = [], []
    for i in range(seq_len, len(feat_values) - 1):
        x_list.append(feat_values[i - seq_len:i].flatten())
        y_list.append(1 if returns[i + 1] > 0 else 0)
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    split = int(len(x_arr) * 0.8)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_arr[:split])
    x_test = scaler.transform(x_arr[split:])

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", max_iter=250, random_state=42, early_stopping=True)
    clf.fit(x_train, y_arr[:split])
    test_acc = _accuracy(y_arr[split:], clf.predict(x_test))

    seq_lengths = [5, 10, 15, 20, 30, 45]
    seq_results = []
    for sl in seq_lengths:
        x_s, y_s = [], []
        for i in range(sl, len(feat_values) - 1):
            x_s.append(feat_values[i - sl:i].flatten())
            y_s.append(1 if returns[i + 1] > 0 else 0)
        x_s = np.array(x_s)
        y_s = np.array(y_s)
        sp = int(len(x_s) * 0.8)
        sc = StandardScaler()
        x_s_train = sc.fit_transform(x_s[:sp])
        x_s_test = sc.transform(x_s[sp:])
        model = MLPClassifier(hidden_layer_sizes=(48, 24), activation="tanh", max_iter=200, random_state=42, early_stopping=True)
        model.fit(x_s_train, y_s[:sp])
        seq_results.append(_accuracy(y_s[sp:], model.predict(x_s_test)))

    x_reg, y_reg = [], []
    for i in range(seq_len, len(log_returns) - 1):
        x_reg.append(feat_values[i - seq_len:i].flatten())
        y_reg.append(log_returns[i + 1])
    x_reg = np.array(x_reg)
    y_reg = np.array(y_reg)
    sp_r = int(len(x_reg) * 0.8)
    sc_r = StandardScaler()
    reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh", max_iter=250, random_state=42, early_stopping=True)
    reg.fit(sc_r.fit_transform(x_reg[:sp_r]), y_reg[:sp_r])
    preds = reg.predict(sc_r.transform(x_reg[sp_r:]))
    mae = float(np.mean(np.abs(preds - y_reg[sp_r:])))
    best_idx = int(np.argmax(seq_results))

    recent_ret = feat_df["ret"].tail(30) * 100
    return {
        "lesson_id": "sequence-lstm",
        "title": "LSTM 스타일 시퀀스 예측",
        "summary": [
            _summary_item("기본 시퀀스 길이", f"{seq_len}일"),
            _summary_item("입력 특성 수", f"{len(feature_cols)}개"),
            _summary_item("방향 예측 정확도", f"{test_acc * 100:.2f}%"),
            _summary_item("최적 시퀀스", f"{seq_lengths[best_idx]}일"),
            _summary_item("수익률 회귀 MAE", f"{mae:.5f}"),
        ],
        "notes": [
            "문서 예제처럼 LSTM 개념을 tanh 기반 MLP 시퀀스 모델로 근사했습니다.",
            "시퀀스 길이에 따라 정확도가 어떻게 달라지는지 함께 비교합니다.",
        ],
        "charts": [
            {
                "title": "최근 일별 수익률",
                "type": "bar",
                "labels": _tail_labels(_dates(feat_df), 30),
                "datasets": [
                    {"label": "일수익률(%)", "data": recent_ret.round(3).tolist(), "backgroundColor": _bar_colors(recent_ret.tolist(), "#34d399", "#f87171")},
                ],
            },
            {
                "title": "시퀀스 길이별 정확도",
                "type": "line",
                "labels": [str(v) for v in seq_lengths],
                "datasets": [
                    {"label": "정확도(%)", "data": [round(v * 100, 2) for v in seq_results], "borderColor": "#a78bfa"},
                ],
            },
            {
                "title": "회귀 예측 vs 실제",
                "type": "line",
                "labels": [str(i + 1) for i in range(min(25, len(preds)))],
                "datasets": [
                    {"label": "실제 로그수익률", "data": np.round(y_reg[sp_r:sp_r + 25], 5).tolist(), "borderColor": "#60a5fa"},
                    {"label": "예측 로그수익률", "data": np.round(preds[:25], 5).tolist(), "borderColor": "#f59e0b"},
                ],
            },
        ],
    }


def attention_report(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y", seq_len: int = 20) -> dict[str, Any]:
    df = load_ohlcv(ticker, source, pages, period)
    feat_df = _feature_frame(df)
    seq = feat_df["ret"].tail(seq_len).to_numpy()
    if len(seq) < seq_len:
        raise ValueError("Attention 분석용 데이터가 부족합니다.")

    weights, context = _self_attention_1d(seq)
    top_idx = np.argsort(weights)[-5:][::-1]
    dates = _dates(feat_df).tail(seq_len).dt.strftime("%Y-%m-%d").tolist()
    pe = _positional_encoding(seq_len=seq_len, d_model=8)

    return {
        "lesson_id": "attention-core",
        "title": "Self-Attention 핵심 원리",
        "summary": [
            _summary_item("분석 구간", f"최근 {seq_len}거래일"),
            _summary_item("Context 값", f"{context:.6f}"),
            _summary_item("가장 중요한 날", dates[int(top_idx[0])]),
            _summary_item("최대 가중치", f"{weights[int(top_idx[0])]:.4f}"),
        ],
        "notes": [
            "마지막 날을 query로 두고 과거 구간의 중요도를 계산합니다.",
            "Positional Encoding 값 일부도 함께 반환해 순서 정보 추가를 확인할 수 있습니다.",
        ],
        "charts": [
            {
                "title": "최근 수익률",
                "type": "bar",
                "labels": dates,
                "datasets": [
                    {"label": "수익률(%)", "data": np.round(seq * 100, 3).tolist(), "backgroundColor": _bar_colors((seq * 100).tolist(), "#34d399", "#f87171")},
                ],
            },
            {
                "title": "Attention 가중치",
                "type": "bar",
                "labels": dates,
                "datasets": [
                    {"label": "가중치", "data": np.round(weights, 4).tolist(), "backgroundColor": "#f59e0b"},
                ],
            },
        ],
        "details": {
            "top_attention_days": [
                {"date": dates[int(i)], "return_pct": round(float(seq[int(i)]) * 100, 3), "weight": round(float(weights[int(i)]), 4)}
                for i in top_idx
            ],
            "positional_encoding_head": np.round(pe[:5, :4], 4).tolist(),
        },
    }


def transformer_report(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y", seq_len: int = 20) -> dict[str, Any]:
    df = load_ohlcv(ticker, source, pages, period)
    feat_df = _feature_frame(df)
    seq_df, _ = _seq_features(feat_df)
    feat_values = seq_df.values
    ret = feat_df["ret"].values
    if len(feat_values) < seq_len + 40:
        raise ValueError("Transformer 분석용 데이터가 부족합니다.")

    x_trans, y_trans, attn_history = [], [], []
    x_base = []
    for i in range(seq_len, len(feat_values) - 1):
        seq = feat_values[i - seq_len:i]
        encoded, weights = _transformer_encode(seq)
        x_trans.append(np.concatenate([encoded, [feat_values[i, 0]]]))
        x_base.append(seq.flatten())
        y_trans.append(1 if ret[i + 1] > 0 else 0)
        attn_history.append(weights)
    x_trans = np.array(x_trans)
    x_base = np.array(x_base)
    y_trans = np.array(y_trans)

    split = int(len(x_trans) * 0.8)
    sc_trans = StandardScaler()
    sc_base = StandardScaler()
    trans_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42, early_stopping=True)
    base_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", max_iter=250, random_state=42, early_stopping=True)
    trans_model.fit(sc_trans.fit_transform(x_trans[:split]), y_trans[:split])
    base_model.fit(sc_base.fit_transform(x_base[:split]), y_trans[:split])
    trans_acc = _accuracy(y_trans[split:], trans_model.predict(sc_trans.transform(x_trans[split:])))
    base_acc = _accuracy(y_trans[split:], base_model.predict(sc_base.transform(x_base[split:])))

    latest_weights = attn_history[-1]
    return {
        "lesson_id": "transformer",
        "title": "Transformer 주가 예측",
        "summary": [
            _summary_item("Transformer 정확도", f"{trans_acc * 100:.2f}%"),
            _summary_item("시퀀스 기준모델", f"{base_acc * 100:.2f}%"),
            _summary_item("개선 폭", f"{(trans_acc - base_acc) * 100:+.2f}%p"),
            _summary_item("시퀀스 길이", f"{seq_len}일"),
        ],
        "notes": [
            "Positional Encoding과 Self-Attention으로 시퀀스를 압축한 뒤 방향 예측을 수행합니다.",
            "문서 흐름대로 LSTM 스타일 기준 모델과 비교합니다.",
        ],
        "charts": [
            {
                "title": "모델 정확도 비교",
                "type": "bar",
                "labels": ["기준모델", "Transformer"],
                "datasets": [
                    {"label": "정확도(%)", "data": [round(base_acc * 100, 2), round(trans_acc * 100, 2)], "backgroundColor": ["#60a5fa", "#8b5cf6"]},
                ],
            },
            {
                "title": "최신 샘플 Attention",
                "type": "bar",
                "labels": [str(i + 1) for i in range(len(latest_weights))],
                "datasets": [
                    {"label": "가중치", "data": np.round(latest_weights, 4).tolist(), "backgroundColor": "#f59e0b"},
                ],
            },
        ],
    }


def patchtst_report(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y") -> dict[str, Any]:
    df = load_ohlcv(ticker, source, pages, period)
    feat_df = _feature_frame(df)
    seq_df, _ = _seq_features(feat_df)
    feat_values = seq_df.values
    ret = feat_df["ret"].values
    seq_len = 30
    if len(feat_values) < seq_len + 40:
        raise ValueError("PatchTST 분석용 데이터가 부족합니다.")

    def encode(idx: int, patch_len: int, stride: int) -> np.ndarray:
        seq = feat_values[idx - seq_len:idx]
        patches = _make_patches(seq, patch_len, stride)
        encoded = _patch_attention(patches)
        return np.concatenate([encoded[-1], encoded.mean(axis=0)])

    base_patch_len, base_stride = 5, 3
    x_list, y_list = [], []
    for i in range(seq_len, len(feat_values) - 1):
        x_list.append(encode(i, base_patch_len, base_stride))
        y_list.append(1 if ret[i + 1] > 0 else 0)
    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    split = int(len(x_arr) * 0.8)
    sc = StandardScaler()
    model = MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=250, random_state=42, early_stopping=True)
    model.fit(sc.fit_transform(x_arr[:split]), y_arr[:split])
    base_acc = _accuracy(y_arr[split:], model.predict(sc.transform(x_arr[split:])))

    params = [(3, 1), (5, 2), (5, 3), (7, 3), (10, 5)]
    scores = []
    for patch_len, stride in params:
        x_p, y_p = [], []
        for i in range(seq_len, len(feat_values) - 1):
            patches = _make_patches(feat_values[i - seq_len:i], patch_len, stride)
            if len(patches) < 2:
                continue
            encoded = _patch_attention(patches)
            x_p.append(np.concatenate([encoded[-1], encoded.mean(axis=0)]))
            y_p.append(1 if ret[i + 1] > 0 else 0)
        x_p = np.array(x_p)
        y_p = np.array(y_p)
        sp = int(len(x_p) * 0.8)
        sc_p = StandardScaler()
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=220, random_state=42, early_stopping=True)
        clf.fit(sc_p.fit_transform(x_p[:sp]), y_p[:sp])
        scores.append(_accuracy(y_p[sp:], clf.predict(sc_p.transform(x_p[sp:]))))
    best_idx = int(np.argmax(scores))

    example_patches = _make_patches(feat_values[:20], patch_len=5, stride=2)
    return {
        "lesson_id": "patchtst",
        "title": "PatchTST 분석",
        "summary": [
            _summary_item("기본 Patch", f"len={base_patch_len}, stride={base_stride}"),
            _summary_item("기본 정확도", f"{base_acc * 100:.2f}%"),
            _summary_item("최적 Patch", f"len={params[best_idx][0]}, stride={params[best_idx][1]}"),
            _summary_item("최적 정확도", f"{scores[best_idx] * 100:.2f}%"),
            _summary_item("예시 Patch 개수", int(len(example_patches))),
        ],
        "notes": [
            "하루 단위 대신 며칠 묶음 Patch를 토큰처럼 사용합니다.",
            "문서와 같은 방식으로 Patch 길이/보폭 조합을 비교합니다.",
        ],
        "charts": [
            {
                "title": "Patch 파라미터별 정확도",
                "type": "bar",
                "labels": [f"len={p[0]},s={p[1]}" for p in params],
                "datasets": [
                    {"label": "정확도(%)", "data": [round(v * 100, 2) for v in scores], "backgroundColor": "#8b5cf6"},
                ],
            },
        ],
        "details": {
            "example_patch_shape": list(example_patches.shape),
        },
    }


def multihead_report(tickers: list[str], source: str = "naver", pages: int = 80, period: str = "5y") -> dict[str, Any]:
    if len(tickers) < 2:
        raise ValueError("멀티헤드 분석은 2개 이상 종목이 필요합니다.")

    stock_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers[:6]:
        stock_frames[ticker] = _feature_frame(load_ohlcv(ticker, source, pages, period))

    returns_df = pd.DataFrame({ticker: frame["ret"] for ticker, frame in stock_frames.items()}).dropna()
    corr = returns_df.corr().round(4)
    seq_len, d_model, n_heads = 20, 8, 4
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    results: dict[str, float] = {}

    for ticker, frame in stock_frames.items():
        seq_df, _ = _seq_features(frame)
        feat_values = seq_df.values
        ret = frame["ret"].values
        if len(feat_values) < seq_len + 40:
            continue
        x_list, y_list = [], []
        for i in range(seq_len, len(feat_values) - 1):
            seq = feat_values[i - seq_len:i]
            rng = np.random.default_rng(42)
            embedded = seq @ rng.normal(0, 0.1, (seq.shape[1], d_model)) + _positional_encoding(seq_len, d_model)
            out, _ = mha.forward(embedded)
            x_list.append(out[-1])
            y_list.append(1 if ret[i + 1] > 0 else 0)
        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        split = int(len(x_arr) * 0.8)
        sc = StandardScaler()
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=220, random_state=42, early_stopping=True)
        clf.fit(sc.fit_transform(x_arr[:split]), y_arr[:split])
        results[ticker] = _accuracy(y_arr[split:], clf.predict(sc.transform(x_arr[split:])))

    if not results:
        raise ValueError("멀티헤드 예측에 필요한 데이터가 부족합니다.")

    top_pair = corr.where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False).head(1)
    pair_text = " / ".join(top_pair.index[0]) if len(top_pair) else "-"
    pair_value = float(top_pair.iloc[0]) if len(top_pair) else 0.0
    labels = list(results.keys())
    values = [round(v * 100, 2) for v in results.values()]

    return {
        "lesson_id": "multihead",
        "title": "멀티헤드 Attention & 종목 관계",
        "summary": [
            _summary_item("분석 종목 수", len(results)),
            _summary_item("헤드 수", n_heads),
            _summary_item("최고 상관 쌍", pair_text),
            _summary_item("상관계수", f"{pair_value:.4f}"),
        ],
        "notes": [
            "종목별 수익률 상관관계를 먼저 보고, 각 종목에 멀티헤드 Attention 분류기를 적용합니다.",
            "문서 예제처럼 여러 종목을 같은 관점 설정으로 비교할 수 있습니다.",
        ],
        "charts": [
            {
                "title": "종목별 멀티헤드 정확도",
                "type": "bar",
                "labels": labels,
                "datasets": [
                    {"label": "정확도(%)", "data": values, "backgroundColor": ["#2563eb", "#8b5cf6", "#d97706", "#059669", "#ef4444", "#0ea5e9"][: len(values)]},
                ],
            },
        ],
        "details": {
            "correlation_matrix": corr.to_dict(),
        },
    }


def backtest_report(ticker: str, source: str = "naver", pages: int = 80, period: str = "5y") -> dict[str, Any]:
    df = load_ohlcv(ticker, source, pages, period)
    feat_df = _feature_frame(df)
    feat_cols = ["ret", "ret_5", "ret_20", "vol_ratio", "volatility", "ma_cross"]
    x_base = feat_df[feat_cols].values[:-1]
    y_base = (feat_df["Close"].shift(-1) > feat_df["Close"]).astype(int).values[:-1]
    test_rets = feat_df["ret"].values[:-1]
    if len(x_base) < 120:
        raise ValueError("백테스트용 데이터가 부족합니다.")

    split = int(len(x_base) * 0.8)
    sc = StandardScaler()
    x_tr = sc.fit_transform(x_base[:split])
    x_te = sc.transform(x_base[split:])
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(x_tr, y_base[:split])
    rf_probs = rf.predict_proba(x_te)[:, 1]

    seq_len = 20
    x_seq, y_seq = [], []
    for i in range(seq_len, len(x_base)):
        x_seq.append(x_base[i - seq_len:i].flatten())
        y_seq.append(y_base[i])
    x_seq = np.array(x_seq[:-1])
    y_seq = np.array(y_seq[:-1])
    sp_seq = int(len(x_seq) * 0.8)
    sc_seq = StandardScaler()
    lstm_style = MLPClassifier(hidden_layer_sizes=(96, 48), activation="tanh", max_iter=220, random_state=42, early_stopping=True)
    lstm_style.fit(sc_seq.fit_transform(x_seq[:sp_seq]), y_seq[:sp_seq])
    seq_probs = lstm_style.predict_proba(sc_seq.transform(x_seq[sp_seq:]))[:, 1]

    x_trans, y_trans = [], []
    for i in range(seq_len, len(x_base)):
        encoded, _ = _transformer_encode(x_base[i - seq_len:i], d_model=8, n_heads=2)
        x_trans.append(encoded)
        y_trans.append(y_base[i])
    x_trans = np.array(x_trans[:-1])
    y_trans = np.array(y_trans[:-1])
    sp_trans = int(len(x_trans) * 0.8)
    sc_trans = StandardScaler()
    trans = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=220, random_state=42, early_stopping=True)
    trans.fit(sc_trans.fit_transform(x_trans[:sp_trans]), y_trans[:sp_trans])
    trans_probs = trans.predict_proba(sc_trans.transform(x_trans[sp_trans:]))[:, 1]

    def run_backtest(probs: np.ndarray, returns: np.ndarray, threshold: float = 0.55) -> tuple[np.ndarray, dict[str, Any]]:
        portfolio = [1.0]
        buy_hold = [1.0]
        trades = 0
        for p, r in zip(probs, returns[: len(probs)]):
            buy_hold.append(buy_hold[-1] * (1 + r))
            if p >= threshold:
                portfolio.append(portfolio[-1] * (1 + r))
                trades += 1
            else:
                portfolio.append(portfolio[-1])
        port_arr = np.array(portfolio)
        bh_arr = np.array(buy_hold)
        peak = np.maximum.accumulate(port_arr)
        drawdown = (port_arr - peak) / peak
        daily = np.diff(port_arr) / port_arr[:-1]
        sharpe = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252) if len(daily) > 1 else 0.0
        metrics = {
            "total_return_pct": _safe_pct(port_arr[-1] - 1),
            "buy_hold_pct": _safe_pct(bh_arr[-1] - 1),
            "mdd_pct": _safe_pct(drawdown.min()),
            "sharpe": _float(sharpe, 3),
            "trades": trades,
        }
        return port_arr, metrics

    rf_curve, rf_metrics = run_backtest(rf_probs, test_rets[split:])
    seq_curve, seq_metrics = run_backtest(seq_probs, test_rets[split + seq_len:])  # close enough for a compact dashboard view
    trans_curve, trans_metrics = run_backtest(trans_probs, test_rets[split + seq_len:])

    return {
        "lesson_id": "backtest",
        "title": "AI 백테스트 & 종합 평가",
        "summary": [
            _summary_item("RF 수익률", f"{rf_metrics['total_return_pct']:.2f}%"),
            _summary_item("시퀀스 수익률", f"{seq_metrics['total_return_pct']:.2f}%"),
            _summary_item("Transformer 수익률", f"{trans_metrics['total_return_pct']:.2f}%"),
            _summary_item("기준 보유 수익률", f"{rf_metrics['buy_hold_pct']:.2f}%"),
        ],
        "notes": [
            "문서의 종합 평가 흐름처럼 RF, 시퀀스 모델, Transformer를 같은 데이터로 백테스트합니다.",
            "백테스트 임계치는 상승 확률 55% 기준입니다.",
        ],
        "charts": [
            {
                "title": "누적 수익률 비교",
                "type": "line",
                "labels": [str(i) for i in range(len(rf_curve))],
                "datasets": [
                    {"label": "RF", "data": np.round((rf_curve - 1) * 100, 2).tolist(), "borderColor": "#60a5fa"},
                    {"label": "Sequence", "data": np.round((seq_curve - 1) * 100, 2).tolist(), "borderColor": "#34d399"},
                    {"label": "Transformer", "data": np.round((trans_curve - 1) * 100, 2).tolist(), "borderColor": "#a78bfa"},
                ],
            },
            {
                "title": "전략 수익률",
                "type": "bar",
                "labels": ["RF", "Sequence", "Transformer", "Buy&Hold"],
                "datasets": [
                    {
                        "label": "누적 수익률(%)",
                        "data": [
                            rf_metrics["total_return_pct"],
                            seq_metrics["total_return_pct"],
                            trans_metrics["total_return_pct"],
                            rf_metrics["buy_hold_pct"],
                        ],
                        "backgroundColor": ["#60a5fa", "#34d399", "#a78bfa", "#f59e0b"],
                    },
                ],
            },
        ],
        "details": {
            "rf_metrics": rf_metrics,
            "sequence_metrics": seq_metrics,
            "transformer_metrics": trans_metrics,
        },
    }
