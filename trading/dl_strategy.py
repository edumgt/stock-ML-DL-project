"""LSTM / MLP 딥러닝 주가 방향성 예측 모듈.

주요 클래스
-----------
LSTMStrategy   – Keras LSTM 모델 (TensorFlow 필요)
MLPStrategy    – sklearn MLP 폴백 모델 (TensorFlow 없을 때)
DLStrategy     – 환경에 따라 자동 선택하는 통합 래퍼

학습 절차 (LSTM/MLP 공통)
--------------------------
1. OHLCV → 20가지 기술적 지표 산출
2. 시계열 순서를 유지해 80/20 분할
3. **훈련 데이터에만** StandardScaler fit → 테스트 데이터는 transform만
4. (LSTM) seq_len 슬라이딩 윈도우 시퀀스 생성
5. EarlyStopping(patience=5) 으로 과적합 조기 차단

사용 예시::

    from trading.dl_strategy import DLStrategy
    df = ...  # OHLCV DataFrame
    strategy = DLStrategy(model_type="mlp", seq_len=20, forward_days=5)
    result   = strategy.train(df)
    signal   = strategy.predict(df)  # "BUY" | "SELL" | "HOLD"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    from tensorflow import keras  # type: ignore
    _TF_OK = True
    logger.debug("TensorFlow %s 감지됨", tf.__version__)
except ImportError:
    _TF_OK = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ── 공통: 기술적 지표 20개 산출 ──────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 20가지 기술적 지표 DataFrame (NaN 행 제거).

    지표 목록
    ---------
    추세  : MA5/20/60 비율
    MACD  : MACD 라인, 시그널, 히스토그램
    오실레이터 : RSI(14), 스토캐스틱(14,3), Williams %R(14)
    밴드  : 볼린저 밴드 폭·위치
    변동성 : ATR(14), 20일 롤링 표준편차
    거래량 : Vol/MA5, Vol/MA20, OBV 변화율
    모멘텀 : 1일·5일·20일 수익률
    """
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float) if "High" in df.columns else close
    low    = df["Low"].astype(float) if "Low" in df.columns else close
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    out    = df.copy()

    # ── 이동평균 비율 ──────────────────────────────────────────────
    for w in [5, 20, 60]:
        out[f"MA{w}_r"] = close / (close.rolling(w).mean() + 1e-9)

    # ── MACD (가격 정규화) ─────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"]   = (ema12 - ema26) / (close + 1e-9)
    out["MACD_S"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_H"] = out["MACD"] - out["MACD_S"]

    # ── RSI(14) ────────────────────────────────────────────────────
    delta    = close.diff()
    gain_avg = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss_avg = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + gain_avg / (loss_avg + 1e-9))

    # ── 스토캐스틱 %K/%D (14, 3) ──────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k    = 100 * (close - low14) / (high14 - low14 + 1e-9)
    out["Stoch_K"] = stoch_k
    out["Stoch_D"] = stoch_k.rolling(3).mean()

    # ── Williams %R (14) ──────────────────────────────────────────
    out["Williams_R"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

    # ── 볼린저 밴드 (20, 2σ) ──────────────────────────────────────
    bb_m = close.rolling(20).mean()
    bb_s = close.rolling(20).std()
    out["BB_W"] = (4 * bb_s) / (bb_m + 1e-9)
    out["BB_P"] = (close - (bb_m - 2 * bb_s)) / (4 * bb_s + 1e-9)

    # ── ATR(14) — 가격 정규화 ─────────────────────────────────────
    tr = pd.concat(
        [high - low,
         (high - close.shift()).abs(),
         (low  - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.rolling(14).mean() / (close + 1e-9)

    # ── 거래량 ────────────────────────────────────────────────────
    out["Vol_R"]      = volume / (volume.rolling(5).mean()  + 1e-9)
    out["Vol_MA20_R"] = volume / (volume.rolling(20).mean() + 1e-9)
    obv = (np.sign(close.diff()) * volume).cumsum()
    out["OBV_R"] = obv.pct_change()

    # ── 수익률 / 변동성 / 모멘텀 ─────────────────────────────────
    out["Ret1"]  = close.pct_change(1)
    out["Ret5"]  = close.pct_change(5)
    out["Vol20"] = out["Ret1"].rolling(20).std()
    out["Mom20"] = close.pct_change(20)

    return out.dropna()


_FEAT_COLS: list[str] = [
    # MA 기반 추세
    "MA5_r", "MA20_r", "MA60_r",
    # MACD
    "MACD", "MACD_S", "MACD_H",
    # 오실레이터
    "RSI", "Stoch_K", "Stoch_D", "Williams_R",
    # 볼린저 밴드
    "BB_W", "BB_P",
    # 변동성
    "ATR", "Vol20",
    # 거래량
    "Vol_R", "Vol_MA20_R", "OBV_R",
    # 수익률 / 모멘텀
    "Ret1", "Ret5", "Mom20",
]  # 총 20개


def _make_labels(close: pd.Series, fwd: int = 5, thr: float = 0.01) -> pd.Series:
    fut = close.shift(-fwd) / close - 1
    lbl = pd.Series(0, index=close.index)
    lbl[fut >  thr] =  1
    lbl[fut < -thr] = -1
    return lbl


# ── 결과 모델 ─────────────────────────────────────────────────────────────────

@dataclass
class DLResult:
    model_type: str
    accuracy: float
    report: str
    history: dict = field(default_factory=dict)


# ── LSTM 전략 (TensorFlow 필요) ───────────────────────────────────────────────

class LSTMStrategy:
    """Keras LSTM 기반 방향성 예측 모델.

    Parameters
    ----------
    seq_len      : 입력 시퀀스 길이 (봉, 기본 20)
    forward_days : 예측 대상 기간 (봉, 기본 5)
    threshold    : 상승/하락 판정 임계값 (기본 1%)
    """

    def __init__(
        self,
        seq_len: int = 20,
        forward_days: int = 5,
        threshold: float = 0.01,
        model_dir: str = "models",
    ) -> None:
        if not _TF_OK:
            raise ImportError("pip install tensorflow 을 먼저 실행하세요.")
        self.seq_len      = seq_len
        self.forward_days = forward_days
        self.threshold    = threshold
        self.model_dir    = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._scaler: Optional["StandardScaler"] = StandardScaler() if _SKLEARN_OK else None
        self._model: Optional["keras.Model"] = None
        self._trained = False

    def _build_model(self, n_features: int) -> "keras.Model":
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_len, n_features)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(3, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, df: pd.DataFrame, epochs: int = 30, batch_size: int = 32) -> DLResult:
        feat_df = _build_features(df)
        labels  = _make_labels(feat_df["Close"], self.forward_days, self.threshold)
        valid   = feat_df.index[:-self.forward_days]
        X_raw   = feat_df.loc[valid, _FEAT_COLS].values
        y_raw   = labels.loc[valid].values + 1  # {-1,0,1} → {0,1,2}

        # ── 시퀀스 생성 (스케일링 전) ─────────────────────────────
        Xs, ys = self._make_sequences(X_raw, y_raw)
        if len(Xs) < 30:
            raise ValueError("시퀀스 생성에 필요한 데이터가 부족합니다.")

        split = int(len(Xs) * 0.8)
        X_tr_raw, X_te_raw = Xs[:split], Xs[split:]
        y_tr, y_te         = ys[:split], ys[split:]

        # ── 스케일링: 훈련 시퀀스에만 fit — 데이터 누수 방지 ─────────
        if _SKLEARN_OK and self._scaler is not None:
            n_tr, seq, feat = X_tr_raw.shape
            n_te            = len(X_te_raw)
            # 2D로 reshape해 fit → transform → 3D 복원
            self._scaler.fit(X_tr_raw.reshape(-1, feat))
            X_tr = self._scaler.transform(X_tr_raw.reshape(-1, feat)).reshape(n_tr, seq, feat)
            X_te = self._scaler.transform(X_te_raw.reshape(-1, feat)).reshape(n_te, seq, feat)
        else:
            X_tr, X_te = X_tr_raw, X_te_raw

        self._model = self._build_model(len(_FEAT_COLS))
        cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        hist = self._model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[cb],
            verbose=0,
        )
        self._trained = True

        preds    = self._model.predict(X_te, verbose=0).argmax(axis=1)
        accuracy = float((preds == y_te).mean())
        report   = classification_report(
            y_te, preds,
            target_names=["하락", "보합", "상승"],
            zero_division=0,
        ) if _SKLEARN_OK else ""

        history = {
            "loss":     [round(v, 4) for v in hist.history.get("loss", [])],
            "val_loss": [round(v, 4) for v in hist.history.get("val_loss", [])],
            "accuracy": [round(v, 4) for v in hist.history.get("accuracy", [])],
        }
        logger.info("LSTM 학습 완료 – 정확도: %.4f", accuracy)
        return DLResult(model_type="lstm", accuracy=accuracy, report=report, history=history)

    def predict(self, df: pd.DataFrame) -> str:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습. train() 을 먼저 호출하세요.")
        feat_df = _build_features(df)
        if len(feat_df) < self.seq_len:
            return "HOLD"
        X = feat_df[_FEAT_COLS].values[-self.seq_len:]  # (seq_len, n_feat)
        if _SKLEARN_OK and self._scaler is not None:
            X = self._scaler.transform(X)               # scaler는 2D (n, n_feat) 기준으로 fit됨
        proba = self._model.predict(X[np.newaxis, :, :], verbose=0)[0]
        return ["SELL", "HOLD", "BUY"][int(proba.argmax())]

    def predict_proba(self, df: pd.DataFrame) -> dict:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습.")
        feat_df = _build_features(df)
        if len(feat_df) < self.seq_len:
            return {"하락": 0.0, "보합": 1.0, "상승": 0.0}
        X = feat_df[_FEAT_COLS].values[-self.seq_len:]
        if _SKLEARN_OK and self._scaler is not None:
            X = self._scaler.transform(X)
        proba = self._model.predict(X[np.newaxis, :, :], verbose=0)[0]
        return {"하락": round(float(proba[0]), 4),
                "보합": round(float(proba[1]), 4),
                "상승": round(float(proba[2]), 4)}

    def save(self, filename: str = "lstm_model") -> Path:
        if self._model is None:
            raise RuntimeError("저장할 모델 없음.")
        path = self.model_dir / filename
        self._model.save(str(path))
        if _SKLEARN_OK and self._scaler is not None:
            joblib.dump(self._scaler, str(path) + "_scaler.pkl")
        logger.info("LSTM 모델 저장: %s", path)
        return path


# ── MLP 폴백 전략 (TensorFlow 없을 때) ───────────────────────────────────────

class MLPStrategy:
    """sklearn MLPClassifier 기반 다층 퍼셉트론 전략.

    TensorFlow 없이도 동작하는 딥러닝 폴백.
    훈련 데이터에만 StandardScaler를 fit해 데이터 누수를 방지합니다.
    """

    def __init__(
        self,
        forward_days: int = 5,
        threshold: float = 0.01,
        hidden_layers: tuple = (128, 64, 32),
        model_dir: str = "models",
    ) -> None:
        if not _SKLEARN_OK:
            raise ImportError("pip install scikit-learn joblib 을 먼저 실행하세요.")
        self.forward_days = forward_days
        self.threshold    = threshold
        self.model_dir    = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._scaler = StandardScaler()
        self._model  = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self._trained = False

    def train(self, df: pd.DataFrame) -> DLResult:
        feat_df = _build_features(df)
        labels  = _make_labels(feat_df["Close"], self.forward_days, self.threshold)
        valid   = feat_df.index[:-self.forward_days]
        X = feat_df.loc[valid, _FEAT_COLS]
        y = labels.loc[valid]

        # 80/20 시계열 분할
        split    = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # 훈련 데이터에만 fit — 테스트는 transform만
        X_tr_sc = self._scaler.fit_transform(X_tr)
        X_te_sc = self._scaler.transform(X_te)

        self._model.fit(X_tr_sc, y_tr)
        self._trained = True

        preds    = self._model.predict(X_te_sc)
        accuracy = float(accuracy_score(y_te, preds))
        report   = classification_report(
            y_te, preds,
            target_names=["하락(-1)", "보합(0)", "상승(1)"],
            zero_division=0,
        )
        logger.info("MLP 학습 완료 – 정확도: %.4f", accuracy)
        return DLResult(model_type="mlp", accuracy=accuracy, report=report)

    def predict(self, df: pd.DataFrame) -> str:
        if not self._trained:
            raise RuntimeError("모델 미학습.")
        feat_df = _build_features(df)
        if feat_df.empty:
            return "HOLD"
        X   = self._scaler.transform(feat_df.iloc[[-1]][_FEAT_COLS])
        lbl = int(self._model.predict(X)[0])
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(lbl, "HOLD")

    def predict_proba(self, df: pd.DataFrame) -> dict:
        if not self._trained:
            raise RuntimeError("모델 미학습.")
        feat_df = _build_features(df)
        if feat_df.empty:
            return {"하락": 0.0, "보합": 1.0, "상승": 0.0}
        X     = self._scaler.transform(feat_df.iloc[[-1]][_FEAT_COLS])
        proba = self._model.predict_proba(X)[0]
        cls   = list(self._model.classes_)
        mp    = {-1: "하락", 0: "보합", 1: "상승"}
        return {mp[int(c)]: round(float(p), 4) for c, p in zip(cls, proba)}

    def save(self, filename: str = "mlp_model.pkl") -> Path:
        path = self.model_dir / filename
        joblib.dump({"model": self._model, "scaler": self._scaler,
                     "config": {"forward_days": self.forward_days,
                                "threshold": self.threshold}}, path)
        return path

    @classmethod
    def load(cls, path: str) -> "MLPStrategy":
        data = joblib.load(path)
        cfg  = data.get("config", {})
        s    = cls(forward_days=cfg.get("forward_days", 5),
                   threshold=cfg.get("threshold", 0.01))
        s._model   = data["model"]
        s._scaler  = data["scaler"]
        s._trained = True
        return s


# ── 통합 래퍼 ─────────────────────────────────────────────────────────────────

class DLStrategy:
    """LSTM(TensorFlow) 또는 MLP(sklearn)를 자동 선택하는 통합 딥러닝 전략.

    Parameters
    ----------
    model_type   : "lstm" | "mlp" | "auto" (TensorFlow 설치 여부로 자동 결정)
    seq_len      : LSTM 시퀀스 길이 (lstm 전용, 기본 20)
    forward_days : 예측 대상 기간 (봉)
    threshold    : 상승/하락 임계값
    """

    def __init__(
        self,
        model_type: str = "auto",
        seq_len: int = 20,
        forward_days: int = 5,
        threshold: float = 0.01,
        model_dir: str = "models",
    ) -> None:
        mt = model_type.lower()
        if mt == "lstm" or (mt == "auto" and _TF_OK):
            self._impl = LSTMStrategy(
                seq_len=seq_len,
                forward_days=forward_days,
                threshold=threshold,
                model_dir=model_dir,
            )
            self.model_type = "lstm"
        else:
            if not _SKLEARN_OK:
                raise ImportError("pip install scikit-learn joblib 을 먼저 실행하세요.")
            self._impl = MLPStrategy(
                forward_days=forward_days,
                threshold=threshold,
                model_dir=model_dir,
            )
            self.model_type = "mlp"

    def train(self, df: pd.DataFrame, **kwargs) -> DLResult:
        return self._impl.train(df, **kwargs)

    def predict(self, df: pd.DataFrame) -> str:
        return self._impl.predict(df)

    def predict_proba(self, df: pd.DataFrame) -> dict:
        return self._impl.predict_proba(df)

    def save(self, filename: str = "dl_model") -> Path:
        return self._impl.save(filename)
