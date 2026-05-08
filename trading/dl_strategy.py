"""
LSTM 딥러닝 주가 방향성 예측 모듈 (DL Strategy)
=================================================
LSTM(Long Short-Term Memory) 신경망을 이용해 주가 변동 방향성을 예측합니다.
TensorFlow/Keras 없이도 sklearn MLPClassifier 기반 폴백으로 동작합니다.

주요 클래스:
    LSTMStrategy   – Keras LSTM 모델 (TensorFlow 필요)
    MLPStrategy    – sklearn MLP 폴백 모델 (TensorFlow 없을 때)
    DLStrategy     – 환경에 따라 자동 선택하는 통합 래퍼

사용 예시::

    from trading.dl_strategy import DLStrategy
    from trading.naver_crawler import NaverFinanceCrawler

    crawler  = NaverFinanceCrawler()
    df       = crawler.get_daily_ohlcv("005930", pages=30)  # 삼성전자

    strategy = DLStrategy(model_type="lstm", seq_len=20, forward_days=5)
    result   = strategy.train(df)
    print(f"정확도: {result.accuracy:.4f}")
    signal   = strategy.predict(df)  # "BUY" | "SELL" | "HOLD"

필요 패키지:
    pip install scikit-learn numpy pandas
    pip install tensorflow   # LSTM 사용 시
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TF 로그 억제

# --------------------------------------------------------------------------- #
# 선택적 임포트
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 공통 – 특성/레이블 생성
# --------------------------------------------------------------------------- #

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 기술적 지표 특성 DataFrame 반환 (NaN 제거)"""
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    out    = df.copy()

    for w in [5, 20, 60]:
        out[f"MA{w}_r"] = close / close.rolling(w).mean()

    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()
    out["MACD"]  = out["EMA12"] - out["EMA26"]
    out["MACD_S"] = out["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain_avg = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss_avg = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + gain_avg / (loss_avg + 1e-9))

    bb_m = close.rolling(20).mean()
    bb_s = close.rolling(20).std()
    out["BB_W"] = (4 * bb_s) / (bb_m + 1e-9)
    out["BB_P"] = (close - (bb_m - 2*bb_s)) / (4 * bb_s + 1e-9)

    tr = pd.concat([high-low,
                    (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()

    out["Vol_R"]  = volume / (volume.rolling(5).mean() + 1e-9)
    out["Ret1"]   = close.pct_change(1)
    out["Ret5"]   = close.pct_change(5)
    out["Vol20"]  = out["Ret1"].rolling(20).std()

    return out.dropna()


_FEAT_COLS = [
    "MA5_r", "MA20_r", "MA60_r",
    "MACD", "MACD_S",
    "RSI", "BB_W", "BB_P", "ATR",
    "Vol_R", "Ret1", "Ret5", "Vol20",
]


def _make_labels(close: pd.Series, fwd: int = 5, thr: float = 0.01) -> pd.Series:
    fut = close.shift(-fwd) / close - 1
    lbl = pd.Series(0, index=close.index)
    lbl[fut >  thr] =  1
    lbl[fut < -thr] = -1
    return lbl


# --------------------------------------------------------------------------- #
# 결과 모델
# --------------------------------------------------------------------------- #

@dataclass
class DLResult:
    model_type: str
    accuracy: float
    report: str
    history: dict = field(default_factory=dict)   # epoch별 loss/acc (LSTM)


# --------------------------------------------------------------------------- #
# LSTM 전략 (TensorFlow 필요)
# --------------------------------------------------------------------------- #

class LSTMStrategy:
    """
    Keras LSTM 기반 방향성 예측 모델.

    Parameters
    ----------
    seq_len      : 입력 시퀀스 길이 (일, 기본 20)
    forward_days : 예측 대상 기간 (일, 기본 5)
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
        self._scaler  = StandardScaler() if _SKLEARN_OK else None
        self._model: Optional[keras.Model] = None
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
        X       = feat_df.loc[valid, _FEAT_COLS].values
        y_raw   = labels.loc[valid].values + 1  # {-1,0,1} → {0,1,2}

        if _SKLEARN_OK and self._scaler:
            X = self._scaler.fit_transform(X)

        Xs, ys = self._make_sequences(X, y_raw)
        split  = int(len(Xs) * 0.8)
        X_tr, X_te = Xs[:split], Xs[split:]
        y_tr, y_te = ys[:split], ys[split:]

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
        X = feat_df[_FEAT_COLS].values[-self.seq_len:]
        if _SKLEARN_OK and self._scaler:
            X = self._scaler.transform(X)
        Xs = X[np.newaxis, :, :]
        proba = self._model.predict(Xs, verbose=0)[0]
        idx   = int(proba.argmax())
        return ["SELL", "HOLD", "BUY"][idx]

    def predict_proba(self, df: pd.DataFrame) -> dict:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습.")
        feat_df = _build_features(df)
        if len(feat_df) < self.seq_len:
            return {"하락": 0.0, "보합": 1.0, "상승": 0.0}
        X = feat_df[_FEAT_COLS].values[-self.seq_len:]
        if _SKLEARN_OK and self._scaler:
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
        if _SKLEARN_OK and self._scaler:
            joblib.dump(self._scaler, str(path) + "_scaler.pkl")
        logger.info("LSTM 모델 저장: %s", path)
        return path


# --------------------------------------------------------------------------- #
# MLP 폴백 전략 (TensorFlow 없을 때)
# --------------------------------------------------------------------------- #

class MLPStrategy:
    """
    sklearn MLPClassifier 기반 다층 퍼셉트론 전략.
    TensorFlow 없이도 동작하는 딥러닝 폴백.
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
        self.forward_days   = forward_days
        self.threshold      = threshold
        self.model_dir      = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._scaler  = StandardScaler()
        self._model   = MLPClassifier(
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

        split    = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

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
        X = self._scaler.transform(feat_df.iloc[[-1]][_FEAT_COLS])
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


# --------------------------------------------------------------------------- #
# 통합 래퍼
# --------------------------------------------------------------------------- #

class DLStrategy:
    """
    LSTM(TensorFlow 필요) 또는 MLP(sklearn) 를 자동 선택하는 통합 딥러닝 전략.

    Parameters
    ----------
    model_type   : "lstm" | "mlp" | "auto" (자동 선택)
    seq_len      : LSTM 시퀀스 길이 (lstm 전용, 기본 20)
    forward_days : 예측 대상 기간 (일)
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
