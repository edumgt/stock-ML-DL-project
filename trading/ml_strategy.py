"""
주가 방향성 예측용 머신러닝 전략 모듈.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except Exception:
    _XGB_OK = False


@dataclass
class MLResult:
    model_type: str
    accuracy: float
    report: str
    feature_importance: dict[str, float]


class FeatureBuilder:
    def __init__(self) -> None:
        self.feature_columns = [
            "Returns",
            "MA_Ratio",
            "MA20_Ratio",
            "RSI14",
            "Volatility",
            "Volume_Change",
            "Momentum_5",
            "Momentum_20",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["Close"].astype(float)
        volume = out["Volume"].astype(float)

        out["Returns"] = close.pct_change()
        out["MA5"] = close.rolling(5).mean()
        out["MA20"] = close.rolling(20).mean()
        out["MA_Ratio"] = close / (out["MA5"] + 1e-9)
        out["MA20_Ratio"] = close / (out["MA20"] + 1e-9)
        out["Volatility"] = out["Returns"].rolling(20).std()
        out["Volume_Change"] = volume.pct_change()
        out["Momentum_5"] = close.pct_change(5)
        out["Momentum_20"] = close.pct_change(20)

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        out["RSI14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        return out.dropna()


class MLStrategy:
    def __init__(
        self,
        model_type: str = "rf",
        forward_days: int = 5,
        threshold: float = 0.01,
        model_dir: str = "models",
    ) -> None:
        self.model_type = model_type.lower()
        self.forward_days = forward_days
        self.threshold = threshold
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_builder = FeatureBuilder()
        self._model = None
        self._trained = False

    def _build_target(self, close: pd.Series) -> pd.Series:
        fut_ret = close.shift(-self.forward_days) / close - 1
        target = pd.Series(0, index=close.index)
        target[fut_ret > self.threshold] = 1
        target[fut_ret < -self.threshold] = -1
        return target

    def _init_model(self):
        if self.model_type == "gb":
            return GradientBoostingClassifier(random_state=42)
        if self.model_type == "xgb" and _XGB_OK:
            return XGBClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
            )
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )

    def train(self, df: pd.DataFrame) -> MLResult:
        feat_df = self.feature_builder.transform(df)
        if len(feat_df) <= self.forward_days + 40:
            raise ValueError("학습 데이터가 부족합니다. 더 긴 기간 데이터를 사용하세요.")

        target = self._build_target(feat_df["Close"])
        valid_idx = feat_df.index[:-self.forward_days]
        X = feat_df.loc[valid_idx, self.feature_builder.feature_columns]
        y = target.loc[valid_idx].astype(int)

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self._model = self._init_model()
        self._model.fit(X_train, y_train)
        self._trained = True

        pred = self._model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        report = classification_report(
            y_test,
            pred,
            labels=[-1, 0, 1],
            target_names=["하락(-1)", "보합(0)", "상승(1)"],
            zero_division=0,
        )

        imp = {}
        if hasattr(self._model, "feature_importances_"):
            vals = self._model.feature_importances_
            imp = {
                col: round(float(v), 6)
                for col, v in sorted(
                    zip(self.feature_builder.feature_columns, vals),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

        if self.model_type == "xgb" and not _XGB_OK:
            self.model_type = "rf"

        return MLResult(
            model_type=self.model_type,
            accuracy=acc,
            report=report,
            feature_importance=imp,
        )

    def _latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = self.feature_builder.transform(df)
        if feat_df.empty:
            raise ValueError("예측에 필요한 데이터가 부족합니다.")
        return feat_df.iloc[[-1]][self.feature_builder.feature_columns]

    def predict(self, df: pd.DataFrame) -> str:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습. train()을 먼저 호출하세요.")
        x = self._latest_features(df)
        lbl = int(self._model.predict(x)[0])
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(lbl, "HOLD")

    def predict_proba(self, df: pd.DataFrame) -> dict:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습.")
        x = self._latest_features(df)
        if not hasattr(self._model, "predict_proba"):
            sig = self.predict(df)
            return {"SELL": 1.0 if sig == "SELL" else 0.0, "HOLD": 1.0 if sig == "HOLD" else 0.0, "BUY": 1.0 if sig == "BUY" else 0.0}

        probs = self._model.predict_proba(x)[0]
        classes = list(getattr(self._model, "classes_", [-1, 0, 1]))
        out = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        for cls, p in zip(classes, probs):
            out[mapping.get(int(cls), "HOLD")] = round(float(p), 4)
        return out

    def save(self, filename: str = "model.pkl") -> Path:
        if not self._trained or self._model is None:
            raise RuntimeError("저장할 모델이 없습니다.")
        path = self.model_dir / filename
        joblib.dump(
            {
                "model": self._model,
                "model_type": self.model_type,
                "forward_days": self.forward_days,
                "threshold": self.threshold,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "MLStrategy":
        data = joblib.load(path)
        obj = cls(
            model_type=data.get("model_type", "rf"),
            forward_days=int(data.get("forward_days", 5)),
            threshold=float(data.get("threshold", 0.01)),
        )
        obj._model = data["model"]
        obj._trained = True
        return obj
