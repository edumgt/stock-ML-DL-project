"""주가 방향성 예측용 머신러닝 전략 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except Exception:
    _XGB_OK = False


@dataclass
class MLResult:
    model_type: str
    accuracy: float       # 홀드아웃 정확도 (마지막 20%)
    report: str
    feature_importance: dict[str, float]
    cv_scores: list[float] = field(default_factory=list)  # 5-fold Walk-Forward 각 점수
    cv_mean: float = 0.0   # CV 평균 정확도
    cv_std: float = 0.0    # CV 표준편차


class FeatureBuilder:
    """OHLCV → 20가지 기술적 지표 특성 행렬 변환기.

    지표 분류
    ---------
    추세  : 이동평균 비율 (MA5/20/60), Returns
    MACD  : MACD 라인, 시그널, 히스토그램
    오실레이터 : RSI(14), 스토캐스틱(14,3), Williams %R(14)
    밴드  : 볼린저 밴드 폭·위치
    변동성 : ATR(14), 20일 롤링 표준편차
    거래량 : Volume 변화율, Volume/MA20, OBV 변화율
    모멘텀 : 5일·20일 수익률
    """

    def __init__(self) -> None:
        self.feature_columns: list[str] = [
            # ── 추세 ──────────────────────────────────────────────
            "Returns",
            "MA5_Ratio", "MA20_Ratio", "MA60_Ratio",
            # ── MACD ──────────────────────────────────────────────
            "MACD", "MACD_Signal", "MACD_Hist",
            # ── 오실레이터 ─────────────────────────────────────────
            "RSI14",
            "Stoch_K", "Stoch_D",
            "Williams_R",
            # ── 볼린저 밴드 ────────────────────────────────────────
            "BB_Width", "BB_Position",
            # ── 변동성 ────────────────────────────────────────────
            "ATR14", "Volatility",
            # ── 거래량 ────────────────────────────────────────────
            "Volume_Change", "Volume_MA_Ratio", "OBV_Change",
            # ── 모멘텀 ────────────────────────────────────────────
            "Momentum_5", "Momentum_20",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV DataFrame → 기술적 지표 DataFrame 반환 (NaN 행 제거)."""
        out = df.copy()
        close  = out["Close"].astype(float)
        high   = out["High"].astype(float) if "High" in out.columns else close
        low    = out["Low"].astype(float) if "Low" in out.columns else close
        volume = out["Volume"].astype(float) if "Volume" in out.columns else pd.Series(1.0, index=out.index)

        # ── 기초 수익률 ────────────────────────────────────────────
        out["Returns"] = close.pct_change()

        # ── 이동평균 비율 ──────────────────────────────────────────
        ma5  = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        out["MA5_Ratio"]  = close / (ma5  + 1e-9)
        out["MA20_Ratio"] = close / (ma20 + 1e-9)
        out["MA60_Ratio"] = close / (ma60 + 1e-9)

        # ── MACD (가격 정규화로 종목 간 스케일 통일) ─────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = (ema12 - ema26) / (close + 1e-9)
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        out["MACD"]        = macd
        out["MACD_Signal"] = macd_sig
        out["MACD_Hist"]   = macd - macd_sig

        # ── RSI(14) ────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        out["RSI14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

        # ── 스토캐스틱 %K/%D (14, 3) ──────────────────────────────
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        out["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
        out["Stoch_D"] = out["Stoch_K"].rolling(3).mean()

        # ── Williams %R (14) ──────────────────────────────────────
        out["Williams_R"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

        # ── 볼린저 밴드 (20, 2σ) ──────────────────────────────────
        bb_std = close.rolling(20).std()
        out["BB_Width"]    = (4 * bb_std) / (ma20 + 1e-9)
        out["BB_Position"] = (close - (ma20 - 2 * bb_std)) / (4 * bb_std + 1e-9)

        # ── ATR(14) — 가격 정규화 ─────────────────────────────────
        tr = pd.concat(
            [high - low,
             (high - close.shift()).abs(),
             (low  - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        out["ATR14"] = tr.rolling(14).mean() / (close + 1e-9)

        # ── 변동성 ────────────────────────────────────────────────
        out["Volatility"] = out["Returns"].rolling(20).std()

        # ── 거래량 지표 ────────────────────────────────────────────
        out["Volume_Change"]   = volume.pct_change()
        out["Volume_MA_Ratio"] = volume / (volume.rolling(20).mean() + 1e-9)
        obv = (np.sign(close.diff()) * volume).cumsum()
        out["OBV_Change"] = obv.pct_change()

        # ── 모멘텀 ────────────────────────────────────────────────
        out["Momentum_5"]  = close.pct_change(5)
        out["Momentum_20"] = close.pct_change(20)

        return out.dropna()


class MLStrategy:
    """RandomForest / GradientBoosting / XGBoost 기반 방향성 예측 전략.

    학습 절차
    ---------
    1. 20가지 기술적 지표 산출
    2. TimeSeriesSplit 5-fold Walk-Forward 교차검증 → CV 정확도 측정
    3. 80/20 시계열 분할로 최종 모델 학습 → 홀드아웃 정확도 측정
    4. 최신 데이터 기준 방향성 예측 (BUY / SELL / HOLD)
    """

    def __init__(
        self,
        model_type: str = "rf",
        forward_days: int = 5,
        threshold: float = 0.01,
        model_dir: str = "models",
    ) -> None:
        self.model_type   = model_type.lower()
        self.forward_days = forward_days
        self.threshold    = threshold
        self.model_dir    = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_builder = FeatureBuilder()
        self._model  = None
        self._trained = False

    def _build_target(self, close: pd.Series) -> pd.Series:
        fut_ret = close.shift(-self.forward_days) / close - 1
        target  = pd.Series(0, index=close.index)
        target[fut_ret >  self.threshold] =  1
        target[fut_ret < -self.threshold] = -1
        return target

    def _init_model(self):
        if self.model_type == "gb":
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )
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
        # RandomForest (기본) — class_weight="balanced" 로 BUY/SELL/HOLD 불균형 보정
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
        )

    def train(self, df: pd.DataFrame) -> MLResult:
        feat_df = self.feature_builder.transform(df)
        # MA60 포함 → 최소 ~61봉 + forward_days + 여유
        min_rows = self.forward_days + 70
        if len(feat_df) <= min_rows:
            raise ValueError(
                f"학습 데이터 부족. 최소 {min_rows + self.forward_days}봉 이상 필요합니다."
            )

        target    = self._build_target(feat_df["Close"])
        valid_idx = feat_df.index[:-self.forward_days]
        X = feat_df.loc[valid_idx, self.feature_builder.feature_columns]
        y = target.loc[valid_idx].astype(int)

        # ── 시계열 교차검증 5-fold Walk-Forward ──────────────────────
        # gap=forward_days: 훈련 끝과 검증 시작 사이에 미래 정보 누수 차단
        tscv = TimeSeriesSplit(n_splits=5, gap=self.forward_days)
        cv_scores: list[float] = []
        for tr_idx, te_idx in tscv.split(X):
            if len(te_idx) < 5:
                continue
            clf_cv = self._init_model()
            clf_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred_cv = clf_cv.predict(X.iloc[te_idx])
            cv_scores.append(float(accuracy_score(y.iloc[te_idx], pred_cv)))

        # ── 최종 모델: 80/20 시계열 분할 ─────────────────────────────
        split   = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self._model = self._init_model()
        self._model.fit(X_train, y_train)
        self._trained = True

        pred = self._model.predict(X_test)
        acc  = float(accuracy_score(y_test, pred))
        report = classification_report(
            y_test, pred,
            labels=[-1, 0, 1],
            target_names=["하락(-1)", "보합(0)", "상승(1)"],
            zero_division=0,
        )

        imp: dict[str, float] = {}
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

        cv_arr = np.array(cv_scores) if cv_scores else np.array([acc])
        return MLResult(
            model_type=self.model_type,
            accuracy=acc,
            report=report,
            feature_importance=imp,
            cv_scores=[round(s, 4) for s in cv_scores],
            cv_mean=round(float(cv_arr.mean()), 4),
            cv_std=round(float(cv_arr.std()), 4),
        )

    def _latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = self.feature_builder.transform(df)
        if feat_df.empty:
            raise ValueError("예측에 필요한 데이터가 부족합니다.")
        return feat_df.iloc[[-1]][self.feature_builder.feature_columns]

    def predict(self, df: pd.DataFrame) -> str:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습. train()을 먼저 호출하세요.")
        lbl = int(self._model.predict(self._latest_features(df))[0])
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(lbl, "HOLD")

    def predict_proba(self, df: pd.DataFrame) -> dict:
        if not self._trained or self._model is None:
            raise RuntimeError("모델 미학습.")
        x = self._latest_features(df)
        if not hasattr(self._model, "predict_proba"):
            sig = self.predict(df)
            return {"SELL": 1.0 if sig == "SELL" else 0.0,
                    "HOLD": 1.0 if sig == "HOLD" else 0.0,
                    "BUY":  1.0 if sig == "BUY"  else 0.0}
        probs   = self._model.predict_proba(x)[0]
        classes = list(getattr(self._model, "classes_", [-1, 0, 1]))
        out     = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        for cls, p in zip(classes, probs):
            out[mapping.get(int(cls), "HOLD")] = round(float(p), 4)
        return out

    def save(self, filename: str = "model.pkl") -> Path:
        if not self._trained or self._model is None:
            raise RuntimeError("저장할 모델이 없습니다.")
        path = self.model_dir / filename
        joblib.dump(
            {"model": self._model, "model_type": self.model_type,
             "forward_days": self.forward_days, "threshold": self.threshold},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "MLStrategy":
        data = joblib.load(path)
        obj  = cls(
            model_type=data.get("model_type", "rf"),
            forward_days=int(data.get("forward_days", 5)),
            threshold=float(data.get("threshold", 0.01)),
        )
        obj._model   = data["model"]
        obj._trained = True
        return obj
