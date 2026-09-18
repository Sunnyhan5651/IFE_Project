"""
IFE Term Project — Cluster-driven Hierarchical Portfolio
                   + Volatility-Managed Layer (Moreira & Muir, 2017)
======================================================================
IFE_Cluster.py 를 기반으로 한 개선판.

[그대로인 것]  ─ 원본과 결과가 동일
  - 데이터 정제, 유니버스 필터, 모든 하이퍼파라미터
  - 전략 ① Cluster v2, ② Cluster v1, ③ Naive GMVP-LW, ④ EW Market 의
    월별 비중과 수익률 (난수 순서까지 동일하게 유지)

[바뀐 것]
  1. 속도 (약 4배)
     - build_cluster_portfolio: R회 반복마다 새로 하던 고유값분해와
       Ledoit-Wolf 계산을 반복 밖에서 1회만 수행 (둘 다 난수를 쓰지 않는
       결정적 계산이라 결과 동일)
     - ledoit_wolf: T회 파이썬 루프를 동일한 값을 주는 행렬곱으로 대체
  2. 중간 저장 (체크포인트)
     - 매달 결과와 난수 상태를 저장. 중간에 꺼져도 다시 실행하면 이어서
       진행하며, 이어 돌려도 결과는 한 번에 돌린 것과 동일
  3. 벤치마크
     - (기존) S&P500 연 수익률을 12등분한 근사치 → 월 변동성이 사라져
       연변동성 4.7%, Sharpe 2.24 같은 왜곡 발생
     - (수정) yfinance ^SP500TR (배당 포함 총수익지수) 실제 월별 수익률
  4. 날짜 매칭
     - CRSP date(마지막 거래일)와 외부 데이터(달력 월말)를 Timestamp로
       맞추면 불일치 → 연-월(Period) 기준으로 매칭
  5. 무위험수익률
     - (기존) RF = 0
     - (수정) 미국 3개월 T-bill (FRED DTB3, 전월 말 수익률). RF_SOURCE="ff1m" 으로
       Ken French RF(1개월 T-bill)로도 바꿀 수 있음
  6. 그래프 한글 폰트 깨짐 수정

[추가한 것] 변동성 관리 레이어 (Moreira & Muir, 2017, JF)
  x_t      = r_t − rf_t                       (기초 포트폴리오의 월 초과수익률)
  RV_{t-1} = Σ_{d∈t-1} (Mkt-RF_d)²            (직전 달 일간 시장 초과수익률 제곱합)
  w_t      = c / RV_{t-1}
  관리 포트폴리오 수익률 = rf_t + w_t · x_t

  - VM(M&M)     : 논문 원안. c 는 전체 기간에서 관리 전·후 변동성이 같도록
                  설정, 레버리지 제한 없음. (c 는 사후적이지만 Sharpe·알파의
                  t값에는 영향 없음)
  - VM(실시간)   : 실제 운용 가능 버전. c 를 매달 과거 자료로만 추정
                  (최소 24개월, 그 전에는 w=1), 레버리지 0 ~ 1.5배 제한
  - TV12m(수정) : 비교용. 기존 V2_60m 방식(전략 자신의 과거 12개월 월간
                  수익률 표준편차로 목표 10% 조정)이되, 스케일 적용 "전"
                  수익률로 변동성을 추정하도록 되먹임 문제를 수정. v2 에만 적용

  RV 는 포트폴리오 자체가 아니라 시장 전체의 변동성이다 (공통 대리변수).
  CRSP 월간 패널에는 일간 수익률이 없기 때문.

출력 (Cluster/ 폴더, 원본 결과를 덮어쓰지 않도록 _mm 접미사):
  cluster_metrics_mm.csv              성과지표 (rf 반영)
  cluster_monthly_returns_mm.csv      전 전략 월별 수익률
  cluster_monthly_returns_base_mm.csv 기초 4전략 (원본과 같은 형식, 대조용)
  cluster_vm_alpha_mm.csv             관리 vs 비관리 알파 회귀
  cluster_vm_risk_mm.csv              MDD·하락/회복 기간·Calmar·Sortino·최악의 달/12개월
  cluster_vm_crisis_mm.csv            위기 구간별 누적수익률
  cluster_vm_weights_mm.csv           월별 레버리지(w_t)
  cluster_risk_ratio_mm.csv           [T08] 예측/실현 위험 비율
  cluster_k_history_mm.csv            [C23] 월별 동적 K
  cluster_results_mm.png              5-panel 그래프

필요 데이터:
  IFE_term_stock_data.csv  이 파일 폴더, 상위, 상상위 폴더 순으로 탐색
  data/market_rv_monthly.csv  없으면 Ken French 사이트에서 자동 다운로드
  data/sp500_monthly.csv      없으면 yfinance 로 자동 다운로드
  data/tbill3m_monthly.csv    없으면 FRED 에서 자동 다운로드

체크포인트(Cluster/_checkpoint_mm.pkl)는 완료 후에도 남겨 캐시로 쓴다.
무위험수익률·변동성 레이어·지표만 바꾸면 루프 없이 몇 초 만에 다시 계산된다.
"""

import os
import sys
import io
import time
import pickle
import zipfile
import hashlib
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_DIR)
DATA_DIR = os.path.join(_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _find_stock_data():
    # 탐색 순서: 이 폴더 → 상위 → 상상위(레포 루트) → 레포 루트의 Cluster/ (팀 레포 원본 위치)
    repo_root = os.path.dirname(_ROOT)
    for d in (_DIR, _ROOT, repo_root, os.path.join(repo_root, "Cluster")):
        p = os.path.join(d, "IFE_term_stock_data.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "IFE_term_stock_data.csv 를 찾을 수 없습니다. 이 스크립트 폴더, 상위 폴더, "
        "레포 루트, 또는 레포 루트의 Cluster/ 폴더에 두세요.")


DATA_PATH = _find_stock_data()

# ──────────────────────────────────────────────────────────────────
# 0. 하이퍼파라미터  (1~3 은 IFE_Cluster.py 와 동일)
# ──────────────────────────────────────────────────────────────────
MIN_PRICE       = 5.0
VOL_PCTL_CUTOFF = 0.20
MIN_HISTORY     = 36
COV_LOOKBACK    = 36        # look-back 창 (개월)

# 클러스터 수 설정 ────────────────────────────────────────────────
K_FIXED         = 10        # v1 전략 고정 K  (GICS 10섹터)
K_MIN, K_MAX    = 5, 20     # v2 동적 K 범위
PCA_THRESHOLD   = 0.70      # 누적 분산 비율 임계값 [C23] Section 3

# 기타 ────────────────────────────────────────────────────────────
R_REPEAT        = 5         # 반복 횟수 [K24] Section 4.1-iii
EWA_BETA        = 0.94      # EWA 감쇠율 (반감기 ≈ 12개월) [K24] Sec 4.6
SIGMA_SCALE     = 0.5       # Gaussian σ² 스케일 [K24] Section 4.2
MAX_WEIGHT      = 0.05      # 종목별 비중 상한

# 변동성 관리 레이어 (신규) ───────────────────────────────────────
VM_MIN_HIST     = 24        # VM(실시간): c 추정 최소 개월 수
VM_CAP          = 1.5       # VM(실시간): 레버리지 상한
TV_TARGET       = 0.10      # TV12m: 목표 연 변동성 (V2_60m 과 동일)
TV_WINDOW       = 12        # TV12m: 추정 창 (개월)
TV_CAP          = 2.0       # TV12m: 레버리지 상한 (V2_60m 과 동일)

# 무위험수익률 ────────────────────────────────────────────────────
#   "tbill3m": 미국 3개월 T-bill (FRED DTB3). 전월 말 수익률을 이번 달에 적용
#              (월초 매수 시점에 이미 알려진 금리 → 미래 정보 없음)
#   "ff1m"   : Ken French RF (1개월 T-bill, 소수 둘째 자리 % 반올림)
RF_SOURCE       = "tbill3m"

# 위기 구간 (S&P500 기준으로 널리 쓰이는 구간, 전략 결과를 보고 정한 것 아님)
CRISIS_WINDOWS = {
    "금융위기 하락 (2007-11~2009-02)":   ("2007-11", "2009-02"),
    "금융위기 반등 (2009-03~2009-12)":   ("2009-03", "2009-12"),
    "유럽 재정위기 (2011-05~2011-09)":   ("2011-05", "2011-09"),
    "중국 쇼크 (2015-08~2016-02)":       ("2015-08", "2016-02"),
    "2018 4분기 급락 (2018-10~2018-12)": ("2018-10", "2018-12"),
    "코로나 폭락 (2020-02~2020-03)":     ("2020-02", "2020-03"),
    "코로나 반등 (2020-04~2020-12)":     ("2020-04", "2020-12"),
}

OUT_SUFFIX      = "_mm"
CHECKPOINT      = os.path.join(_DIR, "_checkpoint_mm.pkl")

# 한글 폰트 (Windows: 맑은 고딕, macOS: AppleGothic) ────────────────
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if _f in _installed:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 클리닝  (원본과 동일)
# ──────────────────────────────────────────────────────────────────
print("[1] 데이터 로드 중...", flush=True)
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["PERMNO", "date"])

bad_permno = df.loc[df["ADJ_PRC"] > 1e6, "PERMNO"].unique()
clean = df[~df["PERMNO"].isin(bad_permno) & df["EXCHCD"].isin([1, 2, 3])].copy()
print(f"    정제 후 행 수: {len(clean):,}  (원본 {len(df):,})", flush=True)
clean["ret"] = clean.groupby("PERMNO")["ADJ_PRC"].pct_change()

# ──────────────────────────────────────────────────────────────────
# 2. Wide 매트릭스  (원본과 동일)
# ──────────────────────────────────────────────────────────────────
print("[2] Wide 매트릭스 구성 중...", flush=True)
ret_wide   = clean.pivot(index="date", columns="PERMNO", values="ret").sort_index()
price_wide = clean.pivot(index="date", columns="PERMNO", values="ADJ_PRC").sort_index()
vol_wide   = clean.pivot(index="date", columns="PERMNO", values="VOL").sort_index()
dates      = ret_wide.index
obs_count  = price_wide.notna().cumsum()

ret_wins = ret_wide.clip(
    lower=ret_wide.quantile(0.01, axis=1),
    upper=ret_wide.quantile(0.99, axis=1),
    axis=0,
)

# ──────────────────────────────────────────────────────────────────
# 3. 핵심 함수 정의
#    capm_residuals ~ predicted_vol 은 원본과 동일.
#    ledoit_wolf, 스펙트럴 클러스터링, build_cluster_portfolio 만 고속화.
# ──────────────────────────────────────────────────────────────────

# ── [C23] CAPM 잔차 수익률 ─────────────────────────────────────────
def capm_residuals(X, mkt_ret):
    """
    시장 공통 인자를 제거한 잔차 수익률 계산 [C23].
      β_i = cov(r_i, r_mkt) / var(r_mkt)
      ε_i = r_i − β_i · r_mkt
    """
    mkt_dm  = mkt_ret - mkt_ret.mean()
    X_dm    = X - X.mean(axis=0, keepdims=True)
    mkt_var = (mkt_dm ** 2).mean() + 1e-12
    beta    = (X_dm * mkt_dm[:, None]).mean(axis=0) / mkt_var   # (n,)
    return X - beta[None, :] * mkt_ret[:, None]                 # (T, n)


# ── [C23] eigenvalue 기반 동적 K 결정 ─────────────────────────────
def optimal_k_pca(corr_mat, pct=PCA_THRESHOLD,
                  k_min=K_MIN, k_max=K_MAX):
    """상관행렬 고유값의 누적 비율이 pct 이상인 최소 K 반환 [C23]."""
    eig     = np.sort(np.linalg.eigvalsh(corr_mat))[::-1]
    cumvar  = np.cumsum(eig) / (eig.sum() + 1e-12)
    k_raw   = int(np.searchsorted(cumvar, pct)) + 1
    return int(np.clip(k_raw, k_min, k_max))


# ── [K24] K-means++ (NumPy 벡터화) ────────────────────────────────
def kmeans_pp(X, K, max_iter=150):
    n, d = X.shape
    first   = np.random.randint(n)
    centers = X[first:first+1].copy()
    for _ in range(K - 1):
        diff    = X[:, None, :] - centers[None, :, :]          # (n, k, d)
        min_d   = (diff**2).sum(axis=2).min(axis=1)            # (n,)
        prob    = min_d / (min_d.sum() + 1e-12)
        centers = np.vstack([centers, X[np.random.choice(n, p=prob)]])

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        diff       = X[:, None, :] - centers[None, :, :]       # (n, K, d)
        new_labels = (diff**2).sum(axis=2).argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(K):
            if (labels == k).any():
                centers[k] = X[labels == k].mean(axis=0)
    return labels


# ── [K24] Signed Spectral Clustering (2단계로 분리) ────────────────
# 원본 signed_spectral_clustering() 은 [고유값분해 → K-means] 를 한 함수에서
# 했고, build_cluster_portfolio 가 이를 R회 호출해 같은 고유값분해를 R번 반복했다.
# 고유값분해는 난수를 쓰지 않으므로 1회만 하고 K-means 만 R회 반복해도
# 난수 소비 순서와 결과가 원본과 동일하다.
def spectral_embedding(corr_mat, K):
    """
    상관행렬을 부호 그래프 인접행렬로 해석 [K24] Section 3.1.
    정규화 Laplacian L_N = I − D^{-½} Γ D^{-½} 의 K개 최소 고유벡터(행 정규화).
    """
    np.fill_diagonal(corr_mat, 0.0)
    d          = np.abs(corr_mat).sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    L_N        = np.eye(len(d)) - (d_inv_sqrt[:, None] * corr_mat * d_inv_sqrt[None, :])

    _, U = np.linalg.eigh(L_N)
    U    = U[:, :K]
    nrm  = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.maximum(nrm, 1e-10)


def cluster_embedding(T_mat, K, n_init=3):
    """임베딩 T_mat 위에서 K-means++ 를 n_init 회 수행해 관성 최소 라벨 반환."""
    best_lbl, best_inertia = None, np.inf
    for _ in range(n_init):
        lbl = kmeans_pp(T_mat, K)
        inertia = sum(
            np.sum((T_mat[lbl == k] - T_mat[lbl == k].mean(axis=0))**2)
            for k in range(K) if (lbl == k).any()
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_lbl     = lbl.copy()
    return best_lbl


# ── [K24] Gaussian 클러스터 내 가중치 β ───────────────────────────
def gaussian_intra_weights(X, labels, K):
    """β_{i,k} ∝ exp(−‖r^(i) − μ_k‖² / 2σ²)  [K24] Section 4.2."""
    n    = X.shape[1]
    beta = np.zeros(n)
    for k in range(K):
        mask      = labels == k
        if not mask.any():
            continue
        cr        = X[:, mask]
        mu_k      = cr.mean(axis=1)
        sq_dist   = np.sum((cr - mu_k[:, None])**2, axis=0)
        sigma_sq  = SIGMA_SCALE * (sq_dist.mean() + 1e-10)
        raw       = np.exp(-sq_dist / (2 * sigma_sq))
        beta[mask] = raw / (raw.sum() + 1e-10)
    return beta


# ── [K24] EWA 공분산 ───────────────────────────────────────────────
def ewa_covariance(X, beta=EWA_BETA):
    """지수가중 공분산 [K24] Section 4.6."""
    T, N = X.shape
    w    = (1 - beta) / (1 - beta**T + 1e-12) * beta**(T - 1 - np.arange(T))
    w   /= w.sum()
    Xc   = X - (X * w[:, None]).sum(axis=0)
    Xw   = Xc * np.sqrt(w)[:, None]
    return Xw.T @ Xw


# ── 투영 함수 ─────────────────────────────────────────────────────
def project_simplex(v):
    n    = len(v)
    u    = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho  = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    return np.maximum(v - (cssv[rho] - 1.0) / (rho + 1), 0.0)


def project_capped_simplex(v, cap, n_iter=100):
    v = np.asarray(v, dtype=float)
    lo, hi = v.min() - 1.0, v.max()
    for _ in range(n_iter):
        tau = (lo + hi) / 2.0
        w   = np.clip(v - tau, 0.0, cap)
        if w.sum() > 1.0:
            lo = tau
        else:
            hi = tau
    w = np.clip(v - (lo + hi) / 2.0, 0.0, cap)
    s = w.sum()
    return np.full(len(v), 1.0/len(v)) if s <= 1e-10 else w / s


# ── [K24] K차원 GMVP (클러스터 레벨) ─────────────────────────────
def solve_gmvp_cluster(Sigma, n_iter=1000):
    K    = Sigma.shape[0]
    w    = np.full(K, 1.0 / K)
    lam  = np.linalg.eigvalsh(Sigma)[-1]
    step = 1.0 / (2.0 * lam + 1e-12)
    for _ in range(n_iter):
        w_new = project_simplex(w - step * 2.0 * (Sigma @ w))
        if np.max(np.abs(w_new - w)) < 1e-12:
            break
        w = w_new
    return w


# ── Ledoit-Wolf + GMVP (비교용) ────────────────────────────────────
def ledoit_wolf(X):
    T, N = X.shape
    Xc   = X - X.mean(axis=0, keepdims=True)
    S    = (Xc.T @ Xc) / T
    mu_s = np.trace(S) / N
    F    = mu_s * np.eye(N)
    # 원본: for t in range(T): pi_mat += (x_t x_t' − S)²  후 /T
    # 전개하면 mean_t(x_ti² x_tj²) − S_ij²  (mean_t x_ti x_tj = S_ij 이므로)
    X2     = Xc ** 2
    pi_mat = (X2.T @ X2) / T - S ** 2
    pi_hat  = pi_mat.sum()
    rho_hat = np.trace(pi_mat)
    gamma   = np.sum((S - F)**2)
    delta   = 0.0 if gamma < 1e-12 else float(np.clip((pi_hat - rho_hat) / gamma / T, 0, 1))
    return delta * F + (1 - delta) * S


def solve_gmvp_large(Sigma, cap=0.10, n_iter=500):
    N    = Sigma.shape[0]
    w    = np.full(N, 1.0 / N)
    lam  = np.linalg.eigvalsh(Sigma)[-1]
    step = 1.0 / (2.0 * lam + 1e-12)
    for _ in range(n_iter):
        w_new = project_capped_simplex(w - step * 2.0 * (Sigma @ w), cap)
        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new
    return w


# ── [T08] 예측 위험 계산 (Risk Ratio용) ───────────────────────────
def predicted_vol(w, Sigma):
    """포트폴리오 예측 월간 표준편차 σ_pred = √(w^T Σ w) [T08]."""
    var = float(w @ Sigma @ w)
    return np.sqrt(max(var, 0.0))


# ──────────────────────────────────────────────────────────────────
# 4. 클러스터 포트폴리오 생성 공통 함수 (고속화)
# ──────────────────────────────────────────────────────────────────
def build_cluster_portfolio(X, K, use_residuals, mkt_ret):
    """
    단일 시점의 클러스터 포트폴리오 비중 계산.
    R회 반복 후 평균 반환 (Robustness, [K24] Section 4.1-iii).
    반환: (n,) 포트폴리오 비중, float 예측 분산

    원본과의 차이: 반복마다 동일했던 고유값분해(spectral_embedding)와
    Ledoit-Wolf(Sigma_raw)를 1회만 계산. 예외 처리 동작도 원본과 같게 유지.
    """
    X_clust  = capm_residuals(X, mkt_ret) if use_residuals else X
    corr_mat = np.corrcoef(X_clust.T)   # n×n

    # 원본에서 고유값분해가 실패하면 R회 모두 K-means 전에 실패 → None 반환
    try:
        T_mat = spectral_embedding(corr_mat.copy(), K)
    except Exception:
        return None, None

    Sigma_raw  = None
    port_list  = []
    pred_vars  = []

    for _ in range(R_REPEAT):
        try:
            labels = cluster_embedding(T_mat, K, n_init=2)
            if len(np.unique(labels)) < K:
                continue

            # Gaussian β 가중치 (원시 수익률 기준, [K24])
            beta = gaussian_intra_weights(X, labels, K)

            # 합성자산 수익률 r*_k = Σ_i β_{i,k} · r_i
            X_syn = np.zeros((X.shape[0], K))
            for k in range(K):
                mask = labels == k
                if mask.any():
                    X_syn[:, k] = (X[:, mask] * beta[mask]).sum(axis=1)

            # EWA 공분산 Σ* (K×K) → K차원 GMVP → 클러스터 비중 α
            alpha = solve_gmvp_cluster(ewa_covariance(X_syn))

            # 최종 종목 비중 = α_k × β_{i,k}
            w = np.zeros(X.shape[1])
            for k in range(K):
                mask = labels == k
                w[mask] = alpha[k] * beta[mask]

            total = w.sum()
            if total < 1e-10:
                continue
            w /= total
            w  = np.clip(w, 0.0, MAX_WEIGHT)
            w /= w.sum()

            # [T08] 예측 분산: 원시 공분산으로 계산 (in-sample 위험)
            if Sigma_raw is None:
                Sigma_raw = ledoit_wolf(X)
            pred_vars.append(predicted_vol(w, Sigma_raw)**2)
            port_list.append(w)

        except Exception:
            continue

    if not port_list:
        return None, None

    mean_w    = np.mean(port_list, axis=0)
    mean_w   /= mean_w.sum()
    mean_pred = float(np.mean(pred_vars))
    return mean_w, mean_pred


# ──────────────────────────────────────────────────────────────────
# 5. 백테스트 메인 루프 (체크포인트 지원)
# ──────────────────────────────────────────────────────────────────
_CONFIG = (MIN_PRICE, VOL_PCTL_CUTOFF, MIN_HISTORY, COV_LOOKBACK, K_FIXED,
           K_MIN, K_MAX, PCA_THRESHOLD, R_REPEAT, EWA_BETA, SIGMA_SCALE,
           MAX_WEIGHT, os.path.getsize(DATA_PATH), "fast-v1")
CONFIG_KEY = hashlib.md5(repr(_CONFIG).encode()).hexdigest()


def save_checkpoint(state):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, CHECKPOINT)     # 저장 도중 꺼져도 기존 파일은 안전


state = {"config": CONFIG_KEY, "i_next": 0, "rows": [], "k_history": [],
         "rng": np.random.get_state()}
saved = {}
if os.path.exists(CHECKPOINT):
    try:
        with open(CHECKPOINT, "rb") as f:
            saved = pickle.load(f)
    except Exception as e:      # pandas/numpy 버전 차이 등으로 못 읽으면 새로 계산
        print(f"[체크포인트] 읽기 실패({type(e).__name__}) → 무시하고 처음부터 실행", flush=True)
        saved = {}
if saved:
    if saved.get("config") == CONFIG_KEY:
        state = saved
        np.random.set_state(state["rng"])
        if state["i_next"] >= len(dates) - 1:
            print("[체크포인트] 백테스트 완료본 재사용 → 루프 생략 (다시 계산하려면 "
                  "_checkpoint_mm.pkl 삭제)", flush=True)
        else:
            print(f"[체크포인트] {state['i_next']}/{len(dates)-1} 부터 이어서 실행", flush=True)
    else:
        print("[체크포인트] 설정이 달라 무시하고 처음부터 실행", flush=True)

print("[3] 백테스트 루프 시작...", flush=True)
_total = len(dates) - 1
_t0    = time.time()
_i0    = state["i_next"]
_n_done = 0

for i in range(_i0, _total):
    t  = dates[i]
    t1 = dates[i + 1]

    # ── 유니버스 필터 ──────────────────────────────────────────────
    hist_ok  = obs_count.loc[t] >= MIN_HISTORY
    price_ok = price_wide.loc[t] >= MIN_PRICE
    vol_t    = vol_wide.loc[t]
    vol_ok   = vol_t >= vol_t.quantile(VOL_PCTL_CUTOFF)

    universe = (hist_ok & price_ok & vol_ok)
    universe = universe[universe].index

    X_df  = ret_wins.loc[:t, universe].tail(COV_LOOKBACK)
    valid = X_df.columns[X_df.notna().all(axis=0)]

    if len(universe) >= K_MAX * 3 and len(valid) >= K_MAX * 3:
        X        = X_df[valid].values           # (T, n)
        next_ret = ret_wide.loc[t1]
        mkt_ret  = X.mean(axis=1)               # 시장 수익률 프록시 (EW) [C23]

        # ── [C23] 동적 K 결정 ──────────────────────────────────────
        X_resid  = capm_residuals(X, mkt_ret)
        corr_res = np.corrcoef(X_resid.T)
        K_dyn    = optimal_k_pca(corr_res)
        state["k_history"].append((t, K_dyn))

        # ① Cluster GMVP v2: CAPM 잔차 + 동적 K  [C23 + K24]
        w_v2, pv_v2 = build_cluster_portfolio(X, K_dyn, True, mkt_ret)
        if w_v2 is not None:
            r_v2 = float(next_ret.reindex(valid).fillna(0.0).values @ w_v2)
        else:
            r_v2, pv_v2 = 0.0, np.nan

        # ② Cluster GMVP v1: 원시 수익률 + 고정 K  [K24 원안]
        w_v1, pv_v1 = build_cluster_portfolio(X, K_FIXED, False, mkt_ret)
        if w_v1 is not None:
            r_v1 = float(next_ret.reindex(valid).fillna(0.0).values @ w_v1)
        else:
            r_v1, pv_v1 = 0.0, np.nan

        # ③ Naive GMVP-LW (클러스터링 없음)
        Sigma_lw = ledoit_wolf(X)
        w_naive  = solve_gmvp_large(Sigma_lw, cap=0.10)
        r_naive  = float(next_ret.reindex(valid).fillna(0.0).values @ w_naive)
        pv_naive = predicted_vol(w_naive, Sigma_lw)**2

        # ④ EW Market
        r_ew = float(next_ret.reindex(universe).fillna(0.0).mean())

        state["rows"].append(dict(date=t1, v2=r_v2, v1=r_v1, naive=r_naive, ew=r_ew,
                                  pv_v2=pv_v2, pv_v1=pv_v1, pv_naive=pv_naive,
                                  K=K_dyn, n=len(valid)))

        _n_done += 1                            # 실제 계산한 달만 집계 (건너뛴 초기 달 제외)
        elapsed = time.time() - _t0
        eta     = elapsed / _n_done * (_total - i - 1)
        print(f"  {t:%Y-%m}  [{(i+1)/_total*100:5.1f}%]  유니버스={len(valid)}  K={K_dyn:2d}  "
              f"경과={elapsed:5.0f}s  남은≈{eta:5.0f}s", flush=True)

    state["i_next"] = i + 1
    state["rng"]    = np.random.get_state()
    save_checkpoint(state)

rows = pd.DataFrame(state["rows"])
k_history = state["k_history"]
print(f"\n    총 {len(rows)}개월 완료  "
      f"({rows['date'].iloc[0]:%Y-%m} ~ {rows['date'].iloc[-1]:%Y-%m})", flush=True)

# ──────────────────────────────────────────────────────────────────
# 6. 기초 전략 수익률 (원본과 동일한 값·열 이름)
# ──────────────────────────────────────────────────────────────────
N_V2, N_V1 = "Cluster v2 (잔차+동적K, [C23+K24])", f"Cluster v1 (원시+K={K_FIXED}, [K24])"
N_NV, N_EW = "Naive GMVP-LW", "EW Market"
SHORT = {N_V2: "Cluster v2", N_V1: "Cluster v1", N_NV: "Naive GMVP-LW", N_EW: "EW Market"}

dt_idx = pd.DatetimeIndex(rows["date"], name="date")
base = pd.DataFrame({N_V2: rows["v2"].values, N_V1: rows["v1"].values,
                     N_NV: rows["naive"].values, N_EW: rows["ew"].values}, index=dt_idx)
base.to_csv(os.path.join(_DIR, f"cluster_monthly_returns_base{OUT_SUFFIX}.csv"),
            encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 7. 외부 데이터: 시장 실현분산·무위험수익률 (Ken French), S&P500 (yfinance)
#    모든 매칭은 연-월(Period) 기준
# ──────────────────────────────────────────────────────────────────
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")


def _download_ff_daily():
    """Ken French 일간 팩터 (Mkt-RF, RF, 소수 단위). pandas_datareader → 직접 다운로드 순."""
    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                            start="2002-01-01", end="2021-12-31")[0]
        ff.index = pd.to_datetime(ff.index)
    except Exception as e:
        print(f"    pandas_datareader 실패({e}) → 직접 다운로드", flush=True)
        raw = urllib.request.urlopen(FF_URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
        body = []
        for l in lines[start + 1:]:
            if not l.strip() or not l.strip()[0].isdigit():
                break
            body.append(l)
        ff = pd.read_csv(io.StringIO("\n".join([lines[start]] + body)), index_col=0)
        ff.index = pd.to_datetime(ff.index.astype(str).str.strip(), format="%Y%m%d")
        ff = ff.loc["2002-01-01":"2021-12-31"]
    ff = ff[["Mkt-RF", "RF"]] / 100.0
    ff.index.name = "date"
    return ff


def load_market_rv():
    path = os.path.join(DATA_DIR, "market_rv_monthly.csv")
    if not os.path.exists(path):
        print("    market_rv_monthly.csv 없음 → Ken French 일간 팩터 다운로드", flush=True)
        ff = _download_ff_daily()
        ff.to_csv(os.path.join(DATA_DIR, "ff_daily.csv"))
        g = ff.groupby(ff.index.to_period("M"))
        rv = pd.DataFrame({
            "RV2":        g["Mkt-RF"].apply(lambda x: (x ** 2).sum()),
            "RV2_var":    g["Mkt-RF"].var(ddof=1) * g["Mkt-RF"].count(),
            "n_days":     g["Mkt-RF"].count(),
            "mkt_rf_sum": g["Mkt-RF"].sum(),
            "RF":         g["RF"].sum(),
        })
        rv["ann_vol"] = np.sqrt(rv["RV2"] / rv["n_days"] * 252)
        rv.index = rv.index.to_timestamp("M")
        rv.index.name = "date"
        rv.to_csv(path)
    rv = pd.read_csv(path, index_col=0, parse_dates=True)
    rv.index = rv.index.to_period("M")
    return rv


def load_sp500():
    path = os.path.join(DATA_DIR, "sp500_monthly.csv")
    if not os.path.exists(path):
        print("    sp500_monthly.csv 없음 → yfinance 다운로드", flush=True)
        import yfinance as yf
        out = {}
        for col, tk in [("SP500TR", "^SP500TR"), ("GSPC", "^GSPC")]:
            raw = yf.download(tk, start="2002-01-01", end="2021-02-01",
                              interval="1mo", auto_adjust=True, progress=False)
            s = raw["Close"].squeeze().dropna().pct_change().dropna()
            s.index = s.index.to_period("M").to_timestamp("M")
            out[col] = s
        sp = pd.DataFrame(out)
        sp.index.name = "date"
        sp.to_csv(path)
    sp = pd.read_csv(path, index_col=0, parse_dates=True)
    sp.index = sp.index.to_period("M")
    return sp


FRED_DTB3_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
                 "?id=DTB3&cosd=2002-01-01&coed=2021-12-31")


def load_tbill3m():
    """
    미국 3개월 T-bill (FRED DTB3, 연율 %) → 월 무위험수익률.
    rf_t = (1 + y_{t-1})^(1/12) − 1,  y_{t-1} = 전월 마지막 관측 수익률
    반환: 월(Period) 인덱스의 Series
    """
    path = os.path.join(DATA_DIR, "tbill3m_monthly.csv")
    if not os.path.exists(path):
        print("    tbill3m_monthly.csv 없음 → FRED DTB3 다운로드", flush=True)
        raw = urllib.request.urlopen(FRED_DTB3_URL, timeout=60).read().decode()
        d = pd.read_csv(io.StringIO(raw))
        d.columns = ["date", "DTB3"]
        d["date"] = pd.to_datetime(d["date"])
        d["DTB3"] = pd.to_numeric(d["DTB3"], errors="coerce")     # 휴일 '.' → NaN
        y = d.dropna().set_index("date")["DTB3"].resample("M").last()
        out = pd.DataFrame({"DTB3_month_end_pct": y})
        out.index.name = "date"
        out.to_csv(path)
    y = pd.read_csv(path, index_col=0, parse_dates=True)["DTB3_month_end_pct"]
    y.index = y.index.to_period("M")
    return (1 + y.shift(1) / 100.0) ** (1 / 12) - 1


print("\n[4] 외부 데이터 로드 (시장 RV, 무위험수익률, S&P500)...", flush=True)
mrv = load_market_rv()
try:
    sp = load_sp500()
except Exception as e:
    print(f"    [경고] S&P500 로드 실패: {e} → 벤치마크 없이 진행", flush=True)
    sp = None

P = base.index.to_period("M")
if RF_SOURCE == "tbill3m":
    rf = pd.Series(load_tbill3m().reindex(P).values, index=base.index)
    RF_LABEL = "미국 3개월 T-bill (FRED DTB3, 전월말)"
elif RF_SOURCE == "ff1m":
    rf = pd.Series(mrv["RF"].reindex(P).values, index=base.index)
    RF_LABEL = "Ken French RF (1개월 T-bill)"
else:
    raise ValueError(f"RF_SOURCE 는 'tbill3m' 또는 'ff1m': {RF_SOURCE}")
print(f"    무위험수익률: {RF_LABEL}, 연 평균 {rf.mean() * 12:.2%}", flush=True)
rv_prev = pd.Series(mrv["RV2"].reindex(P - 1).values, index=base.index)  # 직전 달 RV
if rf.isna().any() or rv_prev.isna().any():
    raise ValueError("RF 또는 RV 에 결측 월이 있습니다. data/market_rv_monthly.csv 를 확인하세요.")

# ──────────────────────────────────────────────────────────────────
# 8. 변동성 관리 레이어
# ──────────────────────────────────────────────────────────────────
def vm_paper(x, rvp):
    """M&M 원안: w_t = c/RV_{t-1}, c 는 전체기간 변동성 일치, 레버리지 무제한."""
    raw = x / rvp
    c = x.std(ddof=1) / raw.std(ddof=1)
    return c / rvp


def vm_realtime(x, rvp, min_hist=VM_MIN_HIST, cap=VM_CAP):
    """실시간: c_t 를 t 이전 자료로만 추정, 레버리지 [0, cap]."""
    xv, rv = x.values, rvp.values
    w = np.ones(len(xv))
    for j in range(min_hist, len(xv)):
        past = xv[:j]
        c_j  = past.std(ddof=1) / (past / rv[:j]).std(ddof=1)
        w[j] = np.clip(c_j / rv[j], 0.0, cap)
    return pd.Series(w, index=x.index)


def target_vol_12m(x, window=TV_WINDOW, target=TV_TARGET, cap=TV_CAP):
    """비교용: 전략 자신의 과거 12개월 (스케일 전) 초과수익률 변동성으로 목표 변동성 조정."""
    xv = x.values
    w = np.ones(len(xv))
    for j in range(window, len(xv)):
        vol = xv[j - window:j].std(ddof=1) * np.sqrt(12)
        w[j] = np.clip(target / (vol + 1e-8), 0.0, cap)
    return pd.Series(w, index=x.index)


bt = pd.DataFrame(index=base.index)
weights = pd.DataFrame(index=base.index)
VM_P, VM_R, TV = "VM(M&M)", f"VM(실시간,≤{VM_CAP}x)", "TV12m(수정)"

for col in base.columns:
    x = base[col] - rf
    bt[col] = base[col]
    for tag, w in [(VM_P, vm_paper(x, rv_prev)), (VM_R, vm_realtime(x, rv_prev))]:
        name = f"{SHORT[col]} + {tag}"
        bt[name] = rf + w * x
        weights[name] = w
    if col == N_V2:
        w = target_vol_12m(x)
        name = f"{SHORT[col]} + {TV}"
        bt[name] = rf + w * x
        weights[name] = w

if sp is not None:
    bt["S&P500 TR (^SP500TR)"] = sp["SP500TR"].reindex(P).values
    bt["S&P500 (^GSPC, 가격)"]  = sp["GSPC"].reindex(P).values
    if bt.isna().any().any():
        print("    [경고] S&P500 결측 월 존재 — 해당 열 지표는 결측 제외 후 계산", flush=True)

# ──────────────────────────────────────────────────────────────────
# 9. 성과지표 (rf 반영)
# ──────────────────────────────────────────────────────────────────
def compute_metrics(r):
    r      = r.dropna()
    rf_r   = rf.reindex(r.index)
    cum    = (1 + r).cumprod()
    n_yrs  = len(r) / 12.0
    ann_r  = cum.iloc[-1]**(1/n_yrs) - 1
    ann_s  = r.std(ddof=1) * np.sqrt(12)
    ann_rf = (1 + rf_r).prod() ** (1/n_yrs) - 1
    return {
        "Cumulative Return": cum.iloc[-1] - 1,
        "Annualized Return": ann_r,
        "Ann. Risk-Free":    ann_rf,
        "Annualized Std":    ann_s,
        "Sharpe Ratio":      (ann_r - ann_rf) / ann_s if ann_s > 1e-8 else np.nan,
        "Sharpe (rf=0, 원본 정의)": ann_r / ann_s if ann_s > 1e-8 else np.nan,
        "Max Drawdown":      (cum / cum.cummax() - 1).min(),
    }


metrics = pd.DataFrame({c: compute_metrics(bt[c]) for c in bt.columns}).T
lev = pd.DataFrame({"Avg Leverage": weights.mean(), "Max Leverage": weights.max(),
                    "Min Leverage": weights.min()})
metrics = metrics.join(lev)
print(f"\n=== 성과지표 요약 (rf = {RF_LABEL}) ===")
print(metrics.round(4).to_string())
metrics.round(4).to_csv(os.path.join(_DIR, f"cluster_metrics{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 10. M&M 알파: 관리 초과수익률을 비관리 초과수익률에 회귀
#     x^σ_t = α + β x_t + ε_t,  α>0 이면 변동성 관리가 가치를 더함
#     표준오차는 White(HC0) 이분산-강건
# ──────────────────────────────────────────────────────────────────
def mm_alpha(y, x):
    Xm  = np.column_stack([np.ones(len(x)), x])
    b   = np.linalg.lstsq(Xm, y, rcond=None)[0]
    e   = y - Xm @ b
    XtX = np.linalg.inv(Xm.T @ Xm)
    V   = XtX @ (Xm.T * e**2) @ Xm @ XtX
    se  = np.sqrt(np.diag(V))
    r2  = 1 - (e**2).sum() / ((y - y.mean())**2).sum()
    return {"Alpha (ann.)": b[0] * 12, "t(Alpha)": b[0] / se[0], "Beta": b[1],
            "R2": r2, "Appraisal Ratio (ann.)": b[0] / e.std(ddof=2) * np.sqrt(12)}


alpha_rows = {}
for col in base.columns:
    x = (base[col] - rf).values
    for tag in (VM_P, VM_R, TV):
        name = f"{SHORT[col]} + {tag}"
        if name in bt:
            alpha_rows[name] = mm_alpha((bt[name] - rf).values, x)
alpha_df = pd.DataFrame(alpha_rows).T
print("\n=== M&M 알파 회귀 (관리 초과수익률 ~ 비관리 초과수익률) ===")
print(alpha_df.round(4).to_string())
alpha_df.round(4).to_csv(os.path.join(_DIR, f"cluster_vm_alpha{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 10-2. 위험 비교: 낙폭·회복 기간, 하방위험, 위기 구간 수익률
# ──────────────────────────────────────────────────────────────────
def risk_profile(r):
    r      = r.dropna()
    rf_r   = rf.reindex(r.index)
    per    = r.index.to_period("M")
    cum    = pd.Series((1 + r).cumprod().values, index=per)
    dd     = cum / cum.cummax() - 1
    trough = dd.idxmin()
    peak   = cum.loc[:trough].idxmax()
    rec    = cum.loc[trough:][cum.loc[trough:] >= cum.loc[peak]]
    rec_at = rec.index[0] if len(rec) else None

    n_yrs  = len(r) / 12.0
    ann_r  = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_rf = (1 + rf_r).prod() ** (1 / n_yrs) - 1
    x      = (r - rf_r).values
    down   = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * np.sqrt(12)
    roll12 = (1 + r).rolling(12).apply(np.prod, raw=True) - 1
    return {
        "Max Drawdown":         dd.min(),
        "MDD 고점":             str(peak),
        "MDD 저점":             str(trough),
        "하락 기간(개월)":       (trough - peak).n,
        "회복 시점":             str(rec_at) if rec_at is not None else "미회복",
        "회복 기간(개월)":       (rec_at - trough).n if rec_at is not None else np.nan,
        "Calmar (연수익/|MDD|)": ann_r / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Sortino":              (ann_r - ann_rf) / down if down > 0 else np.nan,
        "하방변동성 (연)":       down,
        "최악의 달":             r.min(),
        "최악의 12개월":         roll12.min(),
        "플러스 달 비율":        (r > 0).mean(),
    }


risk_df = pd.DataFrame({c: risk_profile(bt[c]) for c in bt.columns}).T

crisis = {}
for c in bt.columns:
    r = pd.Series(bt[c].values, index=P)
    crisis[c] = {k: (1 + r.loc[a:b]).prod() - 1 for k, (a, b) in CRISIS_WINDOWS.items()}
crisis_df = pd.DataFrame(crisis).T

show = ["Max Drawdown", "하락 기간(개월)", "회복 기간(개월)", "Calmar (연수익/|MDD|)",
        "Sortino", "최악의 달", "최악의 12개월"]
print("\n=== 위험 비교 (낙폭·회복·하방위험) ===")
print(risk_df[show].astype(float).round(4).to_string())
print("\n=== 위기 구간 누적수익률 ===")
print(crisis_df.round(4).to_string())
risk_df.to_csv(os.path.join(_DIR, f"cluster_vm_risk{OUT_SUFFIX}.csv"), encoding="utf-8-sig")
crisis_df.round(4).to_csv(os.path.join(_DIR, f"cluster_vm_crisis{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

w_out = weights.copy()
w_out["Market RV2 (t-1)"] = rv_prev.values
w_out["Market ann.vol (t-1)"] = mrv["ann_vol"].reindex(P - 1).values
w_out.to_csv(os.path.join(_DIR, f"cluster_vm_weights{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

bt.to_csv(os.path.join(_DIR, f"cluster_monthly_returns{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 11. [T08] 예측/실현 위험 비율 (기초 전략, 원본과 동일)
# ──────────────────────────────────────────────────────────────────
def risk_ratio(pred_vols_sq, realized_rets):
    pred_arr  = np.array([v for v in pred_vols_sq if not np.isnan(v)])
    mean_pred = float(np.sqrt(pred_arr).mean()) * np.sqrt(12)  # 연율화
    real_std  = float(realized_rets.std(ddof=1)) * np.sqrt(12) # 연율화
    ratio     = mean_pred / (real_std + 1e-10)
    return {
        "Mean Predicted Vol (Ann.)": round(mean_pred, 4),
        "Realized Vol (Ann.)":       round(real_std, 4),
        "Risk Ratio (pred/real)":    round(ratio, 4),
        "Ratio Bias (|1 - ratio|)":  round(abs(1 - ratio), 4),
    }


rr_v2    = risk_ratio(rows["pv_v2"].tolist(),    base[N_V2])
rr_v1    = risk_ratio(rows["pv_v1"].tolist(),    base[N_V1])
rr_naive = risk_ratio(rows["pv_naive"].tolist(), base[N_NV])
rr_df = pd.DataFrame({"Cluster v2 (잔차+동적K)": rr_v2,
                      f"Cluster v1 (원시+K={K_FIXED})": rr_v1,
                      "Naive GMVP-LW": rr_naive}).T
print("\n=== [T08] 예측/실현 위험 비율 ===")
print(rr_df.to_string())
rr_df.to_csv(os.path.join(_DIR, f"cluster_risk_ratio{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

k_df = pd.DataFrame(k_history, columns=["date", "K_dynamic"])
k_df.to_csv(os.path.join(_DIR, f"cluster_k_history{OUT_SUFFIX}.csv"),
            index=False, encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 12. 시각화 (5-panel)
# ──────────────────────────────────────────────────────────────────
plot_cols = [(N_V2, "#e6194b", "-"), (f"Cluster v2 + {VM_R}", "#911eb4", "-"),
             (f"Cluster v2 + {VM_P}", "#f58231", "--"), (N_EW, "#808080", "-")]
if sp is not None:
    plot_cols.append(("S&P500 TR (^SP500TR)", "#4363d8", "-"))

fig, axes = plt.subplots(5, 1, figsize=(14, 25))
fig.suptitle(
    "Cluster-driven Hierarchical Portfolio + Volatility-Managed Layer\n"
    "[K24] Khelifa et al. '24  |  [C23] Cartea et al. '23  |  [T08] Tola et al. '08  |  "
    "Moreira & Muir '17",
    fontsize=12, y=0.995,
)

ax = axes[0]
for col, c, ls in plot_cols:
    cum = (1 + bt[col].dropna()).cumprod()
    ax.plot(cum.index, cum.values, label=col, lw=1.5, color=c, ls=ls)
ax.set_yscale("log")
ax.set_title("① Portfolio Value of $1  (log scale)")
ax.set_ylabel("Value ($)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
for col, c, ls in plot_cols:
    cum = (1 + bt[col].dropna()).cumprod()
    ax.plot(cum.index, (cum / cum.cummax() - 1) * 100, label=col, lw=1.2, color=c, ls=ls)
ax.set_title("② Drawdown (%)")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(weights.index, weights[f"Cluster v2 + {VM_R}"], color="#911eb4", lw=1.4,
        label=f"w_t  {VM_R}")
ax.plot(weights.index, weights[f"Cluster v2 + {VM_P}"], color="#f58231", lw=1.0, ls="--",
        label=f"w_t  {VM_P}")
ax.axhline(1.0, color="black", lw=0.8, ls=":")
ax.set_ylabel("레버리지 w_t")
ax.set_title("③ 변동성 관리 레버리지 (Cluster v2) vs 직전 달 시장 실현변동성")
ax2 = ax.twinx()
ax2.fill_between(weights.index, 0, w_out["Market ann.vol (t-1)"] * 100,
                 color="#4363d8", alpha=0.15, label="시장 연율 변동성 t-1 (%)")
ax2.set_ylabel("시장 변동성 (%)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[3]
k_series = pd.Series([k for _, k in k_history],
                     index=pd.DatetimeIndex([t for t, _ in k_history]))
ax.fill_between(k_series.index, K_MIN, k_series.values,
                alpha=0.4, color="#4363d8", label="Dynamic K (v2)")
ax.axhline(K_FIXED, color="#e6194b", lw=1.5, ls="--", label=f"Fixed K={K_FIXED} (v1)")
ax.set_ylim(0, K_MAX + 2)
ax.set_title(f"④ [C23] 동적 K 선택 (Eigenvalue 누적 분산 ≥ {int(PCA_THRESHOLD*100)}%)")
ax.set_ylabel("K (클러스터 수)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[4]
rr_vals  = [rr_v2["Risk Ratio (pred/real)"], rr_v1["Risk Ratio (pred/real)"],
            rr_naive["Risk Ratio (pred/real)"]]
rr_lbls  = ["Cluster v2\n(잔차+동적K)", f"Cluster v1\n(원시+K={K_FIXED})", "Naive\nGMVP-LW"]
bars = ax.bar(rr_lbls, rr_vals, color=["#e6194b", "#3cb44b", "#4363d8"], alpha=0.8, width=0.5)
ax.axhline(1.0, color="black", lw=1.5, ls="--", label="이상적 비율 = 1")
for bar, val in zip(bars, rr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("⑤ [T08] 예측/실현 위험 비율  (1에 가까울수록 공분산 추정 정확)")
ax.set_ylabel("σ_pred / σ_real")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(rr_vals) * 1.3)

plt.tight_layout()
plt.savefig(os.path.join(_DIR, f"cluster_results{OUT_SUFFIX}.png"), dpi=140)
plt.close()

# ──────────────────────────────────────────────────────────────────
# 13. MDD 개선 검증 (변동성 관리 레이어 = VM(실시간))
#   (1) 대조군: 원본을 VM 과 같은 변동성이 되도록 "고정" 비율로만 축소
#       → 이걸 이겨야 "덜 투자해서"가 아니라 "타이밍"의 효과
#   (2) 한 사건에 덜 좌우되는 지표: 평균 낙폭, CDaR95, 낙폭 10% 이상 기간, 구간별 MDD
#   (3) 12개월 블록 부트스트랩: (원본, VM) 월 수익률 쌍을 같은 블록으로 재표본
#       → 역사가 달리 전개됐어도 VM 의 MDD 가 더 나은가
#       (각 달의 실제 레버리지는 고정한 채 경로만 섞는 검정)
# ──────────────────────────────────────────────────────────────────
BOOT_N, BOOT_BLOCK = 2000, 12
SUB_PERIODS = [("2003~2009", "2003", "2009"), ("2010~2014", "2010", "2014"),
               ("2015~2020", "2015", "2020")]


def dd_series(r):
    cum = (1 + r).cumprod()
    return cum / cum.cummax() - 1


def dd_profile(r):
    dd  = dd_series(r)
    q05 = dd.quantile(0.05)
    out = {"MDD": dd.min(), "평균 낙폭": dd.mean(), "CDaR95": dd[dd <= q05].mean(),
           "낙폭 10% 이상 달 비율": (dd <= -0.10).mean()}
    for lbl, a, b in SUB_PERIODS:
        out[f"MDD {lbl}"] = dd_series(r.loc[a:b]).min()
    return out


def block_boot_mdd(a, b, n_boot=BOOT_N, block=BOOT_BLOCK, seed=0):
    rs = np.random.RandomState(seed)          # 전역 난수와 분리
    n, nb = len(a), int(np.ceil(len(a) / block))
    out = np.empty((n_boot, 2))
    for i in range(n_boot):
        starts = rs.randint(0, n, nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        for j, s in enumerate((a, b)):
            cum = np.cumprod(1 + s[idx])
            out[i, j] = (cum / np.maximum.accumulate(cum) - 1).min()
    return out


BASES = [N_V2, N_V1, N_NV, N_EW]
ctrl, dd_rows, boot = {}, {}, {}
for col in BASES:
    vm = f"{SHORT[col]} + {VM_R}"
    x  = base[col] - rf
    k  = (bt[vm] - rf).std(ddof=1) / x.std(ddof=1)
    ctrl[col] = rf + k * x
    dd_rows[f"{SHORT[col]} (원본)"] = dd_profile(base[col])
    dd_rows[f"{SHORT[col]} (변동성 맞춘 원본, {k:.2f}배 고정)"] = dd_profile(ctrl[col])
    dd_rows[f"{SHORT[col]} + {VM_R}"] = dd_profile(bt[vm])
    bs = block_boot_mdd(base[col].values, bt[vm].values)
    bc = block_boot_mdd(ctrl[col].values, bt[vm].values)
    boot[col] = bs
    dd_rows[f"{SHORT[col]} + {VM_R}"].update({
        "부트스트랩 P(VM MDD가 원본보다 나음)": (bs[:, 1] > bs[:, 0]).mean(),
        "부트스트랩 P(VM MDD가 대조군보다 나음)": (bc[:, 1] > bc[:, 0]).mean(),
        "부트스트랩 MDD 개선폭 중앙값(%p)": np.median(bs[:, 1] - bs[:, 0]) * 100,
        "부트스트랩 MDD 개선폭 5%(%p)": np.percentile(bs[:, 1] - bs[:, 0], 5) * 100,
        "부트스트랩 MDD 개선폭 95%(%p)": np.percentile(bs[:, 1] - bs[:, 0], 95) * 100,
    })

dd_df = pd.DataFrame(dd_rows).T
print("\n=== MDD 개선 검증 (원본 vs 변동성 맞춘 원본 vs VM 실시간) ===")
print(dd_df.round(4).to_string())
dd_df.round(4).to_csv(os.path.join(_DIR, f"cluster_vm_drawdown{OUT_SUFFIX}.csv"), encoding="utf-8-sig")

# ── 그래프 ────────────────────────────────────────────────────────
C_BASE, C_VM, C_CTRL, C_SP = "#d62728", "#7b2cbf", "#8c8c8c", "#1f77b4"
fig = plt.figure(figsize=(14, 13))
gs  = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.28, wspace=0.22, top=0.94)
fig.suptitle("변동성 관리 레이어의 MDD 개선 효과  (Cluster v2, 2003-02 ~ 2020-12)",
             fontsize=14, fontweight="bold", y=0.975)

# (a) 낙폭 곡선
ax = fig.add_subplot(gs[0, :])
for k_, (a, b) in CRISIS_WINDOWS.items():
    if "반등" in k_:
        continue
    ax.axvspan(pd.Period(a).start_time, pd.Period(b).end_time, color="#f2c14e", alpha=0.18, lw=0)
curves = [(base[N_V2], "Cluster v2 원본 (GitHub)", C_BASE, "-", 1.8),
          (ctrl[N_V2], "원본을 같은 변동성으로 고정 축소 (대조군)", C_CTRL, "--", 1.3),
          (bt[f"Cluster v2 + {VM_R}"], "Cluster v2 + 변동성 관리", C_VM, "-", 2.0)]
if sp is not None:
    curves.append((bt["S&P500 TR (^SP500TR)"].dropna(), "S&P500 TR", C_SP, ":", 1.3))
for s, lbl, c, ls, lw in curves:
    d = dd_series(s) * 100
    ax.plot(d.index, d.values, color=c, ls=ls, lw=lw, label=f"{lbl}  (MDD {d.min():.1f}%)")
    if c in (C_BASE, C_VM):
        ax.annotate(f"{d.min():.1f}%", (d.idxmin(), d.min()), xytext=(8, -4),
                    textcoords="offset points", color=c, fontsize=10, fontweight="bold")
ax.set_title("(a) 낙폭 곡선 — 음영은 하락 위기 구간 (금융위기·유럽재정·중국쇼크·2018 4Q·코로나)",
             fontsize=11, loc="left")
ax.set_ylabel("고점 대비 낙폭 (%)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

# (b) 전략별 MDD 막대
ax = fig.add_subplot(gs[1, 0])
xs = np.arange(len(BASES))
wd = 0.26
vals = [[dd_rows[f"{SHORT[c]} (원본)"]["MDD"] * 100 for c in BASES],
        [dd_series(ctrl[c]).min() * 100 for c in BASES],
        [dd_rows[f"{SHORT[c]} + {VM_R}"]["MDD"] * 100 for c in BASES]]
for j, (v, lbl, c) in enumerate(zip(vals, ["원본 (GitHub)", "대조군 (고정 축소)", "+ 변동성 관리"],
                                    [C_BASE, C_CTRL, C_VM])):
    bars = ax.bar(xs + (j - 1) * wd, v, wd, color=c, alpha=0.85, label=lbl)
    for bb, vv in zip(bars, v):
        ax.text(bb.get_x() + bb.get_width() / 2, vv - 1.0, f"{vv:.0f}", ha="center",
                va="top", fontsize=8, color=c)
ax.set_xticks(xs)
ax.set_xticklabels([SHORT[c] for c in BASES], fontsize=9)
ax.set_ylabel("MDD (%)")
ax.set_ylim(min(min(v) for v in vals) * 1.18, 0)
ax.set_title("(b) 네 기초 전략 모두에서 MDD 비교", fontsize=11, loc="left")
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", alpha=0.3)

# (c) 부트스트랩 분포
ax = fig.add_subplot(gs[1, 1])
diff = (boot[N_V2][:, 1] - boot[N_V2][:, 0]) * 100
p_better = (diff > 0).mean()
ax.hist(diff, bins=50, color=C_VM, alpha=0.75)
ax.axvline(0, color="black", lw=1.2)
ax.axvline(np.median(diff), color=C_VM, lw=1.5, ls="--")
ax.text(0.03, 0.95,
        f"VM 쪽 MDD가 더 나은 비율: {p_better:.1%}\n"
        f"개선폭 중앙값: {np.median(diff):+.1f}%p\n"
        f"90% 구간: {np.percentile(diff, 5):+.1f} ~ {np.percentile(diff, 95):+.1f}%p",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
ax.set_xlabel("MDD 개선폭 (%p, +면 변동성 관리가 나음)")
ax.set_ylabel("빈도")
ax.set_title(f"(c) {BOOT_BLOCK}개월 블록 부트스트랩 {BOOT_N:,}회 — Cluster v2", fontsize=11, loc="left")
ax.grid(alpha=0.3)

plt.savefig(os.path.join(_DIR, f"cluster_vm_drawdown{OUT_SUFFIX}.png"), dpi=140, bbox_inches="tight")
plt.close()

# 체크포인트는 지우지 않고 남겨둔다: 완료된 백테스트 결과의 캐시 역할.
# 무위험수익률·변동성 레이어·지표만 바꿔 다시 실행하면 루프를 건너뛰고 몇 초 만에 끝난다.
# 백테스트 파라미터(1~3절)를 바꾸면 설정 해시가 달라져 자동으로 처음부터 다시 계산한다.

print(f"\n완료 (Cluster/ 폴더, 접미사 {OUT_SUFFIX}):")
for f in ["cluster_metrics", "cluster_monthly_returns", "cluster_monthly_returns_base",
          "cluster_vm_alpha", "cluster_vm_risk", "cluster_vm_crisis", "cluster_vm_drawdown",
          "cluster_vm_weights", "cluster_risk_ratio", "cluster_k_history"]:
    print(f"  {f}{OUT_SUFFIX}.csv")
print(f"  cluster_results{OUT_SUFFIX}.png")
print(f"  cluster_vm_drawdown{OUT_SUFFIX}.png")
