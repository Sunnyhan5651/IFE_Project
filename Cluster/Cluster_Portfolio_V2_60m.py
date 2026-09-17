"""
IFE Term Project — Cluster-driven Hierarchical Portfolio (V2, COV_LOOKBACK=60m)
================================================================================
참고 논문:
  [K24] Khelifa, Allier & Cucuringu (ICAIF '24)
        "Cluster-driven Hierarchical Representation of Large Asset Universes
         for Optimal Portfolio Construction"
        → 핵심 프레임워크: Signed Spectral Clustering, Gaussian β 가중치,
          EWA 공분산, K차원 GMVP, R회 반복 평균

  [C23] Cartea, Cucuringu & Jin (ICAIF '23)
        "Correlation Matrix Clustering for Statistical Arbitrage Portfolios"
        → 클러스터링 전 CAPM 베타로 시장 수익률 제거 (잔차 수익률 사용)
          → 순수 섹터/산업 공동 움직임 포착 정확도 향상
        → eigenvalue 누적 분산 비율로 K를 매 시점 동적 결정

  [T08] Tola, Lillo, Gallegati & Mantegna (J. Econ. Dyn. Control, 2008)
        "Cluster Analysis for Portfolio Optimization"
        → 예측 위험(σ_pred) vs 실현 위험(σ_real) 비율 추적
          → 클러스터링이 공분산 추정의 안정성을 개선함을 정량적으로 검증

전략 (3개):
  ① Cluster GMVP v2  : 잔차 수익률 + 동적 K  (세 논문 통합, COV_LOOKBACK=60개월)
  ② EW Market        : 유니버스 동일가중
  ③ S&P500           : yfinance 실제 월간 수익률 (^GSPC)

출력:
  cluster_metrics_60m.csv, cluster_monthly_returns_60m.csv,
  cluster_results_60m.png, cluster_risk_ratio_60m.csv,
  cluster_k_history_60m.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_DIR, "IFE_term_stock_data.csv")

# ──────────────────────────────────────────────────────────────────
# 0. 하이퍼파라미터
# ──────────────────────────────────────────────────────────────────
MIN_PRICE       = 5.0
VOL_PCTL_CUTOFF = 0.20
MIN_HISTORY     = 36        # 유니버스 진입 최소 관측 기간 (36개월)
COV_LOOKBACK    = 60        # 공분산 추정 look-back 창 (개월, 부족분은 0으로 채움)

# 클러스터 수 설정 ────────────────────────────────────────────────
K_FIXED         = 10        # v1 전략 고정 K  (GICS 10섹터)
K_MIN, K_MAX    = 5, 20     # v2 동적 K 범위
PCA_THRESHOLD   = 0.50      # 누적 분산 비율 임계값 [C23] Section 3

# 기타 ────────────────────────────────────────────────────────────
R_REPEAT        = 5         # 반복 횟수 [K24] Section 4.1-iii
EWA_BETA        = 0.94      # EWA 감쇠율 (반감기 ≈ 12개월) [K24] Sec 4.6
SIGMA_SCALE     = 0.5       # Gaussian σ² 스케일 [K24] Section 4.2
MAX_WEIGHT      = 0.05      # 종목별 비중 상한
TRANS_COST      = 0.001     # 편도 거래비용 10bps (회전율 1단위당)

# 변동성 오버레이 (Moreira & Muir 2017) ──────────────────────────────
VOL_TARGET      = 0.10      # 목표 연 변동성 10%
VOL_WINDOW      = 12        # 실현 변동성 추정 윈도우 (개월)
VOL_SCALE_MAX   = 2.0       # 레버리지 상한 (2배)

# ──────────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 클리닝
# ──────────────────────────────────────────────────────────────────
print("[1] 데이터 로드 중...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["PERMNO", "date"])

bad_permno = df.loc[df["ADJ_PRC"] > 1e6, "PERMNO"].unique()
clean = df[~df["PERMNO"].isin(bad_permno) & df["EXCHCD"].isin([1, 2, 3])].copy()
print(f"    정제 후 행 수: {len(clean):,}  (원본 {len(df):,})")
clean["ret"] = clean.groupby("PERMNO")["ADJ_PRC"].pct_change()

# ──────────────────────────────────────────────────────────────────
# 2. Wide 매트릭스
# ──────────────────────────────────────────────────────────────────
print("[2] Wide 매트릭스 구성 중...")
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
# ──────────────────────────────────────────────────────────────────

# ── [C23] CAPM 잔차 수익률 ─────────────────────────────────────────
def capm_residuals(X, mkt_ret):
    """
    시장 공통 인자를 제거한 잔차 수익률 계산 [C23].
      β_i = cov(r_i, r_mkt) / var(r_mkt)
      ε_i = r_i − β_i · r_mkt

    클러스터링에 잔차를 사용하면 시장 노출(β) 차이가 아닌
    섹터/산업 단위 공동 움직임(idiosyncratic co-movement)으로
    종목을 그룹화할 수 있어 분산 효과가 높아진다.

    X       : (T, n) 수익률 행렬
    mkt_ret : (T,)  시장 수익률 (EW 프록시)
    반환    : (T, n) 잔차 수익률
    """
    mkt_dm  = mkt_ret - mkt_ret.mean()
    X_dm    = X - X.mean(axis=0, keepdims=True)
    mkt_var = (mkt_dm ** 2).mean() + 1e-12
    beta    = (X_dm * mkt_dm[:, None]).mean(axis=0) / mkt_var   # (n,)
    return X - beta[None, :] * mkt_ret[:, None]                 # (T, n)


# ── [C23] eigenvalue 기반 동적 K 결정 ─────────────────────────────
def optimal_k_pca(corr_mat, pct=PCA_THRESHOLD,
                  k_min=K_MIN, k_max=K_MAX):
    """
    상관행렬 고유값의 누적 비율이 pct 이상인 최소 K 반환 [C23].
    K 경계: [k_min, k_max] 로 안전 범위 제한.

    이론적 근거: 상위 K개 고유값이 전체 분산의 pct를 설명할 때
    K차원 부분공간이 데이터를 충분히 표현하므로 K개 클러스터가 적절.
    """
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


# ── [K24] Signed Spectral Clustering ──────────────────────────────
def signed_spectral_clustering(corr_mat, K, n_init=3):
    """
    상관행렬을 부호 그래프 인접행렬로 해석 [K24] Section 3.1.
    정규화 Laplacian L_N = I − D^{-½} Γ D^{-½} 의
    K개 최소 고유벡터로 K-means++ 클러스터링 수행.
    """
    np.fill_diagonal(corr_mat, 0.0)
    d          = np.abs(corr_mat).sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    L_N        = np.eye(len(d)) - (d_inv_sqrt[:, None] * corr_mat * d_inv_sqrt[None, :])

    _, U = np.linalg.eigh(L_N)
    U    = U[:, :K]
    nrm  = np.linalg.norm(U, axis=1, keepdims=True)
    T_mat = U / np.maximum(nrm, 1e-10)

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
    """
    β_{i,k} ∝ exp(−‖r^(i) − μ_k‖² / 2σ²)  [K24] Section 4.2.
    σ²는 데이터 적응형: 클러스터 내 평균 제곱거리의 SIGMA_SCALE배.
    """
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
    pi_mat = np.zeros((N, N))
    for t in range(T):
        xt = Xc[t:t+1, :]
        pi_mat += (xt.T @ xt - S) ** 2
    pi_mat /= T
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
    """
    포트폴리오 예측 월간 표준편차 [T08].
    σ_pred = √(w^T Σ w)
    Tola et al.은 이 예측치와 실현 변동성의 비율이
    클러스터링으로 개선됨을 부트스트랩으로 보임.
    """
    var = float(w @ Sigma @ w)
    return np.sqrt(max(var, 0.0))


# ──────────────────────────────────────────────────────────────────
# 4. 클러스터 포트폴리오 생성 공통 함수
# ──────────────────────────────────────────────────────────────────
def build_cluster_portfolio(X, K, use_residuals, mkt_ret):
    """
    단일 시점의 클러스터 포트폴리오 비중 계산.
    R회 반복 후 평균 반환 (Robustness, [K24] Section 4.1-iii).

    X            : (T, n) 원시 수익률 행렬
    K            : 클러스터 수
    use_residuals: True → CAPM 잔차 사용 [C23], False → 원시 수익률
    mkt_ret      : (T,) 시장 수익률
    반환         : (n,) 포트폴리오 비중, float 예측 분산
    """
    # 클러스터링용 행렬 선택
    X_clust = capm_residuals(X, mkt_ret) if use_residuals else X
    corr_mat = np.corrcoef(X_clust.T)   # n×n

    port_list  = []
    pred_vars  = []

    for _ in range(R_REPEAT):
        try:
            labels = signed_spectral_clustering(corr_mat.copy(), K, n_init=2)
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

            # EWA 공분산 Σ* (K×K)
            Sigma_star = ewa_covariance(X_syn)

            # K차원 GMVP → 클러스터 비중 α
            alpha = solve_gmvp_cluster(Sigma_star)

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
# 5. 백테스트 메인 루프
# ──────────────────────────────────────────────────────────────────
print("[3] 백테스트 루프 시작...")

# 결과 저장용
v2_rets, ew_rets = [], []
result_dates = []

# [T08] Risk Ratio 추적용
v2_pred_vols = []
k_history = []   # 동적 K 기록 [C23]

# 거래비용 계산용: 직전 월 비중 추적
prev_weights = {}   # {PERMNO: weight}

_SPIN   = ["|", "/", "-", "\\"]
_total  = len(dates) - 1
_t0     = time.time()

for i in range(_total):
    t  = dates[i]
    t1 = dates[i + 1]

    elapsed = time.time() - _t0
    pct     = (i + 1) / _total * 100
    spin    = _SPIN[i % len(_SPIN)]
    sys.stdout.write(f"\r  {spin}  {t.strftime('%Y-%m')}  [{pct:5.1f}%]  경과={elapsed:5.0f}s   ")
    sys.stdout.flush()

    # ── 유니버스 필터 ──────────────────────────────────────────────
    hist_ok  = obs_count.loc[t] >= MIN_HISTORY
    price_ok = price_wide.loc[t] >= MIN_PRICE
    vol_t    = vol_wide.loc[t]
    vol_ok   = vol_t >= vol_t.quantile(VOL_PCTL_CUTOFF)

    universe = (hist_ok & price_ok & vol_ok)
    universe = universe[universe].index
    if len(universe) < K_MAX * 3:
        continue

    # ── 롤링 수익률 행렬 ──────────────────────────────────────────
    X_df  = ret_wins.loc[:t, universe].tail(COV_LOOKBACK)
    # MIN_HISTORY 이상 관측치 있는 종목만 허용, 부족분은 0으로 채움
    valid = X_df.columns[X_df.notna().sum(axis=0) >= MIN_HISTORY]
    if len(valid) < K_MAX * 3:
        continue

    X        = X_df[valid].fillna(0.0).values   # (T, n), NaN → 0
    next_ret = ret_wide.loc[t1]

    # 시장 수익률 프록시 (EW 평균) [C23]
    mkt_ret = X.mean(axis=1)               # (T,)

    # ── [C23] 동적 K 결정 ──────────────────────────────────────────
    X_resid  = capm_residuals(X, mkt_ret)
    corr_res = np.corrcoef(X_resid.T)
    K_dyn    = optimal_k_pca(corr_res)
    k_history.append((t, K_dyn))

    # =================================================================
    # ① Cluster GMVP v2: CAPM 잔차 + 동적 K  [C23 + K24]
    # =================================================================
    w_v2, pv_v2 = build_cluster_portfolio(X, K_dyn, True, mkt_ret)
    if w_v2 is not None:
        r_v2 = float(next_ret.reindex(valid).fillna(0.0).values @ w_v2)
        v2_pred_vols.append(pv_v2)
    else:
        r_v2 = 0.0
        v2_pred_vols.append(np.nan)
        w_v2 = np.zeros(len(valid))

    # 거래비용 차감 (회전율 기반) ────────────────────────────────────
    # turnover = Σ|w_new - w_prev|  (유니버스 변동 포함)
    curr_w = dict(zip(valid.tolist(), w_v2.tolist()))
    all_tk = set(prev_weights.keys()) | set(curr_w.keys())
    turnover = sum(abs(curr_w.get(tk, 0.0) - prev_weights.get(tk, 0.0))
                   for tk in all_tk)
    r_v2 -= turnover * TRANS_COST
    prev_weights = curr_w

    # 변동성 오버레이 (Moreira & Muir 2017) ──────────────────────────
    # 직전 VOL_WINDOW개월 실현 변동성으로 스케일 계산 (lookahead 없음)
    if len(v2_rets) >= VOL_WINDOW:
        realized_vol = np.std(v2_rets[-VOL_WINDOW:], ddof=1) * np.sqrt(12)
        scale_v2 = np.clip(VOL_TARGET / (realized_vol + 1e-8), 0.0, VOL_SCALE_MAX)
    else:
        scale_v2 = 1.0
    r_v2 = r_v2 * scale_v2

    # =================================================================
    # ④ EW Market
    # =================================================================
    r_ew = float(next_ret.reindex(universe).fillna(0.0).mean())

    v2_rets.append(r_v2)
    ew_rets.append(r_ew)
    result_dates.append(t1)

    done    = len(result_dates)
    elapsed = time.time() - _t0
    eta     = elapsed / done * (_total - done) if done > 0 else 0
    sys.stdout.write(
        f"\r  {spin}  {t.strftime('%Y-%m')}  [{pct:5.1f}%]  "
        f"유니버스={len(valid)}  K={K_dyn}  "
        f"경과={elapsed:5.0f}s  남은={eta:5.0f}s   "
    )
    sys.stdout.flush()

print(f"\n\n    총 {len(result_dates)}개월 완료  "
      f"({result_dates[0].strftime('%Y-%m')} ~ {result_dates[-1].strftime('%Y-%m')})")

# ──────────────────────────────────────────────────────────────────
# 6. 결과 DataFrame + S&P500
# ──────────────────────────────────────────────────────────────────
dt_idx = pd.DatetimeIndex(result_dates, name="date")
bt = pd.DataFrame({
    "Cluster v2 + VolOverlay": v2_rets,
    "EW Market":               ew_rets,
}, index=dt_idx)

print("[S&P500 + T-bill] yfinance로 데이터 다운로드 중...")
sp500_raw = yf.download("^GSPC", start="2000-01-01", end="2021-01-01",
                        interval="1mo", auto_adjust=True, progress=False)
sp500_ret = sp500_raw["Close"].squeeze().pct_change().dropna()
sp500_ret.index = sp500_ret.index.to_period("M").to_timestamp("M")
bt["S&P500"] = sp500_ret.reindex(bt.index).values
bt = bt.dropna(subset=["S&P500"])

# 무위험수익률: ^IRX (13주 T-bill 연율 %) → 월간 수익률로 변환
tbill_raw = yf.download("^IRX", start="2000-01-01", end="2021-01-01",
                        interval="1mo", auto_adjust=True, progress=False)
tbill_ann = tbill_raw["Close"].squeeze().dropna() / 100.0
tbill_ann.index = tbill_ann.index.to_period("M").to_timestamp("M")
rf_monthly = ((1 + tbill_ann) ** (1/12) - 1).reindex(bt.index).fillna(method="ffill").fillna(0.0)

# ──────────────────────────────────────────────────────────────────
# 7. 성과지표 5종
# ──────────────────────────────────────────────────────────────────
def compute_metrics(r, rf=None):
    cum    = (1 + r).cumprod()
    n_yrs  = len(r) / 12.0
    ann_r  = cum.iloc[-1]**(1/n_yrs) - 1
    ann_s  = r.std(ddof=1) * np.sqrt(12)
    # 실제 무위험수익률 사용 (없으면 0)
    if rf is not None:
        ann_rf = (1 + rf.reindex(r.index).fillna(0.0)).prod() ** (1/n_yrs) - 1
    else:
        ann_rf = 0.0
    sharpe = (ann_r - ann_rf) / ann_s if ann_s > 1e-8 else np.nan
    mdd    = (cum / cum.cummax() - 1).min()
    return {
        "Cumulative Return": round(cum.iloc[-1] - 1, 4),
        "Annualized Return": round(ann_r, 4),
        "Ann. Risk-Free":    round(ann_rf, 4),
        "Annualized Std":    round(ann_s, 4),
        "Sharpe Ratio":      round(sharpe, 4),
        "Max Drawdown":      round(mdd, 4),
    }

metrics = pd.DataFrame({col: compute_metrics(bt[col], rf_monthly) for col in bt.columns}).T
print("\n=== 성과지표 요약 ===")
print(metrics.to_string())
metrics.to_csv(os.path.join(_DIR, "cluster_metrics_60m.csv"), encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 8. [T08] 예측/실현 위험 비율 (Risk Ratio) 계산 및 저장
#
#    Tola et al. (2008): 클러스터링이 σ_pred / σ_real 을
#    1에 가깝게 만들어 공분산 추정 편향을 줄임을 bootstrap으로 입증.
#    여기서는 rolling 방식으로 동일한 지표를 추적.
#
#    - σ_pred_t = 해당 월 추정 공분산으로 계산한 예측 월간 표준편차
#    - σ_real   = 실제 실현 월간 수익률의 표준편차 (전체 기간)
#    - Risk Ratio = mean(σ_pred) / σ_real
#      → 1에 가까울수록 공분산 추정이 정확함
# ──────────────────────────────────────────────────────────────────
def risk_ratio(pred_vols_sq, realized_rets):
    """
    예측 위험 대비 실현 위험 비율 계산.
    pred_vols_sq : list of float (예측 월간 분산)
    realized_rets: Series (실현 월간 수익률)
    반환         : dict
    """
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

rr_v2 = risk_ratio(v2_pred_vols, bt["Cluster v2 + VolOverlay"])

rr_df = pd.DataFrame({"Cluster v2 (잔차+동적K)": rr_v2}).T

print("\n=== [T08] 예측/실현 위험 비율 (1에 가까울수록 공분산 추정 정확) ===")
print(rr_df.to_string())
rr_df.to_csv(os.path.join(_DIR, "cluster_risk_ratio_60m.csv"), encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 9. 동적 K 히스토리 저장 [C23]
# ──────────────────────────────────────────────────────────────────
k_df = pd.DataFrame(k_history, columns=["date", "K_dynamic"])
k_df.to_csv(os.path.join(_DIR, "cluster_k_history_60m.csv"),
            index=False, encoding="utf-8-sig")

# ──────────────────────────────────────────────────────────────────
# 10. 시각화 (4-panel)
# ──────────────────────────────────────────────────────────────────
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#808080", "#f58231"]

fig, axes = plt.subplots(4, 1, figsize=(14, 20))
fig.suptitle(
    "Cluster-driven Hierarchical Portfolio  —  세 논문 통합 구현  (COV_LOOKBACK=60m)\n"
    "[K24] Khelifa et al. '24  |  [C23] Cartea et al. '23  |  [T08] Tola et al. '08",
    fontsize=12, y=0.99,
)

# Panel 1: 누적 포트폴리오 가치 (log scale)
ax = axes[0]
for col, c in zip(bt.columns, COLORS):
    cum = (1 + bt[col]).cumprod()
    ax.plot(cum.index, cum.values, label=col, lw=1.5, color=c)
ax.set_yscale("log")
ax.set_title("① Portfolio Value of $1  (log scale)")
ax.set_ylabel("Value ($)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel 2: Drawdown
ax = axes[1]
for col, c in zip(bt.columns, COLORS):
    cum = (1 + bt[col]).cumprod()
    dd  = (cum / cum.cummax() - 1) * 100
    ax.plot(dd.index, dd.values, label=col, lw=1.2, color=c)
ax.set_title("② Drawdown (%)")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel 3: [C23] 동적 K 시계열
ax = axes[2]
k_series = pd.Series(
    [k for _, k in k_history],
    index=pd.DatetimeIndex([t for t, _ in k_history]),
)
ax.fill_between(k_series.index, K_MIN, k_series.values,
                alpha=0.4, color="#4363d8", label="Dynamic K (v2)")
ax.set_ylim(0, K_MAX + 2)
ax.set_title(f"③ [C23] 동적 K 선택 (Eigenvalue 누적 분산 ≥ {int(PCA_THRESHOLD*100)}%)")
ax.set_ylabel("K (클러스터 수)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: [T08] 예측/실현 위험 비율
ax = axes[3]
rr_val = rr_v2["Risk Ratio (pred/real)"]
bar = ax.bar(["Cluster v2\n(잔차+동적K)"], [rr_val], color="#e6194b", alpha=0.8, width=0.3)
ax.axhline(1.0, color="black", lw=1.5, ls="--", label="이상적 비율 = 1")
ax.text(bar[0].get_x() + bar[0].get_width()/2, rr_val + 0.01,
        f"{rr_val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("④ [T08] 예측/실현 위험 비율  (1에 가까울수록 공분산 추정 정확)")
ax.set_ylabel("σ_pred / σ_real")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, rr_val * 1.5)

plt.tight_layout()
plt.savefig(os.path.join(_DIR, "cluster_results_60m.png"), dpi=140)
plt.close()

bt.to_csv(os.path.join(_DIR, "cluster_monthly_returns_60m.csv"), encoding="utf-8-sig")

print("\n완료:")
print(f"  cluster_metrics_60m.csv          — 성과지표 (Cluster v2 / EW / S&P500)")
print(f"  cluster_risk_ratio_60m.csv       — [T08] 예측/실현 위험 비율")
print(f"  cluster_k_history_60m.csv        — [C23] 월별 동적 K 기록")
print(f"  cluster_monthly_returns_60m.csv  — 월별 수익률")
print(f"  cluster_results_60m.png          — 4-panel 시각화")
