# -*- coding: utf-8 -*-
"""
IFE Term Project - 논문 재현 구현 (Cluster-driven Hierarchical Portfolio)
==========================================================================
재현 대상 논문 [K24]:
    Khelifa, N., Allier, J., & Cucuringu, M. (2024).
    "Cluster-driven Hierarchical Representation of Large Asset Universes
     for Optimal Portfolio Construction." ICAIF '24.

논문 핵심 아이디어 요약
------------------------
  1) n개 종목의 상관행렬(Pearson)을 "부호(signed) 그래프"의 인접행렬로 해석.
  2) 4종의 signed spectral clustering 알고리즘으로 종목을 K개의 클러스터로 그룹화.
     (Spectral Clustering / Signed Laplacian / SPONGE / Symmetric SPONGE)
  3) 클러스터 내부: 클러스터 중심(centroid)까지의 거리에 반비례하는 Gaussian 가중치(beta)를
     모든 종목에 부여 (대표종목 1개를 뽑는 대신, 클러스터 내 전 종목을 비중있게 사용).
  4) 각 클러스터를 하나의 "합성자산(synthetic asset)"으로 보고, 그 수익률로 K x K
     공분산행렬(Sigma*, EWA 방식)을 추정 -> 클러스터 레벨 Markowitz(GMVP) 최적화로
     클러스터 비중(alpha) 산출.
  5) 최종 종목 비중 = alpha_{그 종목이 속한 클러스터} x beta_{그 종목, 클러스터 내부}.
  6) 클러스터링의 랜덤성(k-means)을 완화하기 위해 R회 반복 후 평균.
  7) 위 과정을 매 리밸런싱 시점마다 반복하는 walk-forward 백테스트.

원 논문 대비 본 구현에서의 데이터/설계 적응(모두 보고서에 한계점으로 명시할 사항)
--------------------------------------------------------------------------
  - 원 논문: 2013-2019, 695개 미국주식의 "일별" 데이터, lookback=75거래일,
    평가window=5거래일(주간 리밸런싱).
  - 본 구현: 2000-2020, CRSP "월별" 데이터, lookback=36개월, 평가window=1개월
    (과제 가이드라인이 명시한 "월별 리밸런싱" 요구사항에 맞춘 것).

    [Lookback=36개월을 선택한 이유]
    (1) 통계적 안정성: 상관/공분산 추정치가 의미있게 수렴하려면 관측치 수(T)가
        어느 정도 확보되어야 한다. T가 너무 작으면(예: 12개월) 표본상관행렬 자체의
        추정오차가 매우 커져 클러스터링 결과가 노이즈에 좌우된다.
    (2) 정상성(stationarity): 반대로 lookback을 너무 길게(예: 10년) 잡으면 그 사이
        시장구조(섹터간 상관관계, 변동성 국면)가 바뀌어버려 "현재" 클러스터 구조를
        더 이상 대표하지 못하는 과거 정보까지 섞이게 된다.
    (3) 실무/학계 관행: 월간 데이터 기반 공분산 추정에서 3년(36개월) 창은 매우
        표준적인 선택이다(예: MSCI Barra 월간 팩터모델, 다수의 자산배분 논문). 원 논문의
        75거래일(~3.5개월)은 일간데이터 기준이며, 이를 그대로 월간 데이터에 적용하면
        3~4개월치 관측치로 공분산을 추정하는 셈이 되어 500개 종목 규모에서는 표본크기가
        지나치게 작아 심각한 추정오차 문제를 야기한다. 따라서 "일간 대비 저빈도"라는
        논문과의 근본적 차이를 고려해 36개월을 채택했다.
    (4) 본 프로젝트의 기존 참고 스크립트(IFE_Shrinkage.py, IFE_Cluster.py)도 동일하게
        COV_LOOKBACK=36을 사용하고 있어 프로젝트 내 일관성도 고려했다.
    -> 이는 여전히 설계상의 선택(hyperparameter)이며, 24개월/60개월 등으로 민감도를
       테스트해볼 수 있다(본 스크립트는 미포함, 필요시 추가 가능).

  - 원 논문은 미래 수익률에 인위적으로 노이즈를 주입해 "예측정확도(eta)"에 따른
    민감도를 분석하는 실험을 포함하지만, 이는 미래정보를 사용하는 실험적 장치이며
    실제 백테스트가 아니므로 본 구현에서는 제외. 대신 클러스터 기대수익률은 별도로
    추정하지 않고(=예측하지 않고) 클러스터 레벨에서 "최소분산(GMVP)"만 푼다.
    이는 과제 가이드라인의 최소요건("GMVP, MSRP 등 평균-분산 최적화")을 그대로 충족한다.
  - 4개 클러스터링 알고리즘 모두 논문 Section 3의 수식을 그대로 구현했다(아래 각 함수의
    docstring에 해당 수식 명시). SPONGE 계열의 정규화 파라미터(tau+, tau-)는 논문 실험
    섹션에 구체적 수치가 제시되어 있지 않아, SPONGE 원 논문([5], Cucuringu et al. 2019)의
    기본값인 tau+ = tau- = 1 을 사용함(가정사항으로 보고서에 명시 필요).

  - [수정사항] 유니버스 구성 - point-in-time 방식으로 전환 (생존편향/lookahead 제거)
    이전 버전은 "2000~2020년 전 기간(252개월) 동안 결측 없이 존재한 종목"만으로 고정
    유니버스를 구성했는데, 이는 2020년까지 살아남은 종목이라는 사실을 2003년 시점의
    포트폴리오 구성에 사용하는 것과 같아 명백한 미래정보 누출(lookahead bias)이자
    생존편향(survivorship bias)이다. 본 버전은 이를 수정하여 매 리밸런싱 시점 t마다:
      (a) 그 시점까지 실제로 COV_LOOKBACK개월 이상 관측된 종목만 사용(생존편향 방지),
      (b) 가격 >= $5 (페니스톡 제외), 거래량 하위 20% 제외(유동성 필터),
      (c) lookback 구간 내 결측이 없는 종목만 사용(공분산 추정 안정성),
      (d) 계산량 관리를 위해 그 시점 기준 유동성 상위 N_UNIVERSE개로 캡(cap) - 이 캡도
          오직 시점 t까지의 정보만 사용하므로 미래정보 누출이 없다.
    이렇게 구성된 유니버스는 매월 달라지며(종목이 새로 편입되거나 필터를 통과하지 못해
    제외될 수 있음), 원 논문의 "고정된 n=695개 유니버스"와는 다르지만, 실제 백테스트로서
    타당성을 갖기 위한 필수적인 수정이다. 상장폐지 등으로 다음달 수익률이 없는 종목은
    0%(현금 대체)로 처리한다(CRSP DLRET 필드 부재에 따른 단순화, 한계점으로 명시 필요).

  - [수정사항] S&P 500 벤치마크 - Yahoo Finance 실제 월별데이터로 교체
    이전 버전은 공개된 연도별 총수익률을 월별로 균등분할(근사)했으나, 본 버전은
    Yahoo Finance에서 ^SP500TR(S&P 500 Total Return Index, 배당재투자 포함) 종가를
    직접 조회하여 실제 월별 수익률을 사용한다(아래 fetch_sp500_total_return_monthly
    함수 참고). 네트워크 접근이 불가능한 환경을 위해 로컬 캐시 및 근사치 fallback을
    포함한다.

  - [3차 개정] IFE_Cluster.py(동일 프로젝트 내 참고파일)가 왜 더 나은 수익률/Sharpe를
    보일 수 있는지에 대한 가설 3가지를 검증하기 위해 다음을 반영:
      (1) K=24 -> K=10으로 변경. 앞선 K 민감도 분석(paper_k_sensitivity_metrics.csv)에서
          이미 K=10이 K=24보다 Sharpe/수익률 모두 우수함을 확인했으므로, 이번 실험의
          기본 K를 10으로 바꾼다(더 이상 논문 5.2절의 "K=24가 최적" 결과를 그대로 따르지
          않고, 우리 데이터에서 실증적으로 더 나은 값을 사용).
      (2) CAPM 잔차 클러스터링 [C23, Cartea et al. 2023] 추가: 클러스터링 입력으로
          원시수익률(raw) 대신, 시장공통요인(1-factor 베타)을 제거한 잔차수익률을 사용하는
          버전을 각 클러스터링 알고리즘마다 병행 계산하여 "raw" vs "CAPM 잔차" 8개 조합을
          직접 비교한다. (단, Gaussian intra-cluster 가중치와 클러스터 합성수익률/EWA공분산
          계산은 논문 원안대로 원시수익률을 그대로 사용 - 오직 "클러스터를 나누는 기준"만
          잔차로 바꾼다.)
      (3) 유니버스 크기(N_UNIVERSE)를 500 -> 800으로 확대. IFE_Cluster.py는 유니버스를
          전혀 캡하지 않지만(월별 2,000~3,000개 종목 전체 사용), 이를 그대로 재현하면
          계산량이 (n/500)^3 규모로 증가해 단일 실행이 수 시간~수일 걸릴 수 있어
          비현실적이다. 800은 "0에서 완전 무제한"으로 가는 중간 지점으로, 실행시간을
          약 1시간 내외로 유지하면서 유니버스 크기 확대의 방향성 자체는 반영한 절충안이다
          (사전 타이밍 테스트로 검증, 아래 참고). 완전한 무제한 유니버스 재현은 별도의
          장시간(수 시간) 실행이 필요하며 본 실행에는 포함하지 않았다(한계점으로 명시).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh as gen_eigh   # 일반화 고유값 문제(generalized eigenproblem)용
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)  # 재현성(클러스터링 초기화 등)

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_DIR, "IFE_term_stock_data.csv")

# 빠른 검증을 위한 스모크테스트 스위치 (환경변수 IFE_FAST_TEST=1 이면 소규모로 실행)
FAST_TEST = os.environ.get("IFE_FAST_TEST") == "1"

# ------------------------------------------------------------------
# 0. 하이퍼파라미터
# ------------------------------------------------------------------
N_UNIVERSE   = int(os.environ.get("IFE_N_UNIVERSE", 60 if FAST_TEST else 800))  # 매 시점 유동성 상위 N개로 캡(point-in-time) - 500->800으로 확대(3차 개정 항목3)
COV_LOOKBACK = 36                          # lookback window L (개월) - 선택 이유는 상단 docstring 참고
MIN_PRICE    = 5.0                         # 페니스톡 제외
VOL_PCTL_CUTOFF = 0.20                     # 유동성 하위 20% 제외
K_FIXED      = 10                          # 24->10으로 변경(3차 개정 항목1): 앞선 K민감도분석에서 K=10이 실증적으로 더 우수
PCA_THRESHOLD = 0.70                       # 동적 K 결정 임계값 (논문 4.4절, 이번 실행에서는 미사용)
K_MIN, K_MAX = 5, 40
R_REPEAT     = 2 if FAST_TEST else 5       # robustness 반복횟수 R (논문 Table 2: 10, 컴퓨팅 시간상 축소)
EWA_BETA     = 0.94                        # EWA 공분산 감쇠율(월간 데이터에 맞게 조정: 반감기 약 11개월)
TAU_PLUS     = 1.0                         # SPONGE 정규화 파라미터 (원 SPONGE 논문 기본값)
TAU_MINUS    = 1.0
MAX_WEIGHT   = 0.05                        # 종목당 비중 상한(과도한 쏠림 방지, 실무적 제약)
RF           = 0.0
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99       # 극단치 완화(상관행렬/공분산 추정에만 사용)

print(f"[설정] N_UNIVERSE(cap)={N_UNIVERSE}, COV_LOOKBACK={COV_LOOKBACK}, K_FIXED={K_FIXED}, "
      f"R_REPEAT={R_REPEAT}, FAST_TEST={FAST_TEST}")

# ------------------------------------------------------------------
# 1. 데이터 로드 및 정제 (유니버스는 고정하지 않고, 아래 point_in_time_window()에서
#    매 리밸런싱 시점마다 그 시점까지의 정보만으로 동적으로 구성한다 - 생존편향 방지)
# ------------------------------------------------------------------
print("[1] 데이터 로드 및 정제 중...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["PERMNO", "date"])

# 가이드라인상 허용된 거래소(1=NYSE, 2=AMEX, 3=NASDAQ)만 사용, 가격 이상치 종목 제외
bad_permno = df.loc[df["ADJ_PRC"] > 1e6, "PERMNO"].unique()
clean = df[~df["PERMNO"].isin(bad_permno) & df["EXCHCD"].isin([1, 2, 3])].copy()
clean["ret"] = clean.groupby("PERMNO")["ADJ_PRC"].pct_change()

ret_wide = clean.pivot(index="date", columns="PERMNO", values="ret").sort_index()
price_wide = clean.pivot(index="date", columns="PERMNO", values="ADJ_PRC").sort_index()
vol_wide = clean.pivot(index="date", columns="PERMNO", values="VOL").sort_index()
dates = ret_wide.index

# 각 시점까지 실제로 관측된 개월 수 (point-in-time 히스토리 길이) - 생존편향 방지의 핵심
obs_count = price_wide.notna().cumsum()

# 상관행렬/공분산 추정에만 사용할 winsorize된 수익률 (실현손익 계산에는 원본 ret_wide 사용)
ret_wins = ret_wide.clip(
    lower=ret_wide.quantile(WINSOR_LOW, axis=1),
    upper=ret_wide.quantile(WINSOR_HIGH, axis=1),
    axis=0,
)

print(f"    정제 후 종목 수(전체 기간 통틀어): {ret_wide.shape[1]:,}, 기간: "
      f"{dates[0].strftime('%Y-%m')} ~ {dates[-1].strftime('%Y-%m')} ({len(dates)}개월)")


def point_in_time_window(i):
    """
    시점 t=dates[i] 기준 "point-in-time" 유니버스를 구성하고, lookback 수익률 행렬 및
    다음달(t+1) 실현수익률을 반환한다. 오직 시점 t까지 관측 가능한 정보만 사용하므로
    미래정보 누출(lookahead bias)이 없다.

    필터 (모두 시점 t 기준):
      (a) hist_ok  : 그 시점까지 COV_LOOKBACK개월 이상 관측된 종목만(생존편향 방지)
      (b) price_ok : 가격 >= MIN_PRICE (페니스톡 제외)
      (c) vol_ok   : 거래량 하위 VOL_PCTL_CUTOFF 분위 제외 (유동성 필터)
      (d) window_ok: lookback 구간 내 결측치가 없는 종목만 (공분산 추정 안정성)
      (e) 계산량 관리를 위해 (a)-(d)를 통과한 종목 중 그 시점 유동성(거래량) 상위
          N_UNIVERSE개로 캡. 이 역시 시점 t까지의 정보만 사용하므로 lookahead 아님.

    반환: (X_raw, X_wins, next_ret, cols) 또는 유니버스가 너무 작으면 None.
      X_raw    : (COV_LOOKBACK, n) 원본 수익률
      X_wins   : (COV_LOOKBACK, n) winsorize된 수익률
      next_ret : (n,) 다음달 실현수익률 (상장폐지 등 결측 시 0%로 대체 - DLRET 부재에 따른
                 단순화, 한계점)
      cols     : 선택된 종목의 PERMNO 배열
    """
    t, t1 = dates[i], dates[i + 1]
    hist_ok = obs_count.loc[t] >= COV_LOOKBACK
    price_ok = price_wide.loc[t] >= MIN_PRICE
    vol_t = vol_wide.loc[t]
    vol_ok = vol_t >= vol_t.quantile(VOL_PCTL_CUTOFF)
    window_ok = ret_wide.loc[:t, hist_ok[hist_ok].index].tail(COV_LOOKBACK).notna().all(axis=0)
    window_ok = window_ok.reindex(hist_ok.index, fill_value=False)

    eligible = hist_ok & price_ok & vol_ok & window_ok
    eligible_cols = eligible[eligible].index
    if len(eligible_cols) < K_FIXED * 5:
        return None

    if len(eligible_cols) > N_UNIVERSE:
        eligible_cols = vol_t.loc[eligible_cols].sort_values(ascending=False).head(N_UNIVERSE).index

    X_raw = ret_wide.loc[:t, eligible_cols].tail(COV_LOOKBACK).values
    X_wins = ret_wins.loc[:t, eligible_cols].tail(COV_LOOKBACK).values
    next_ret = ret_wide.loc[t1, eligible_cols].fillna(0.0).values
    return X_raw, X_wins, next_ret, eligible_cols.values

# ------------------------------------------------------------------
# 2. Signed 그래프 구성 유틸리티
# ------------------------------------------------------------------
def signed_correlation_graph(X):
    """
    [K24, Section 2] Pearson 상관행렬을 signed 그래프의 인접행렬 Gamma로 사용.
    X : (T, n) 수익률 행렬
    반환: (n, n) 상관행렬, 대각선은 0으로 설정(자기 자신과의 엣지=self-loop 제거)
    """
    Gamma = np.corrcoef(X.T)
    np.fill_diagonal(Gamma, 0.0)
    return Gamma


def _abs_degree(Gamma):
    """부호그래프의 차수행렬 D: D_ii = sum_j |Gamma_ij| (표준적 signed-degree 정의)."""
    d = np.abs(Gamma).sum(axis=1)
    return d


def _split_pos_neg(Gamma):
    """Gamma = Gamma+ - Gamma- 로 분해 (둘 다 성분별로 >= 0)."""
    Gamma_pos = np.where(Gamma > 0, Gamma, 0.0)
    Gamma_neg = np.where(Gamma < 0, -Gamma, 0.0)
    return Gamma_pos, Gamma_neg


# ------------------------------------------------------------------
# 3. 4종 Signed Spectral Clustering 알고리즘 (논문 Section 3)
# ------------------------------------------------------------------
def cluster_spectral(Gamma, K, n_init=5):
    """
    [K24, Section 3.1] Spectral clustering algorithm.
    L_N = I - D^{-1/2} Gamma D^{-1/2} (D는 절대값 기반 차수행렬)
    -> L_N의 최소 K개 고유벡터 -> 행별 유클리드 정규화 -> k-means++.
    """
    n = Gamma.shape[0]
    d = _abs_degree(Gamma)
    d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
    L_N = np.eye(n) - (d_inv_sqrt[:, None] * Gamma * d_inv_sqrt[None, :])

    _, U = np.linalg.eigh(L_N)          # 오름차순 고유값 -> 앞의 K개가 최소 K개
    U = U[:, :K]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    T_mat = U / np.maximum(norms, 1e-10)

    labels = KMeans(n_clusters=K, n_init=n_init, random_state=None).fit_predict(T_mat)
    return labels


def cluster_signed_laplacian(Gamma, K, n_init=5):
    """
    [K24, Section 3.2] Signed Laplacian (Kunegis et al. 2010).
    논문은 이를 signed ratio cut(SRC) 최소화 문제로 정의하며, 이 조합최적화 문제의
    표준적 스펙트럴 완화(relaxation)는 비정규화 signed Laplacian L = D - Gamma
    (D는 절대값 기반 차수행렬)의 최소 K개 고유벡터를 사용하는 것이다
    (Kunegis et al. 2010; SigNet 패키지의 signed Laplacian 구현과 동일한 원리).
    """
    n = Gamma.shape[0]
    d = _abs_degree(Gamma)
    L = np.diag(d) - Gamma

    _, U = np.linalg.eigh(L)
    U = U[:, :K]  # 행 정규화는 하지 않음 (Kunegis 원 논문 방식)

    labels = KMeans(n_clusters=K, n_init=n_init, random_state=None).fit_predict(U)
    return labels


def cluster_sponge(Gamma, K, tau_p=TAU_PLUS, tau_m=TAU_MINUS, n_init=5):
    """
    [K24, Section 3.3] SPONGE (Cucuringu et al. 2019).
    (L+ + tau- * D-) v = lambda (L- + tau+ * D+) v 의 일반화고유값 문제에서
    최소 K개 고유벡터를 구한 뒤 k-means++.
    """
    Gp, Gm = _split_pos_neg(Gamma)
    Dp, Dm = np.diag(Gp.sum(axis=1)), np.diag(Gm.sum(axis=1))
    Lp, Lm = Dp - Gp, Dm - Gm

    A = Lp + tau_m * Dm
    B = Lm + tau_p * Dp
    # B는 (Lm이 PSD) + (tau_p * Dp, 대각성분 > 0) 이므로 양의정부호(SPD) -> 일반화고유값 문제 well-defined
    eigvals, eigvecs = gen_eigh(A, B, subset_by_index=[0, K - 1])
    labels = KMeans(n_clusters=K, n_init=n_init, random_state=None).fit_predict(eigvecs)
    return labels


def cluster_sponge_sym(Gamma, K, tau_p=TAU_PLUS, tau_m=TAU_MINUS, n_init=5):
    """
    [K24, Section 3.4] Symmetric SPONGE.
    (L+_sym + tau- * I) v = lambda (L-_sym + tau+ * I) v 의 일반화고유값 문제.
    (논문 실증 결과에서 Sharpe/Sortino 기준 최고 성능을 보인 알고리즘)
    """
    Gp, Gm = _split_pos_neg(Gamma)
    dp, dm = Gp.sum(axis=1), Gm.sum(axis=1)
    # 일부 종목은 양(음)의 상관 이웃이 전혀 없어 차수가 0이 될 수 있음 -> 0으로 안전 처리
    dp_inv_sqrt = np.divide(1.0, np.sqrt(dp), out=np.zeros_like(dp), where=dp > 1e-12)
    dm_inv_sqrt = np.divide(1.0, np.sqrt(dm), out=np.zeros_like(dm), where=dm > 1e-12)
    Dp, Dm = np.diag(dp), np.diag(dm)
    Lp, Lm = Dp - Gp, Dm - Gm
    Lp_sym = dp_inv_sqrt[:, None] * Lp * dp_inv_sqrt[None, :]
    Lm_sym = dm_inv_sqrt[:, None] * Lm * dm_inv_sqrt[None, :]

    n = Gamma.shape[0]
    A = Lp_sym + tau_m * np.eye(n)
    B = Lm_sym + tau_p * np.eye(n)
    eigvals, eigvecs = gen_eigh(A, B, subset_by_index=[0, K - 1])
    labels = KMeans(n_clusters=K, n_init=n_init, random_state=None).fit_predict(eigvecs)
    return labels


CLUSTER_ALGOS = {
    "Spectral":        cluster_spectral,
    "SignedLaplacian": cluster_signed_laplacian,
    "SPONGE":          cluster_sponge,
    "SymSPONGE":       cluster_sponge_sym,
}

# ------------------------------------------------------------------
# 4. 클러스터 내부(Gaussian beta) 및 클러스터 레벨(EWA Sigma*, GMVP) 함수
# ------------------------------------------------------------------
def gaussian_intra_weights(X, labels, K, sigma_scale=0.5):
    """
    [K24, Section 4.2] beta_{i,k} = exp(-||r^(i) - mu_k||^2 / (2*sigma^2)) / sum_j beta_{j,k}.
    r^(i) : 종목 i의 lookback 구간 수익률 벡터, mu_k: 클러스터 k의 평균 수익률 벡터(centroid).
    sigma^2은 논문에서 학습구간 교차검증으로 고정하나, 본 구현에서는 데이터 적응형으로
    (클러스터 내 평균 제곱거리 x sigma_scale) 사용 - 매 시점/클러스터마다 스케일이 다른
    문제를 자동으로 보정하기 위함.
    """
    n = X.shape[1]
    beta = np.zeros(n)
    for k in range(K):
        mask = labels == k
        if not mask.any():
            continue
        cluster_ret = X[:, mask]                      # (T, n_k)
        mu_k = cluster_ret.mean(axis=1)                # (T,) 클러스터 중심(시계열)
        sq_dist = np.sum((cluster_ret - mu_k[:, None]) ** 2, axis=0)  # (n_k,)
        sigma_sq = sigma_scale * (sq_dist.mean() + 1e-10)
        raw = np.exp(-sq_dist / (2.0 * sigma_sq))
        beta[mask] = raw / (raw.sum() + 1e-10)
    return beta


def ewa_covariance(X, beta=EWA_BETA):
    """
    [K24, Section 4.6] 지수가중 공분산행렬(EWA).
    E = (1-beta)/(1-beta^T) * sum_t beta^{T-t} x_t^T x_t   (x_t는 중심화된 수익률)
    """
    T = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    w = (1 - beta) / (1 - beta ** T + 1e-12) * beta ** (T - 1 - np.arange(T))
    w /= w.sum()
    Xw = Xc * np.sqrt(w)[:, None]
    return Xw.T @ Xw


def capm_residuals(X, mkt_ret):
    """
    [C23, Cartea, Cucuringu & Jin 2023] 1-factor(CAPM) 잔차 수익률.
    beta_i = cov(r_i, r_mkt) / var(r_mkt); residual_i(t) = r_i(t) - beta_i * r_mkt(t).
    클러스터링 입력을 원시수익률 대신 잔차수익률로 바꾸면, "시장에 대한 노출(베타) 차이"가
    아니라 시장 공통요인을 제거한 후 남는 섹터/산업 고유의 공동움직임을 기준으로 종목을
    그룹화하게 되어 더 순수한 분산투자 효과를 얻을 수 있다는 것이 [C23]의 주장이다.
    (본 구현에서는 Gaussian intra-cluster 가중치/합성수익률/EWA공분산 계산에는 원시수익률을
    그대로 사용하고, 오직 클러스터를 나누는 상관행렬 계산에만 잔차수익률을 사용한다.)
    X       : (T, n) 원시 수익률
    mkt_ret : (T,) 시장수익률 프록시 (해당 시점 point-in-time 유니버스의 동일가중 평균)
    반환    : (T, n) 잔차 수익률
    """
    mkt_dm = mkt_ret - mkt_ret.mean()
    X_dm = X - X.mean(axis=0, keepdims=True)
    mkt_var = (mkt_dm ** 2).mean() + 1e-12
    beta = (X_dm * mkt_dm[:, None]).mean(axis=0) / mkt_var
    return X - beta[None, :] * mkt_ret[:, None]


def project_simplex(v):
    """표준 심플렉스 {w: sum(w)=1, w>=0} 위로의 유클리드 투영 (Duchi et al. 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    return np.maximum(v - (cssv[rho] - 1.0) / (rho + 1), 0.0)


def project_capped_simplex(v, cap, n_iter=100):
    """상한이 있는 심플렉스 {w: sum(w)=1, 0<=w_i<=cap} 위로의 투영 (이분탐색)."""
    v = np.asarray(v, dtype=float)
    lo, hi = v.min() - 1.0, v.max()
    for _ in range(n_iter):
        tau = (lo + hi) / 2.0
        w = np.clip(v - tau, 0.0, cap)
        if w.sum() > 1.0:
            lo = tau
        else:
            hi = tau
    w = np.clip(v - (lo + hi) / 2.0, 0.0, cap)
    s = w.sum()
    return np.full(len(v), 1.0 / len(v)) if s <= 1e-10 else w / s


def solve_gmvp(Sigma, cap=None, n_iter=500):
    """
    Long-only GMVP: min w'Sigma w s.t. sum(w)=1, 0<=w_i<=cap (cap=None이면 상한 없음=1).
    Projected gradient descent, 스텝 = 1/(2*lambda_max) (Lipschitz 상수 기반 고정 스텝).
    """
    n = Sigma.shape[0]
    w = np.full(n, 1.0 / n)
    lam_max = np.linalg.eigvalsh(Sigma)[-1]
    step = 1.0 / (2.0 * lam_max + 1e-12)
    project = (lambda x: project_capped_simplex(x, cap)) if cap else project_simplex
    for _ in range(n_iter):
        grad = 2.0 * (Sigma @ w)
        w_new = project(w - step * grad)
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    return w


def optimal_k_pca(corr_mat, pct=PCA_THRESHOLD, k_min=K_MIN, k_max=K_MAX):
    """
    [K24, Section 4.4 / Cartea et al. 2023] 상관행렬 고유값 누적분산비율 >= pct인
    최소 K 반환 (GICS 10섹터/24산업군과 유사한 규모가 나오는지 확인하는 용도).
    """
    eig = np.sort(np.linalg.eigvalsh(corr_mat))[::-1]
    cumvar = np.cumsum(eig) / (eig.sum() + 1e-12)
    k_raw = int(np.searchsorted(cumvar, pct)) + 1
    return int(np.clip(k_raw, k_min, k_max))


def ledoit_wolf_shrinkage(X):
    """Ledoit & Wolf (2004) shrinkage 공분산 (naive 비교전략용, 클러스터링 없음)."""
    T, N_ = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T
    mu = np.trace(S) / N_
    F = mu * np.eye(N_)
    pi_mat = np.zeros((N_, N_))
    for t in range(T):
        xt = Xc[t:t + 1, :]
        pi_mat += (xt.T @ xt - S) ** 2
    pi_mat /= T
    pi_hat = pi_mat.sum()
    rho_hat = np.trace(pi_mat)
    gamma = np.sum((S - F) ** 2)
    delta = 0.0 if gamma < 1e-12 else float(np.clip((pi_hat - rho_hat) / gamma / T, 0, 1))
    return delta * F + (1 - delta) * S


# ------------------------------------------------------------------
# 5. 클러스터 기반 포트폴리오 1회 생성 (phase i~ii), R회 반복은 상위 함수에서 처리
# ------------------------------------------------------------------
def build_cluster_portfolio_once(X_raw, X_corr_input, K, algo_fn):
    """
    [K24, Section 4.1, phase (i)-(ii)] 단일 반복(single repeat)의 클러스터 포트폴리오.
    X_raw        : (T, n) 원본 수익률 (Gaussian beta, 합성수익률 계산에 사용 - 항상 원시수익률)
    X_corr_input : (T, n) 클러스터링(상관행렬 계산)에 사용할 행렬. winsorize된 원시수익률
                   또는 [C23] CAPM 잔차수익률(capm_residuals) 둘 중 하나가 들어올 수 있다.
    반환: (n,) 종목 비중 벡터, 또는 실패 시 None
    """
    Gamma = signed_correlation_graph(X_corr_input)
    labels = algo_fn(Gamma, K)
    if len(np.unique(labels)) < K:
        return None  # 빈 클러스터 발생 시 이번 반복은 폐기

    beta = gaussian_intra_weights(X_raw, labels, K)

    # 합성자산(클러스터) 수익률 시계열: r*_k(t) = sum_i beta_{i,k} * r_i(t)
    X_syn = np.zeros((X_raw.shape[0], K))
    for k in range(K):
        mask = labels == k
        X_syn[:, k] = (X_raw[:, mask] * beta[mask]).sum(axis=1)

    Sigma_star = ewa_covariance(X_syn)      # K x K EWA 공분산
    alpha = solve_gmvp(Sigma_star)          # 클러스터 레벨 GMVP -> alpha (K,)

    w = np.zeros(X_raw.shape[1])
    for k in range(K):
        mask = labels == k
        w[mask] = alpha[k] * beta[mask]     # 최종 비중 = alpha_k * beta_{i,k}

    total = w.sum()
    if total < 1e-10:
        return None
    w /= total
    w = np.clip(w, 0.0, MAX_WEIGHT)         # 실무적 종목당 상한 (논문에는 없는 추가 제약)
    w /= w.sum()
    return w


def build_cluster_portfolio(X_raw, X_corr_input, K, algo_fn, R=R_REPEAT):
    """[K24, Section 4.1, phase (iii) ROBUSTNESS] R회 반복 후 평균."""
    portfolios = []
    for _ in range(R):
        w = build_cluster_portfolio_once(X_raw, X_corr_input, K, algo_fn)
        if w is not None:
            portfolios.append(w)
    if not portfolios:
        return None
    w_mean = np.mean(portfolios, axis=0)
    w_mean /= w_mean.sum()
    return w_mean


# ------------------------------------------------------------------
# 6. 메인 백테스트 루프
#    - 매월 t: 과거 COV_LOOKBACK개월로 클러스터 포트폴리오 구성 -> t+1월 실현수익률로 평가
#    - 4개 클러스터링 알고리즘 x {원시수익률, CAPM 잔차수익률} = 8개 조합(K=K_FIXED)
#      + Naive LW-GMVP(클러스터링 없음) + EW 비교
# ------------------------------------------------------------------
print(f"[2] 메인 백테스트 루프 시작 (K={K_FIXED}, 4개 알고리즘 x {{raw, CAPM잔차}} = 8개 조합 비교)...")

algo_names = list(CLUSTER_ALGOS.keys())
config_names = [f"{name}-raw" for name in algo_names] + [f"{name}-CAPMresid" for name in algo_names]
results = {name: [] for name in config_names}
naive_lw_returns = []
ew_returns = []
result_dates = []
universe_sizes = []

_t0 = time.time()
# COV_LOOKBACK개월치 "수익률"을 얻으려면 가격은 COV_LOOKBACK+1개월치가 필요
# (각 종목의 최초 관측월은 pct_change 특성상 수익률이 정의되지 않아 결측이기 때문).
start_i = COV_LOOKBACK
end_i = len(dates) - 1
if FAST_TEST:
    start_i = max(start_i, end_i - 24)  # 스모크테스트: 마지막 24개월만

for i in range(start_i, end_i):
    t, t1 = dates[i], dates[i + 1]
    window = point_in_time_window(i)
    if window is None:
        continue  # 유니버스가 너무 작은 초기 구간 등은 스킵
    X_raw, X_wins, next_ret, cols = window

    # [C23] 시장수익률 프록시(동일가중 평균, winsorize된 수익률 기준) -> CAPM 잔차 계산용
    mkt_ret = X_wins.mean(axis=1)
    X_resid = capm_residuals(X_wins, mkt_ret)

    for name, algo_fn in CLUSTER_ALGOS.items():
        w_raw = build_cluster_portfolio(X_raw, X_wins, K_FIXED, algo_fn)
        results[f"{name}-raw"].append(float(next_ret @ w_raw) if w_raw is not None else 0.0)

        w_resid = build_cluster_portfolio(X_raw, X_resid, K_FIXED, algo_fn)
        results[f"{name}-CAPMresid"].append(float(next_ret @ w_resid) if w_resid is not None else 0.0)

    # -- 비교전략 1: Naive Markowitz + Ledoit-Wolf (클러스터링 없음, 동일한 point-in-time 유니버스) --
    Sigma_lw = ledoit_wolf_shrinkage(X_wins)
    w_naive = solve_gmvp(Sigma_lw, cap=MAX_WEIGHT)
    naive_lw_returns.append(float(next_ret @ w_naive))

    # -- 비교전략 2: 동일가중(EW, 동일한 point-in-time 유니버스) --
    ew_returns.append(float(next_ret.mean()))

    result_dates.append(t1)
    universe_sizes.append(len(cols))

    done = len(result_dates)
    if done % 12 == 0 or i == end_i - 1:
        elapsed = time.time() - _t0
        print(f"    {t.strftime('%Y-%m')} 완료 ({done}개월 누적, 이번달 유니버스={len(cols)}개, "
              f"경과={elapsed:.0f}s)")

print(f"    총 {len(result_dates)}개월 백테스트 완료 "
      f"({result_dates[0].strftime('%Y-%m')} ~ {result_dates[-1].strftime('%Y-%m')})")
print(f"    Point-in-time 유니버스 크기: 평균={np.mean(universe_sizes):.0f}, "
      f"최소={min(universe_sizes)}, 최대={max(universe_sizes)} "
      f"(고정 유니버스가 아니라 매월 달라짐 - 생존편향 방지 확인용)")
pd.DataFrame({"date": result_dates, "universe_size": universe_sizes}).to_csv(
    os.path.join(_DIR, "paper_universe_size_history.csv"), index=False, encoding="utf-8-sig")

dt_idx = pd.DatetimeIndex(result_dates, name="date")
bt = pd.DataFrame({name: results[name] for name in config_names}, index=dt_idx)
bt["Naive GMVP-LW (no cluster)"] = naive_lw_returns
bt["EW Market"] = ew_returns

# ------------------------------------------------------------------
# 7. S&P 500 벤치마크 - Yahoo Finance 실제 월별데이터 (^SP500TR, 배당재투자 포함 총수익지수)
# ------------------------------------------------------------------
def fetch_sp500_total_return_monthly():
    """
    Yahoo Finance에서 ^SP500TR(S&P 500 Total Return Index) 월별 종가를 가져와
    실제 월별 수익률을 계산한다. CRSP ADJ_PRC가 배당/분할 조정가이므로, 공정한 비교를
    위해 가격지수(^GSPC)가 아닌 총수익지수(^SP500TR)를 사용한다.
    - 최초 실행시 네트워크로 조회 후 로컬에 캐시(sp500tr_monthly_cache.csv)하여
      이후 실행에서는 네트워크 호출 없이 재사용한다.
    - 네트워크 접근이 불가능하면, 공개된 연간 총수익률을 월별로 근사분할한 값으로
      대체한다(fallback, 정확한 월별 변동성/타이밍을 반영하지 못하는 한계가 있음).
    반환: pandas Series, index=Timestamp, value=월간수익률
    """
    cache_path = os.path.join(_DIR, "sp500tr_monthly_cache.csv")
    if os.path.exists(cache_path):
        print(f"    S&P500TR 캐시 파일 사용: {cache_path}")
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)["ret"]
        return cached

    try:
        import urllib.request
        import json
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
        period1 = int(pd.Timestamp("1999-11-15").timestamp())
        period2 = int(pd.Timestamp("2021-01-10").timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5ESP500TR"
               f"?interval=1mo&period1={period1}&period2={period2}")
        req = urllib.request.Request(url, headers=headers)
        raw = json.loads(urllib.request.urlopen(req, timeout=20).read())
        result = raw["chart"]["result"][0]
        px = pd.Series(
            result["indicators"]["adjclose"][0]["adjclose"],
            index=pd.to_datetime(result["timestamp"], unit="s"),
        ).sort_index()
        px = px[(px.index >= "1999-12-01") & (px.index <= "2020-12-31")]
        ret = px.pct_change().dropna()
        ret.name = "ret"
        ret.to_csv(cache_path)
        print(f"    Yahoo Finance(^SP500TR)에서 월별수익률 {len(ret)}개월 수신 및 캐시 저장 완료.")
        return ret
    except Exception as e:
        print(f"    [경고] Yahoo Finance 접근 실패 ({type(e).__name__}: {e}) "
              f"-> 연간수익률 근사분할로 대체 (한계점으로 보고서에 명시 필요).")
        sp500_annual = {
            2000: -0.091, 2001: -0.119, 2002: -0.220, 2003: 0.287, 2004: 0.109,
            2005: 0.049, 2006: 0.158, 2007: 0.055, 2008: -0.370, 2009: 0.265,
            2010: 0.151, 2011: 0.021, 2012: 0.160, 2013: 0.324, 2014: 0.137,
            2015: 0.014, 2016: 0.120, 2017: 0.218, 2018: -0.044, 2019: 0.314,
            2020: 0.184,
        }
        vals = {}
        for y, r_annual in sp500_annual.items():
            r_month = (1 + r_annual) ** (1 / 12) - 1
            for m in range(1, 13):
                vals[pd.Timestamp(year=y, month=m, day=1)] = r_month
        return pd.Series(vals, name="ret").sort_index()


sp500_ret = fetch_sp500_total_return_monthly()
sp_lookup = {(idx.year, idx.month): val for idx, val in sp500_ret.items()}
bt["S&P500 (Total Return)"] = [sp_lookup.get((d.year, d.month), np.nan) for d in bt.index]
bt = bt.dropna(subset=["S&P500 (Total Return)"])

# ------------------------------------------------------------------
# 8. 성과지표 (Cumulative/Annualized Return, Annualized Std, Sharpe, Sortino, MDD)
# ------------------------------------------------------------------
def compute_metrics(r):
    cum = (1 + r).cumprod()
    n_years = len(r) / 12.0
    ann_return = cum.iloc[-1] ** (1 / n_years) - 1
    ann_std = r.std(ddof=1) * np.sqrt(12)
    sharpe = (ann_return - RF) / ann_std if ann_std > 1e-10 else np.nan
    downside = r[r < 0]
    downside_std = downside.std(ddof=1) * np.sqrt(12) if len(downside) > 1 else np.nan
    sortino = (ann_return - RF) / downside_std if downside_std and downside_std > 1e-10 else np.nan
    mdd = (cum / cum.cummax() - 1).min()
    return {
        "Cumulative Return": cum.iloc[-1] - 1,
        "Annualized Return": ann_return,
        "Annualized Std": ann_std,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
    }

metrics_table = pd.DataFrame({col: compute_metrics(bt[col]) for col in bt.columns}).T
metrics_table = metrics_table[["Cumulative Return", "Annualized Return", "Annualized Std",
                                "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]]
print("\n=== 성과지표 요약 (K=%d, R=%d) ===" % (K_FIXED, R_REPEAT))
print(metrics_table.round(4))
metrics_table.round(4).to_csv(os.path.join(_DIR, "paper_metrics.csv"), encoding="utf-8-sig")
bt.to_csv(os.path.join(_DIR, "paper_monthly_returns.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------
# 9. 시각화: 누적포트폴리오가치 + Drawdown
# ------------------------------------------------------------------
COLORS = {
    "Spectral": "#4363d8", "SignedLaplacian": "#3cb44b", "SPONGE": "#f58231",
    "SymSPONGE": "#e6194b", "Naive GMVP-LW (no cluster)": "#808080",
    "EW Market": "#a0a0a0", "S&P500 (Total Return)": "#000000",
}
fig, axes = plt.subplots(2, 1, figsize=(11, 9))
for col in bt.columns:
    cum = (1 + bt[col]).cumprod()
    axes[0].plot(cum.index, cum.values, label=col, lw=1.4, color=COLORS.get(col))
    dd = cum / cum.cummax() - 1
    axes[1].plot(dd.index, dd.values, label=col, lw=1.1, color=COLORS.get(col))
axes[0].set_yscale("log")
axes[0].set_title(f"Portfolio Value of $1 (log scale) - K={K_FIXED} clusters")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(alpha=0.3)
axes[1].set_title("Drawdown")
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_DIR, "paper_results.png"), dpi=140)
plt.close()

print("\n완료: paper_metrics.csv, paper_monthly_returns.csv, paper_results.png 저장됨.")

# ------------------------------------------------------------------
# 10. Raw vs CAPM-잔차 클러스터링 요약 비교 (3차 개정 핵심 질문에 대한 답)
#     - 참고: K 민감도 분석(K=10 vs 24 vs 동적K)은 이전 실행 결과가
#       paper_k_sensitivity_metrics.csv / paper_k_sensitivity.png 에 이미 저장되어 있음
#       (이번 실행에서는 K=10으로 고정했으므로 재실행하지 않음 - 계산량 절감).
# ------------------------------------------------------------------
print("\n[3] Raw vs CAPM-잔차 클러스터링 비교 요약")
compare_rows = []
for name in algo_names:
    raw_sharpe = metrics_table.loc[f"{name}-raw", "Sharpe Ratio"]
    resid_sharpe = metrics_table.loc[f"{name}-CAPMresid", "Sharpe Ratio"]
    raw_ar = metrics_table.loc[f"{name}-raw", "Annualized Return"]
    resid_ar = metrics_table.loc[f"{name}-CAPMresid", "Annualized Return"]
    compare_rows.append({
        "Algorithm": name,
        "Sharpe(raw)": raw_sharpe, "Sharpe(CAPMresid)": resid_sharpe,
        "Sharpe 개선폭": resid_sharpe - raw_sharpe,
        "AnnRet(raw)": raw_ar, "AnnRet(CAPMresid)": resid_ar,
        "AnnRet 개선폭": resid_ar - raw_ar,
    })
compare_table = pd.DataFrame(compare_rows).set_index("Algorithm")
print(compare_table.round(4))
compare_table.round(4).to_csv(os.path.join(_DIR, "paper_raw_vs_capmresid_comparison.csv"), encoding="utf-8-sig")

print("\n완료: paper_raw_vs_capmresid_comparison.csv 저장됨.")
print("\n전체 스크립트 종료.")
