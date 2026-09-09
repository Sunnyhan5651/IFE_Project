"""
IFE Term Project - 포트폴리오 백테스트 스크립트
=================================================
전략 파이프라인:
  1) 데이터 클리닝 (EDA에서 확인한 가격 이상치 종목 12개 제외, EXCHCD 1/2/3만 사용)
  2) Point-in-time 유니버스 구성 (생존편향 방지: 각 시점에 실제 존재했던 종목만 사용)
  3) 모멘텀(12-1개월) 기반 종목 선정 (Jegadeesh & Titman, 1993)
  4) Ledoit-Wolf shrinkage로 공분산 행렬 추정 (Ledoit & Wolf, 2004)
  5) Long-only GMVP / MSRP(접선 포트폴리오) 평균-분산 최적화 (scipy SLSQP)
  6) 변동성 관리 오버레이 적용 (Moreira & Muir, 2017) - 목표 변동성으로 익스포저 조절
  7) 매월 리밸런싱 백테스트 실행 및 성과지표 5종 산출
     (Cumulative Return, Annualized Return, Annualized Std, Sharpe Ratio, MDD)
  8) S&P 500 벤치마크와 비교

주의(제약사항):
  - 본 데이터셋에는 SHROUT(발행주식수)와 DLRET(상장폐지 수익률) 필드가 없음.
    -> 시가총액 가중 불가(동일가중/모멘텀 선정 기반으로 대체),
       상장폐지 시 정확한 손실(delisting return) 반영 불가(단순화하여 0% 처리, 아래 주석 참고).
  - S&P 500 벤치마크는 도구 접근 제약으로 완전한 월별 시계열을 확보하지 못해
    "연간 수익률을 월별로 균등 분할"한 근사치를 사용함. 최종 제출 시에는
    CRSP sprtrn(S&P500 월간 수익률) 필드를 직접 사용할 것을 권장.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 참고: 샌드박스 환경에 scipy/scikit-learn이 설치되어 있지 않고 인터넷 접근도 불가하여
# (1) Ledoit-Wolf shrinkage 공분산과 (2) long-only 평균-분산 최적화(GMVP/MSRP)를
# 모두 NumPy만으로 직접 구현함(projected gradient descent/ascent, capped-simplex 투영).
# 이는 scipy.optimize.minimize(SLSQP)와 동일한 문제를 푸는 표준적인 수치최적화 기법임.

# 재현성을 위한 랜덤시드 고정 (최적화 초기값 등에 사용)
np.random.seed(42)

# ------------------------------------------------------------------
# 0. 파라미터 설정
# ------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_DIR, "IFE_term_stock_data.csv")
MIN_PRICE = 5.0            # 페니스톡 제외 기준 (ADJ_PRC >= $5)
VOL_PCTL_CUTOFF = 0.20     # 유동성 하위 20% 종목 제외 (VOL 기준)
TOP_N = 50                 # 모멘텀 상위 N개 종목 선정
MOM_LOOKBACK = 12          # 모멘텀 형성 기간(개월)
MOM_SKIP = 1               # 최근 1개월 제외 (단기 반전효과 회피, "12-1" 모멘텀)
COV_LOOKBACK = 36          # 공분산 추정에 사용할 과거 개월 수
MIN_HISTORY = 36           # 종목이 백테스트에 편입되기 위한 최소 관측월수
MAX_WEIGHT = 0.10          # 종목당 최대 비중 (과도한 집중 방지)
TARGET_VOL = 0.15          # 변동성 관리 오버레이의 연율화 목표 변동성 (15%)
VOL_LOOKBACK = 12          # 실현 변동성 추정에 사용할 과거 개월 수
LEV_CAP = 1.5              # 변동성 오버레이 최대 레버리지
LEV_FLOOR = 0.2            # 변동성 오버레이 최소 익스포저
RF = 0.0                   # 무위험수익률 0 가정 (프로젝트 가이드라인)

# ------------------------------------------------------------------
# 1. 데이터 로드 및 클리닝 (EDA 결과 재사용)
# ------------------------------------------------------------------
print("[1] 데이터 로드 중...")
df = pd.read_csv(DATA_PATH)                       # CSV 로드
df["date"] = pd.to_datetime(df["date"])           # 날짜 타입 변환
df = df.sort_values(["PERMNO", "date"])           # 종목-날짜순 정렬

# EDA에서 확인한 가격 이상치 종목(조정계수 오류 추정, ADJ_PRC 최대 2.1조 수준) 제외
bad_permno = df.loc[df["ADJ_PRC"] > 1e6, "PERMNO"].unique()
# 가이드라인상 허용된 거래소코드(1=NYSE, 2=AMEX, 3=NASDAQ)만 사용
clean = df[~df["PERMNO"].isin(bad_permno) & df["EXCHCD"].isin([1, 2, 3])].copy()
print(f"    정제 후 행 수: {len(clean):,} (원본 {len(df):,})")

# 종목별 월간 수익률 계산 (가격 변화율)
clean["ret"] = clean.groupby("PERMNO")["ADJ_PRC"].pct_change()

# ------------------------------------------------------------------
# 2. Wide-format 매트릭스 구성 (행=날짜, 열=PERMNO)
#    이후 모든 시점별 연산을 벡터화하기 위한 준비 단계
# ------------------------------------------------------------------
print("[2] Wide 매트릭스 구성 중...")
ret_wide = clean.pivot(index="date", columns="PERMNO", values="ret").sort_index()
price_wide = clean.pivot(index="date", columns="PERMNO", values="ADJ_PRC").sort_index()
vol_wide = clean.pivot(index="date", columns="PERMNO", values="VOL").sort_index()
dates = ret_wide.index  # 전체 월말 날짜 리스트 (2000-01 ~ 2020-12)

# 각 시점까지 종목이 관측된 개월 수 (point-in-time 히스토리 길이) - 생존편향 방지용
obs_count = price_wide.notna().cumsum()

# EDA에서 확인한 극단치 문제(월 수익률 +500%~+1988%인 뉴스성 급등 종목 100여 건) 대응:
# 공분산(Sigma)·기대수익률(mu) "추정"에만 사용할 winsorize된 수익률 행렬을 별도로 생성.
# (실제 포트폴리오 실현수익률 계산에는 원본 ret_wide를 그대로 사용 - 실현손익은 왜곡하지 않음)
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99
ret_wide_wins = ret_wide.clip(
    lower=ret_wide.quantile(WINSOR_LOW, axis=1), upper=ret_wide.quantile(WINSOR_HIGH, axis=1), axis=0
)

# ------------------------------------------------------------------
# 3. 모멘텀 스코어 계산 (12-1개월): price(t-1)/price(t-12) - 1
#    t-1, t-12 시점 가격만 사용하므로 t 시점 시점에서 완전히 관측 가능(lookahead 없음)
# ------------------------------------------------------------------
print("[3] 모멘텀 스코어 계산 중...")
mom = price_wide.shift(MOM_SKIP) / price_wide.shift(MOM_LOOKBACK) - 1.0

# ------------------------------------------------------------------
# 4. Ledoit-Wolf Shrinkage 공분산 추정 함수 (직접 구현)
#    sklearn 미사용 환경이므로 Ledoit & Wolf (2004) "Honey, I Shrunk the
#    Sample Covariance Matrix" 의 등방적(스케일된 항등행렬) 타겟 축소 공식을 그대로 구현.
#    Sigma_hat = delta * F + (1 - delta) * S
#      S = 표본공분산, F = mu*I (mu = tr(S)/N), delta = 축소강도(0~1)
# ------------------------------------------------------------------
def ledoit_wolf_shrinkage(X):
    """
    X : (T x N) 수익률 행렬 (결측 없이 정제된 상태로 입력)
    반환: (N x N) shrinkage 공분산 행렬
    """
    T, N = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)          # 평균 제거(demean)
    S = (Xc.T @ Xc) / T                               # 표본공분산(모수 추정, T로 나눔)
    mu = np.trace(S) / N                              # 타겟(등방 행렬)의 스케일
    F = mu * np.eye(N)                                # 축소 타겟 행렬

    # 표본공분산 각 원소의 점근분산 추정 (pi_hat)
    # sum_t (x_t x_t' - S)^2 의 원소별 평균
    pi_mat = np.zeros((N, N))
    for t in range(T):
        xt = Xc[t:t+1, :]                             # 1 x N
        outer = xt.T @ xt                             # N x N
        pi_mat += (outer - S) ** 2
    pi_mat /= T
    pi_hat = pi_mat.sum()

    # 등방 타겟(diagonal target)의 경우 off-diagonal target=0 이므로
    # rho_hat(교차보정항)은 대각원소의 점근분산 합으로 근사
    rho_hat = np.trace(pi_mat)

    gamma_hat = np.sum((S - F) ** 2)                  # ||S - F||_F^2
    if gamma_hat < 1e-12:
        delta = 0.0
    else:
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / T))

    Sigma_hat = delta * F + (1 - delta) * S
    return Sigma_hat

# ------------------------------------------------------------------
# 5. Long-only 평균-분산 최적화 함수 (GMVP, MSRP)
# ------------------------------------------------------------------
def project_capped_simplex(v, u, n_iter=100):
    """
    Euclidean projection of vector v onto the "capped simplex"
        { w : sum(w) = 1, 0 <= w_i <= u }
    구현 방법: w(tau) = clip(v - tau, 0, u) 는 tau에 대해 단조감소하는 sum을 가지므로,
    sum(w(tau)) = 1 이 되는 tau를 이분탐색(bisection)으로 찾는다.
    (Duchi et al. 2008의 심플렉스 투영 알고리즘을 상한 제약(capped)까지 확장한 버전)
    """
    v = np.asarray(v, dtype=float)
    lo, hi = v.min() - 1.0, v.max()   # tau의 하한/상한 (sum(w) >= 1 및 <= 1을 보장하는 범위)
    for _ in range(n_iter):
        tau = (lo + hi) / 2.0
        w = np.clip(v - tau, 0.0, u)
        s = w.sum()
        if s > 1.0:
            lo = tau           # sum이 너무 크면 tau를 키워야 함
        else:
            hi = tau           # sum이 너무 작으면 tau를 줄여야 함
    tau = (lo + hi) / 2.0
    w = np.clip(v - tau, 0.0, u)
    total = w.sum()
    if total <= 1e-10:
        # 극단적 수치오류 발생 시 동일가중으로 대체(안전장치)
        return np.full(len(v), 1.0 / len(v))
    return w / total  # 잔여 수치오차 보정을 위한 재정규화


def solve_gmvp(Sigma, max_weight=MAX_WEIGHT, n_iter=500):
    """
    Long-only 글로벌 최소분산 포트폴리오(GMVP).
    목적함수 f(w) = w'Sigma w 는 볼록함수이므로,
    projected gradient descent (고정 스텝 1/L, L=2*lambda_max(Sigma))로 전역최적해에 수렴.
    (scipy 미사용 환경에 맞춘 SLSQP의 대체 구현)
    """
    N = Sigma.shape[0]
    w = np.full(N, 1.0 / N)                              # 초기값: 동일가중
    lam_max = np.linalg.eigvalsh(Sigma)[-1]               # Sigma의 최대 고유값
    step = 1.0 / (2.0 * lam_max + 1e-12)                  # 립시츠 상수 기반 고정 스텝
    for _ in range(n_iter):
        grad = 2.0 * (Sigma @ w)                          # f(w)의 그래디언트
        w_new = project_capped_simplex(w - step * grad, max_weight)
        if np.max(np.abs(w_new - w)) < 1e-10:             # 수렴 판정
            w = w_new
            break
        w = w_new
    return w


def solve_msrp(mu, Sigma, max_weight=MAX_WEIGHT, rf=RF, n_iter=500):
    """
    Long-only 최대 샤프비율(접선) 포트폴리오(MSRP).
    샤프비율 S(w) = (w'mu - rf) / sqrt(w'Sigma w) 는 일반적으로 비볼록이므로,
    projected gradient ascent + backtracking line search로 근사 최적해를 구함.
    (scipy 미사용 환경에 맞춘 SLSQP의 대체 구현)
    """
    N = Sigma.shape[0]
    w = np.full(N, 1.0 / N)

    def sharpe(w):
        num = w @ mu - rf
        den = np.sqrt(max(w @ Sigma @ w, 1e-12))
        return num / den

    def grad_sharpe(w):
        num = w @ mu - rf
        den = np.sqrt(max(w @ Sigma @ w, 1e-12))
        return mu / den - num * (Sigma @ w) / (den ** 3)

    w_best, s_best = w.copy(), sharpe(w)
    step = 1.0
    for _ in range(n_iter):
        grad = grad_sharpe(w)
        s0 = sharpe(w)
        improved = False
        step_try = step
        for _ in range(30):                               # 백트래킹 라인서치
            w_new = project_capped_simplex(w + step_try * grad, max_weight)
            s_new = sharpe(w_new)
            if s_new > s0 + 1e-14:
                improved = True
                break
            step_try *= 0.5
        if not improved:
            break
        w = w_new
        step = step_try * 1.5                              # 다음 스텝 크기 소폭 증가
        if s_new > s_best:
            w_best, s_best = w.copy(), s_new
    return w_best

print("[설정 완료] 파라미터 및 함수 정의 끝.")

# ------------------------------------------------------------------
# 6. 월별 리밸런싱 백테스트 메인 루프
#    - 매월 말(t) 시점에 point-in-time 유니버스에서 모멘텀 상위 TOP_N 종목 선정
#    - 과거 COV_LOOKBACK개월 수익률로 Ledoit-Wolf 공분산 추정 -> GMVP, MSRP 산출
#    - MSRP에 변동성 관리 오버레이 적용(전략 본체) -> 다음달(t+1) 실현수익률 계산
#    - 비교 대상: GMVP(오버레이 없음), 1/N(모멘텀 상위종목 동일가중), 전체 유니버스 동일가중(EW)
# ------------------------------------------------------------------
print("[4] 백테스트 메인 루프 시작...")

strategy_returns = []       # 전략 본체: 모멘텀+MSRP(LW)+변동성오버레이
gmvp_returns = []            # 비교①: 모멘텀+GMVP(LW), 오버레이 없음
naive_returns = []           # 비교②: 모멘텀 상위종목 1/N 동일가중 (DeMiguel et al. 2009 벤치마크)
ew_market_returns = []       # 비교③: 필터 통과 전체 유니버스 동일가중 (EDA의 시장 프록시)
result_dates = []            # 각 수익률이 실현된 "다음달" 날짜

# 전략 본체의 실현수익률 히스토리 (변동성 오버레이의 trailing vol 추정에 사용, 룩어헤드 방지 위해
# 반드시 과거 실현치만 사용)
strat_hist = []

for i in range(len(dates) - 1):
    t = dates[i]        # 형성(리밸런싱) 시점
    t1 = dates[i + 1]   # 실현 시점 (다음달)

    # -- point-in-time 유니버스 필터 --
    # (a) 최소 관측 히스토리 (생존편향 방지: 그 시점까지 실제 존재/관측된 종목만)
    hist_ok = obs_count.loc[t] >= MIN_HISTORY
    # (b) 페니스톡 제외
    price_ok = price_wide.loc[t] >= MIN_PRICE
    # (c) 유동성 하위 20% 제외 (해당 월 거래량 VOL 기준)
    vol_t = vol_wide.loc[t]
    vol_cut = vol_t.quantile(VOL_PCTL_CUTOFF)
    vol_ok = vol_t >= vol_cut
    # (d) 모멘텀 스코어가 유효(NaN 아님)해야 함 (충분한 가격 히스토리 보유)
    mom_ok = mom.loc[t].notna()

    eligible = hist_ok & price_ok & vol_ok & mom_ok
    eligible_permnos = eligible[eligible].index

    if len(eligible_permnos) < TOP_N:
        continue  # 초기 구간 등 유니버스가 너무 작으면 스킵

    # -- 모멘텀 상위 TOP_N 종목 선정 --
    mom_t = mom.loc[t, eligible_permnos].sort_values(ascending=False)
    selected = mom_t.head(TOP_N).index

    # -- 과거 COV_LOOKBACK개월 수익률 행렬 준비 (결측 없는 종목만, winsorize된 수익률 사용) --
    hist_window = ret_wide_wins.loc[:t, selected].tail(COV_LOOKBACK)
    valid_cols = hist_window.columns[hist_window.notna().all(axis=0)]
    if len(valid_cols) < 10:
        continue  # 공분산 추정에 필요한 최소 종목 수 미달 시 스킵
    X = hist_window[valid_cols].values  # T x N

    # -- Ledoit-Wolf shrinkage 공분산 추정 --
    Sigma = ledoit_wolf_shrinkage(X)
    mu_hist = X.mean(axis=0)  # 기대수익률 프록시: 과거 실현평균수익률 (MSRP용)

    # -- GMVP, MSRP 가중치 계산 --
    w_gmvp = solve_gmvp(Sigma)
    w_msrp = solve_msrp(mu_hist, Sigma)

    weight_map_gmvp = pd.Series(w_gmvp, index=valid_cols)
    weight_map_msrp = pd.Series(w_msrp, index=valid_cols)

    # -- 다음달(t1) 실현수익률로 포트폴리오 수익률 계산 --
    # 상장폐지 등으로 t1 시점 수익률이 없는 종목은 0%(현금 대체)로 처리하고
    # 나머지 종목 비중으로 재정규화하지 않음(=해당 비중만큼 무위험 현금 보유로 단순화).
    # * 데이터에 DLRET(상장폐지수익률) 필드가 없어 정확한 상장폐지 손실 반영이 불가능한 데 따른
    #   보수적 단순화이며, 실제 상장폐지 손실을 과소평가할 수 있음(한계점으로 보고서에 명시 필요).
    next_ret = ret_wide.loc[t1]

    def port_return(weight_map):
        r = next_ret.reindex(weight_map.index).fillna(0.0)
        return float((weight_map * r).sum())

    r_gmvp = port_return(weight_map_gmvp)
    r_msrp_raw = port_return(weight_map_msrp)

    # -- 변동성 관리 오버레이 (전략 본체에만 적용) --
    # 과거 VOL_LOOKBACK개월 "전략 자체의 실현수익률"의 연율화 표준편차로 스케일 조정.
    # 오직 과거(t 시점까지의) 실현치만 사용하므로 룩어헤드 편향 없음.
    if len(strat_hist) >= VOL_LOOKBACK:
        trailing_vol = np.std(strat_hist[-VOL_LOOKBACK:], ddof=1) * np.sqrt(12)
        if trailing_vol > 1e-8:
            scale = TARGET_VOL / trailing_vol
        else:
            scale = 1.0
        scale = float(np.clip(scale, LEV_FLOOR, LEV_CAP))
    else:
        scale = 1.0  # 히스토리 부족한 초기 구간은 스케일 미적용

    r_strategy = scale * r_msrp_raw  # 스케일 초과분/부족분은 무위험(0%) 현금으로 가정
    strat_hist.append(r_strategy)

    # -- 비교 벤치마크 ② : 모멘텀 상위종목 1/N 동일가중 --
    r_naive = float(next_ret.reindex(selected).fillna(0.0).mean())

    # -- 비교 벤치마크 ③ : 필터 통과 전체 유니버스 동일가중 --
    r_ew_market = float(next_ret.reindex(eligible_permnos).fillna(0.0).mean())

    strategy_returns.append(r_strategy)
    gmvp_returns.append(r_gmvp)
    naive_returns.append(r_naive)
    ew_market_returns.append(r_ew_market)
    result_dates.append(t1)

print(f"    총 {len(result_dates)}개월 백테스트 완료 "
      f"({result_dates[0].strftime('%Y-%m')} ~ {result_dates[-1].strftime('%Y-%m')})")

# 결과를 하나의 DataFrame으로 정리
bt = pd.DataFrame({
    "Momentum+MSRP(LW)+VolOverlay": strategy_returns,
    "Momentum+GMVP(LW)": gmvp_returns,
    "Momentum 1/N (naive)": naive_returns,
    "EW Market": ew_market_returns,
}, index=pd.DatetimeIndex(result_dates, name="date"))

# ------------------------------------------------------------------
# 7. S&P 500 벤치마크 구성 (제약사항: 완전한 월별 시계열 확보 불가)
#    아래 연간 총수익률(dividend 포함 total return, %) 은 공개된 S&P500 연간수익률을 사용.
#    각 연도의 수익률을 12개월에 걸쳐 "기하평균 기준 균등 분할"하여 근사 월간수익률을 생성.
#    -> 실제 월별 변동성/타이밍은 반영하지 못하는 근사치이므로, 최종 보고서 제출 시에는
#       CRSP의 sprtrn(S&P500 월간수익률) 필드로 반드시 교체할 것을 권장.
# ------------------------------------------------------------------
sp500_annual = {
    2000: -0.091, 2001: -0.119, 2002: -0.220, 2003: 0.287, 2004: 0.109,
    2005: 0.049, 2006: 0.158, 2007: 0.055, 2008: -0.370, 2009: 0.265,
    2010: 0.151, 2011: 0.021, 2012: 0.160, 2013: 0.324, 2014: 0.137,
    2015: 0.014, 2016: 0.120, 2017: 0.218, 2018: -0.044, 2019: 0.314,
    2020: 0.184,
}
sp500_monthly = {}
for year, r_annual in sp500_annual.items():
    r_month = (1 + r_annual) ** (1 / 12) - 1  # 연간 수익률을 월복리 기준으로 균등 분할(근사)
    for m in range(1, 13):
        sp500_monthly[(year, m)] = r_month

# 주의: 원본 데이터의 'date'는 달력상 월말이 아니라 "해당 월의 마지막 거래일"이므로
# (예: 2000-01-31, 2000-02-29, ... 실제 거래소 영업일 기준), Timestamp를 그대로 키로 매칭하면
# 대부분 불일치가 발생함. (year, month) 튜플로 매칭해야 정확히 대응됨.
bt["S&P500 (approx.)"] = [sp500_monthly.get((d.year, d.month), np.nan) for d in bt.index]
bt = bt.dropna(subset=["S&P500 (approx.)"])  # 벤치마크 데이터 있는 구간만 비교

print(f"    벤치마크 포함 최종 비교 구간: {bt.index[0].strftime('%Y-%m')} ~ {bt.index[-1].strftime('%Y-%m')} "
      f"({len(bt)}개월)")

# ------------------------------------------------------------------
# 8. 성과지표 5종 계산 함수
# ------------------------------------------------------------------
def compute_metrics(r):
    """월간수익률 시리즈(r)로부터 5개 성과지표 계산"""
    cum_curve = (1 + r).cumprod()               # 누적 성장곡선 ($1 투자 기준)
    cumulative_return = cum_curve.iloc[-1] - 1   # 총 누적수익률
    n_years = len(r) / 12.0
    annualized_return = cum_curve.iloc[-1] ** (1 / n_years) - 1  # 연율화수익률(기하평균)
    annualized_std = r.std(ddof=1) * np.sqrt(12)                  # 연율화 표준편차
    sharpe = (annualized_return - RF) / annualized_std if annualized_std > 0 else np.nan
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1.0
    mdd = drawdown.min()                          # 최대낙폭(가장 큰 음수)
    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Std": annualized_std,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
    }

metrics_table = pd.DataFrame({col: compute_metrics(bt[col]) for col in bt.columns}).T
metrics_table = metrics_table[["Cumulative Return", "Annualized Return", "Annualized Std",
                                "Sharpe Ratio", "Max Drawdown"]]

print("\n=== 성과지표 요약 ===")
print(metrics_table.round(4))
metrics_table.round(4).to_csv(os.path.join(_DIR, "backtest_metrics.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------
# 9. 포트폴리오 가치(누적수익률) 그래프
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for col in bt.columns:
    cum_curve = (1 + bt[col]).cumprod()
    ax.plot(cum_curve.index, cum_curve.values, label=col, lw=1.4)
ax.set_yscale("log")
ax.set_title("Portfolio Value Growth of $1 (log scale)")
ax.set_ylabel("Portfolio Value ($)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_DIR, "portfolio_value.png"), dpi=140)
plt.close()

# Drawdown 그래프 (전략 본체 기준)
fig, ax = plt.subplots(figsize=(11, 4))
for col in bt.columns:
    cum_curve = (1 + bt[col]).cumprod()
    dd = cum_curve / cum_curve.cummax() - 1
    ax.plot(dd.index, dd.values, label=col, lw=1.2)
ax.set_title("Drawdown")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_DIR, "drawdown.png"), dpi=140)
plt.close()

bt.to_csv(os.path.join(_DIR, "backtest_monthly_returns.csv"), encoding="utf-8-sig")
print("\n완료: backtest_metrics.csv, backtest_monthly_returns.csv, "
      "portfolio_value.png, drawdown.png 저장됨.")
