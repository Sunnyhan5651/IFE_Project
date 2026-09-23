"""
데이터 로드·전처리, 매월 유니버스 선정, 추정용 통계 도구.

전처리 (백테스트 전에 한 번 수행)
  1. 거래소 필터: EXCHCD ∈ {1, 2, 3} (과제 지침)
  2. 가격 오류 제거: ADJ_PRC가 비정상적으로 큰 PERMNO 제외
     (ADJ_PRC는 과거로 소급 조정되어 과거 가격이 낮아지기만 하므로, 큰 값은 오류로 판단)
  3. 수익률: 연속된 두 달 가격으로만 계산 (중간에 빈 달이 있으면 결측)
  4. 급등-반전 오류 제거 (Ince & Porter 2006 방식):
     r_t > 300% 이고 (1+r_t)(1+r_{t+1}) − 1 < 50% 이면 r_t, r_{t+1}을 오류로 보고 결측 처리
  5. NASDAQ 거래량 이중집계 보정 (Gao & Ritter 2010):
     2001-01 이전 ÷2.0, 2001-02~2001-12 ÷1.8, 2002~2003 ÷1.6

적용하지 않는 것
  - 최소 주가($5) 필터: ADJ_PRC는 이후의 분할·배당으로 소급 조정된 가격이라,
    과거 시점의 ADJ_PRC로 거르면 미래 정보가 들어간다 (나중에 분할한 종목이 과거에 저가주로 걸러짐).

모든 유니버스·추정 함수는 시점 t까지의 정보만 사용한다.
"""

from dataclasses import dataclass  # 전처리 결과 묶음
import numpy as np                 # 수치 계산
import pandas as pd                # 표 형태 데이터 처리


@dataclass
class Panel:
    returns: pd.DataFrame     # 월 × PERMNO 수익률 (오류 제거 후)
    price: pd.DataFrame       # 월 × PERMNO 조정 가격
    volume: pd.DataFrame      # 월 × PERMNO 거래량 (NASDAQ 보정 후)
    exchange: pd.DataFrame    # 월 × PERMNO 거래소 코드
    error_mask: pd.DataFrame  # 급등-반전 오류로 결측 처리된 위치 (True)
    info: dict                # 전처리 요약 통계


def _nasdaq_volume_divisor(dates):
    """날짜별 NASDAQ 거래량 나눗수 (Gao & Ritter 2010)."""
    d = pd.Series(1.0, index=dates)                                   # 기본값 1 (2004년 이후)
    d[dates < pd.Timestamp("2004-01-01")] = 1.6                       # 2002~2003
    d[dates < pd.Timestamp("2002-01-01")] = 1.8                       # 2001-02~2001-12
    d[dates < pd.Timestamp("2001-02-01")] = 2.0                       # 2001-01 이전
    return d                                                          # 날짜별 나눗수


def load_panel(cfg):
    """CSV를 읽어 전처리된 Panel을 만든다."""
    cols = ["PERMNO", "date", "EXCHCD", "ADJ_PRC", "VOL"]                   # 전략에 필요한 열만 사용
    df = pd.read_csv(cfg.data_path, usecols=cols, parse_dates=["date"])      # 원본 CSV 로드
    n_rows_raw = len(df)                                                     # 원본 행 수
    df = df[df["EXCHCD"].isin(cfg.exchanges)]                                # 1. NYSE/AMEX/NASDAQ만 유지
    bad = df.loc[df["ADJ_PRC"] > cfg.price_error_threshold, "PERMNO"].unique()  # 2. 비정상 가격 PERMNO
    df = df[~df["PERMNO"].isin(bad)]                                         # 오류 종목 제외

    price = df.pivot(index="date", columns="PERMNO", values="ADJ_PRC").sort_index()  # 월 × 종목 가격
    shape = dict(index=price.index, columns=price.columns)                   # 공통 모양
    volume = df.pivot(index="date", columns="PERMNO", values="VOL").reindex(**shape)      # 거래량
    exchange = df.pivot(index="date", columns="PERMNO", values="EXCHCD").reindex(**shape)  # 거래소 코드

    returns = price / price.shift(1) - 1.0                                   # 3. 연속된 두 달로만 수익률
    nxt = returns.shift(-1)                                                  # 다음 달 수익률
    spike = (returns > cfg.reversal_threshold) & ((1 + returns) * (1 + nxt) - 1 < cfg.reversal_cum)  # 4. 급등 후 반전
    error_mask = spike | spike.shift(1, fill_value=False)                    # 급등 달과 반전 달 모두 표시
    returns = returns.mask(error_mask)                                       # 오류 수익률 결측 처리

    if cfg.nasdaq_volume_adjust:                                             # 5. NASDAQ 거래량 보정
        div = _nasdaq_volume_divisor(volume.index)                           # 날짜별 나눗수
        is_nasdaq = exchange == 3                                            # NASDAQ 위치
        volume = volume.where(~is_nasdaq, volume.div(div, axis=0))           # NASDAQ만 나눔

    info = {"rows_raw": n_rows_raw,                                          # 원본 행 수
            "rows_after_exchange_filter": len(df),                           # 필터 후 행 수
            "bad_price_permno": len(bad),                                    # 가격 오류로 제외한 종목 수
            "spike_reversal_cases": int(spike.values.sum()),                 # 급등-반전 오류 건수
            "returns_masked": int(error_mask.values.sum())}                  # 결측 처리한 수익률 수
    return Panel(returns, price, volume, exchange, error_mask, info)         # 전처리 결과


def select_universe(panel, t_idx, cfg):
    """t 시점 유니버스: 창 전체 수익률이 있는 종목 중 평균 거래량 상위 N개."""
    start = t_idx - cfg.lookback + 1                                  # 창 시작 인덱스
    window = panel.returns.iloc[start : t_idx + 1]                    # 과거 lookback개월 수익률 (t 포함)
    complete = window.notna().all(axis=0)                             # 창 안에 빈 달이 없는 종목
    avg_vol = panel.volume.iloc[start : t_idx + 1].mean(axis=0)       # 창 안 평균 거래량
    avg_vol = avg_vol[complete].dropna()                              # 수익률이 완전한 종목만 남김
    members = avg_vol.nlargest(cfg.universe_size).index               # 거래량 상위 N개 선택
    return window[members]                                            # (lookback × N) 수익률 DataFrame


def realized_next_returns(panel, t_idx, members):
    """t+1월 실현 수익률. 데이터에 있는 그대로 반영하고, 가정 수익률은 넣지 않는다.

    t+1월 가격이 없으면(상장폐지·거래 중단 또는 오류 제거) 데이터가 보여주는 마지막 가격(t월)으로
    평가한다 → 그 달 수익률 0%. 해당 종목은 다음 리밸런싱에서 유니버스 조건(수익률 완비)으로 자동 제외된다.
    데이터에는 상장폐지 수익률(청산·인수 대금)이 없으므로 실제 상장폐지 손익은 반영할 수 없다 (보고서에 명시).
    """
    r = panel.returns.iloc[t_idx + 1].reindex(members)                # t+1월 수익률
    missing = r.isna()                                                # 결측 종목
    err = panel.error_mask.iloc[t_idx + 1].reindex(members, fill_value=False) & missing  # 오류로 지운 종목
    counts = {"next_missing_error": int(err.sum()),                   # 오류 제거로 결측인 종목 수
              "next_missing_no_data": int((missing & ~err).sum())}    # 데이터가 끊긴(상장폐지 등) 종목 수
    return r.fillna(0.0), counts                                      # 마지막 가격 유지 = 0%, 건수 반환


def winsorize_cross_section(window, q):
    """각 월마다 횡단면 분위수로 극단값을 잘라낸다 (추정용)."""
    lo = window.quantile(q, axis=1)                                   # 월별 하위 분위수
    hi = window.quantile(1 - q, axis=1)                               # 월별 상위 분위수
    return window.clip(lower=lo, upper=hi, axis=0)                    # 월별 기준으로 잘라내기


def time_weights(T, halflife):
    """창 안 관측치 가중치 (합 1). 최근 달일수록 크고, halflife=None이면 균등."""
    if halflife is None:                                              # 균등 가중 설정이면
        return np.full(T, 1.0 / T)                                    # 모든 달 1/T
    age = np.arange(T - 1, -1, -1)                                    # 각 관측치의 경과 개월 수 (가장 최근 = 0)
    w = 0.5 ** (age / halflife)                                       # 반감기마다 가중치 절반
    return w / w.sum()                                                # 합 1로 정규화


def effective_n(w):
    """가중치의 유효 관측치 수 (Kish): 1 / Σw²."""
    return 1.0 / np.sum(w ** 2)                                       # 균등 가중이면 T와 같음


def market_residuals(X, w):
    """[C23 식 (2)] 잔차 = r_i − β_i × r_mkt. 시장 수익률은 유니버스 동일가중 평균, β는 가중 회귀."""
    mkt = X.mean(axis=1)                                              # 시장 수익률 프록시 (T,)
    mkt_c = mkt - w @ mkt                                             # 가중 평균 제거
    X_c = X - w @ X                                                   # 종목별 가중 평균 제거
    beta = (w * mkt_c) @ X_c / (w @ mkt_c ** 2)                       # 종목별 가중 시장 베타 (N,)
    return X - np.outer(mkt, beta)                                    # 잔차 수익률 (T, N)
