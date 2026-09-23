"""
성과지표와 대표 전략 선택.

  필수: 연환산 수익률, 연환산 표준편차, Sharpe(Ken French RF 초과수익 기준), MDD
  추가: Calmar, Sortino(RF 초과), S&P 500 대비 베타, 월평균 회전율

백테스트는 비용 전(총) 수익률과 회전율을 저장하고, 여기서 비용 수준별로
순수익률 = 총수익률 − (bp / 10,000) × 회전율 을 계산한다 (다시 실행할 필요 없음).
"""

import numpy as np   # 수치 계산
import pandas as pd  # 결과 표

SP500_NAME = "S&P 500 TR"  # 벤치마크 열 이름


def net_returns(gross, turns, bps):
    """비용 수준 bp를 반영한 월별 순수익률."""
    return gross - bps / 1e4 * turns.reindex_like(gross)              # 거래금액 × 비용률 차감


def performance(r, turnover, rf, mkt):
    """월별 수익률 r의 성과지표 dict. rf, mkt는 같은 날짜의 무위험 수익률과 S&P 500 수익률."""
    r = r.dropna()                                                    # 결측 제거
    rf = rf.reindex(r.index)                                          # 무위험 수익률 날짜 맞추기
    ex = r - rf                                                       # 초과수익률
    wealth = (1.0 + r).cumprod()                                      # 누적 가치 ($1 투자)
    years = len(r) / 12.0                                             # 기간(년)
    ann_ret = wealth.iloc[-1] ** (1.0 / years) - 1.0                  # 기하 연환산 수익률
    mdd = (wealth / wealth.cummax() - 1.0).min()                      # 최대 낙폭 (음수)
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())             # 초과수익 기준 하방 편차 (월)
    mkt_ex = mkt.reindex(r.index) - rf                                # S&P 500 초과수익률
    beta = ex.cov(mkt_ex) / mkt_ex.var()                              # 시장 베타
    return {
        "Annualized Return": ann_ret,                                 # 연환산 수익률 (필수)
        "Annualized Std": r.std(ddof=1) * np.sqrt(12.0),              # 연환산 표준편차 (필수)
        "Sharpe": ex.mean() / ex.std(ddof=1) * np.sqrt(12.0),         # 연환산 샤프 비율, RF 초과 (필수)
        "Max Drawdown": mdd,                                          # 최대 낙폭 (필수)
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,          # 연수익률 ÷ |MDD| (추가)
        "Sortino": ex.mean() / downside * np.sqrt(12.0) if downside > 0 else np.nan,  # 하방 위험 대비 초과수익 (추가)
        "Beta vs S&P 500": beta,                                      # 시장 민감도 (추가)
        "Avg Monthly Turnover": turnover.reindex(r.index).iloc[1:].mean(),  # 첫 달(초기 매수) 제외 평균 회전율
        "Months": len(r),                                             # 관측 개월 수
    }


def periods_of(index, split_dates):
    """(이름, 날짜) 목록: 전체 기간과 split_dates로 나눈 하위 구간."""
    out = [("Full", index)]                                           # 전체 기간
    edges = [None, *pd.to_datetime(list(split_dates)), None]          # 하위 구간 경계
    if split_dates:                                                   # 하위 구간이 있으면
        for start, end in zip(edges[:-1], edges[1:]):                 # 구간별로
            idx = index                                               # 전체 날짜
            if start is not None:                                     # 시작 경계
                idx = idx[idx > start]                                # 경계 이후만
            if end is not None:                                       # 끝 경계
                idx = idx[idx <= end]                                 # 경계 이전만
            out.append((f"{idx[0]:%Y-%m}~{idx[-1]:%Y-%m}", idx))      # 구간 이름과 날짜
    return out                                                        # 구간 목록


def performance_table(gross, turns, cost_bps_list, split_dates, bench):
    """비용 수준 × 기간 × 전략(+ S&P 500 TR) 성과표. bench는 RF, SP500TR 열을 가진 DataFrame."""
    rows = []                                                         # 결과 행
    for bps in cost_bps_list:                                         # 비용 수준별로
        net = net_returns(gross, turns, bps)                          # 순수익률
        net[SP500_NAME] = bench["SP500TR"]                            # 벤치마크 추가 (거래비용 없음)
        tn = turns.copy()                                             # 회전율 복사
        tn[SP500_NAME] = 0.0                                          # 지수 보유는 회전율 0
        for label, idx in periods_of(gross.index, split_dates):       # 기간별로
            for col in net.columns:                                   # 전략별로
                m = performance(net.loc[idx, col], tn[col], bench["RF"], bench["SP500TR"])  # 지표 계산
                rows.append({"Cost (bp)": bps, "Period": label, "Strategy": col, **m})  # 행 추가
    return pd.DataFrame(rows)                                         # 긴 형태 성과표


def select_main_strategy(table, cfg):
    """사전에 정한 규칙으로 대표 클러스터 전략을 고른다.

    규칙: 선택 구간(첫 하위 구간) · 대표 비용(cost_bps_main)에서 클러스터 전략 중 Sharpe 최대.
    동률이면 MDD가 작은 쪽. 나머지 구간(검증 구간) 성과는 선택에 쓰지 않고 보고만 한다.
    """
    labels = [p for p in table["Period"].unique() if p != "Full"]     # 하위 구간 이름
    sel_period = labels[0]                                            # 첫 구간 = 선택 구간
    t = table[(table["Period"] == sel_period) & (table["Cost (bp)"] == cfg.cost_bps_main)]  # 선택 구간·대표 비용
    t = t[t["Strategy"].str.startswith("Cluster-")]                   # 클러스터 전략만
    best = t.sort_values(["Sharpe", "Max Drawdown"], ascending=[False, False]).iloc[0]  # Sharpe 최대, 동률 시 낙폭 작은 쪽
    return best["Strategy"], sel_period                               # 대표 전략 이름과 선택 구간
