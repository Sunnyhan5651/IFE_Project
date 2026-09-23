"""
월별 rolling walk-forward 백테스트 [K24 4.1-iv, 월별 조정].

t월 말: 과거 lookback개월(t 포함)로 비중 계산 → t+1월 실현 수익률 기록 → 한 달 이동.
체결 가정: t월 말 가격으로 비중을 계산하고 같은 월말 가격에 리밸런싱한다.
"""

import numpy as np                     # 수치 계산
import pandas as pd                    # 결과 표
from data import select_universe, realized_next_returns  # 유니버스 선정, 실현 수익률
from optimization import ledoit_wolf, gmvp, msrp, project_simplex  # 벤치마크용 최적화
from portfolio import build_portfolios      # 클러스터 포트폴리오 구성


def _turnover(w_new, w_prev, r_prev):
    """직전 달 수익률로 비중이 변한 뒤, 새 비중으로 맞추는 데 필요한 거래량 Σ|Δw|."""
    if w_prev is None:                                                # 첫 달이면
        return 1.0                                                    # 전액 매수
    grown = w_prev * (1.0 + r_prev.reindex(w_prev.index).fillna(0.0))  # 한 달 동안 변한 보유 금액
    drifted = grown / grown.sum()                                     # 리밸런싱 직전 비중
    idx = w_new.index.union(drifted.index)                            # 신구 종목 합집합
    return float((w_new.reindex(idx, fill_value=0.0) - drifted.reindex(idx, fill_value=0.0)).abs().sum())  # 거래량


def run_backtest(panel, cfg):
    """전체 기간 백테스트를 실행하고 월별 총수익률(비용 전), 회전율, 진단 정보를 반환한다."""
    rng = np.random.default_rng(cfg.seed)                             # 재현 가능한 난수 생성기
    prev_w = {}                                                       # 전략별 직전 달 비중 (첫 달은 없음)
    prev_r = None                                                     # 직전 달 실현 수익률 (가정값 반영)
    rows, turn_rows, diag_rows, member_rows = [], [], [], []          # 결과 누적 리스트
    dates = panel.returns.index                                       # 월말 날짜 목록

    for t_idx in range(cfg.lookback, len(dates) - 1):                 # 첫 완전한 창부터 마지막 전 달까지
        window = select_universe(panel, t_idx, cfg)                   # t 시점 유니버스와 추정 창
        weights, X, tw, diag, labels = build_portfolios(window, cfg, rng)  # 비중, 시간 가중치, 진단, 클러스터 라벨
        members = window.columns                                      # 유니버스 종목 PERMNO
        member_rows.append(pd.DataFrame({"date": dates[t_idx], "PERMNO": members, "cluster": labels}))  # 클러스터 구성 기록
        port = {f"Cluster-{m}": pd.Series(w, index=members) for m, w in weights.items()}  # 이름 붙인 비중
        if cfg.run_benchmarks:                                        # 벤치마크 계산 (같은 유니버스, 같은 시간 가중, 같은 상한)
            port["EW-Universe"] = pd.Series(1.0 / len(members), index=members)  # 동일가중
            Sigma_lw = ledoit_wolf(X, tw)                              # 전체 종목 수축 공분산
            if "gmvp" in cfg.objectives:                              # 최소분산 벤치마크
                port["NoCluster-LW-gmvp"] = pd.Series(gmvp(Sigma_lw, cfg.max_weight), index=members)  # 상한 포함 최적화
            if "msrp" in cfg.objectives:                              # 최대 샤프 벤치마크
                w_ms, _ = msrp(tw @ X, Sigma_lw)                       # 전체 종목 MSRP
                if cfg.max_weight is not None:                        # 상한이 있으면
                    w_ms = project_simplex(w_ms, cfg.max_weight)      # 클러스터 전략과 같은 방식으로 상한 투영
                port["NoCluster-LW-msrp"] = pd.Series(w_ms, index=members)  # 이름 붙여 저장

        nxt, miss = realized_next_returns(panel, t_idx, members)      # t+1월 실현 수익률 (데이터 그대로, 결측은 마지막 가격)

        row, trow = {"date": dates[t_idx + 1]}, {"date": dates[t_idx + 1]}  # 결과 행 초기화
        for name, w in port.items():                                  # 전략별로
            trow[name] = _turnover(w, prev_w.get(name), prev_r)       # 회전율 (비용은 성과표에서 수준별로 차감)
            row[name] = float((w * nxt).sum())                        # 비용 전(총) 수익률
            prev_w[name] = w                                          # 다음 달을 위해 저장
        prev_r = nxt                                                  # 다음 회전율 계산용 실현 수익률

        eff_n = {f"effN_{m}": float(1.0 / (w ** 2).sum()) for m, w in weights.items()}  # 유효 보유 종목 수
        eff_n.update({f"maxW_{m}": float(w.max()) for m, w in weights.items()})  # 종목 최대 비중
        exch = panel.exchange.iloc[t_idx].reindex(members)            # 유니버스 종목의 거래소 코드
        mix = {"share_nyse": float((exch == 1).mean()),               # NYSE 비중
               "share_amex": float((exch == 2).mean()),               # AMEX 비중
               "share_nasdaq": float((exch == 3).mean())}             # NASDAQ 비중
        diag_rows.append({"date": dates[t_idx], **miss, **mix, **diag, **eff_n})  # 진단 기록
        rows.append(row)                                              # 수익률 기록
        turn_rows.append(trow)                                        # 회전율 기록
        print(f"\r  {dates[t_idx]:%Y-%m}  N={diag['N']}  K={diag['K']}  "
              f"size[{diag['cluster_size_min']},{diag['cluster_size_max']}]   ", end="")  # 진행 상황 출력

    print()                                                           # 줄바꿈
    rets = pd.DataFrame(rows).set_index("date")                       # 월별 총수익률 표
    turns = pd.DataFrame(turn_rows).set_index("date")                 # 월별 회전율 표
    diags = pd.DataFrame(diag_rows).set_index("date")                 # 월별 진단 표
    membership = pd.concat(member_rows, ignore_index=True)            # 월별 PERMNO → 클러스터 표
    return rets, turns, diags, membership                             # 반환
