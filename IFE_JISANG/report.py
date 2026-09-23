"""
저장된 백테스트 결과(output/)로 성과표와 그래프를 만든다. 백테스트를 다시 돌리지 않는다.

실행:  python report.py
입력:  output/monthly_returns_gross.csv, output/turnover.csv, output/diagnostics.csv, data/benchmarks_monthly.csv
출력:  output/metrics.csv, output/main_summary.csv, output/results.png
"""

import matplotlib                    # 그래프
matplotlib.use("Agg")                # 창 없이 파일로만 저장
import matplotlib.pyplot as plt      # 그래프 그리기
import pandas as pd                  # 표 형태 데이터 처리
from config import Config            # 실험 설정
from benchmarks import load_benchmarks  # RF, S&P 500 총수익지수
from metrics import performance_table, net_returns, select_main_strategy, SP500_NAME  # 성과지표


def plot_results(net, bench, diags, main_name, path):
    """대표 비용 기준 누적 가치(로그), 낙폭, K 추이."""
    curves = net.copy()                                               # 전략별 순수익률
    curves[SP500_NAME] = bench["SP500TR"]                             # S&P 500 총수익지수 추가
    wealth = (1.0 + curves).cumprod()                                 # 누적 가치
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))                  # 3단 그래프
    for col in wealth:                                                # 전략별로
        lw = 2.4 if col in (main_name, SP500_NAME) else 1.0           # 대표 전략과 S&P 500은 굵게
        axes[0].plot(wealth.index, wealth[col], lw=lw, label=col)     # 1) 누적 가치
        axes[1].plot(wealth.index, wealth[col] / wealth[col].cummax() - 1.0, lw=lw, label=col)  # 2) 낙폭
    axes[0].set_yscale("log")                                         # 로그 축
    axes[0].set_title("Portfolio value of $1 (net of costs, log scale)")  # 제목
    axes[0].legend(fontsize=8, ncol=2)                                # 범례
    axes[1].set_title("Drawdown")                                     # 제목
    axes[2].plot(diags.index, diags["K"], lw=1.4, label="K used (Marchenko–Pastur, bounded)")  # 3) 사용한 K
    axes[2].set_title("Number of clusters per month")                 # 제목
    axes[2].legend(fontsize=8)                                        # 범례
    for ax in axes:                                                   # 모든 축에
        ax.grid(alpha=0.3)                                            # 격자 표시
    fig.tight_layout()                                                # 여백 정리
    fig.savefig(path, dpi=130)                                        # 파일 저장
    plt.close(fig)                                                    # 메모리 해제


def make_report(cfg=None):
    cfg = cfg or Config()                                             # 기본 설정
    out = cfg.output_dir                                              # 결과 폴더
    gross = pd.read_csv(out / "monthly_returns_gross.csv", index_col="date", parse_dates=True)  # 총수익률
    turns = pd.read_csv(out / "turnover.csv", index_col="date", parse_dates=True)  # 회전율
    diags = pd.read_csv(out / "diagnostics.csv", index_col="date", parse_dates=True)  # 진단 정보
    bench = load_benchmarks(gross.index)                              # 같은 날짜의 RF, S&P 500
    table = performance_table(gross, turns, cfg.cost_bps_list, cfg.split_dates, bench)  # 전체 성과표
    table.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")  # 저장
    main_name, sel_period = select_main_strategy(table, cfg)          # 사전 규칙으로 대표 전략 선택
    summary = table[table["Strategy"].isin([main_name, SP500_NAME, "EW-Universe", "NoCluster-LW-gmvp", "NoCluster-LW-msrp"])]  # 요약 대상
    summary.to_csv(out / "main_summary.csv", index=False, encoding="utf-8-sig")  # 요약 저장
    net = net_returns(gross, turns, cfg.cost_bps_main)                # 대표 비용 순수익률
    net.to_csv(out / "monthly_returns_net.csv")                       # 저장
    plot_results(net, bench, diags, main_name, out / "results.png")   # 그래프 저장
    view = table[(table["Cost (bp)"] == cfg.cost_bps_main)]           # 대표 비용 기준
    cols = ["Strategy", "Annualized Return", "Annualized Std", "Sharpe", "Max Drawdown", "Calmar", "Beta vs S&P 500", "Avg Monthly Turnover"]  # 출력 열
    for period in view["Period"].unique():                            # 기간별로 출력
        print(f"\n=== {period} | cost {cfg.cost_bps_main:g}bp ===")      # 구간 제목
        print(view[view["Period"] == period][cols].round(3).to_string(index=False))  # 표 출력
    print(f"\n대표 전략 (선택 구간 {sel_period}, Sharpe 기준): {main_name}")  # 선택 결과
    return table, main_name                                           # 반환


if __name__ == "__main__":                                            # 스크립트로 실행할 때만
    make_report()                                                     # 보고서 생성
