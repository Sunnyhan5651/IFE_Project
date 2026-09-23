"""
실행 진입점:  python main.py   (백테스트 → 결과 저장 → report.py로 성과표·그래프 생성)

출력 (output/ 폴더)
  monthly_returns_gross.csv  전략별 월별 총수익률 (비용 전)
  monthly_returns_net.csv    대표 거래비용(cost_bps_main) 차감 순수익률
  turnover.csv               전략별 월별 회전율
  diagnostics.csv            월별 유니버스 구성, K 후보, 클러스터 크기, 유효 보유 종목 수, 최대 비중, 결측 건수
  cluster_membership.csv     월별 PERMNO → 클러스터 번호 (첫 반복 기준)
  metrics.csv                비용 수준 × 기간 × 전략(+ S&P 500 TR) 성과지표
  main_summary.csv           대표 전략과 벤치마크 요약
  results.png                순수익률 누적 가치, 낙폭, K 추이

사전 준비: python benchmarks.py  (RF와 S&P 500 총수익지수 → data/benchmarks_monthly.csv)
"""

import time                          # 실행 시간 측정
from config import Config            # 실험 설정
from data import load_panel          # 데이터 로드·전처리
from backtest import run_backtest    # 백테스트
from report import make_report       # 성과표·그래프


def main(cfg=None):
    cfg = cfg or Config()                                             # 기본 설정 사용
    cfg.output_dir.mkdir(parents=True, exist_ok=True)                 # 결과 폴더 생성
    t0 = time.time()                                                  # 시작 시각
    print("[1] 데이터 로드·전처리")                                     # 진행 표시
    panel = load_panel(cfg)                                           # 전처리된 패널
    print(f"    기간 {panel.returns.index[0]:%Y-%m} ~ {panel.returns.index[-1]:%Y-%m}, "
          f"종목 {panel.returns.shape[1]:,}개")                         # 데이터 요약
    for k, v in panel.info.items():                                   # 전처리 요약 통계
        print(f"    {k}: {v:,}")                                      # 항목별 출력
    print("[2] 백테스트")                                              # 진행 표시
    rets, turns, diags, membership = run_backtest(panel, cfg)         # 백테스트 실행
    rets.to_csv(cfg.output_dir / "monthly_returns_gross.csv")         # 월별 총수익률 저장
    turns.to_csv(cfg.output_dir / "turnover.csv")                     # 회전율 저장
    diags.to_csv(cfg.output_dir / "diagnostics.csv")                  # 진단 정보 저장
    membership.to_csv(cfg.output_dir / "cluster_membership.csv", index=False)  # 월별 클러스터 구성 저장
    print("[3] 성과지표·그래프")                                        # 진행 표시
    make_report(cfg)                                                  # 저장된 결과로 보고서 생성
    print(f"완료 ({time.time() - t0:.0f}초) → {cfg.output_dir}")        # 종료 메시지


if __name__ == "__main__":                                            # 스크립트로 실행할 때만
    main()                                                            # 기본 설정으로 실행
