"""
실험 설정 모음.

모든 하이퍼파라미터를 한 곳에 모아 두고, 실험마다 Config 객체만 바꿔서 실행한다.
논문 원안(일별 데이터)을 월별 데이터에 맞게 바꾼 부분은 주석에 [월별 조정]으로 표시한다.

참고 논문
  [K24] Khelifa, Allier & Cucuringu (ICAIF 2024)
  [C23] Cartea, Cucuringu & Jin (ICAIF 2023)
"""

from dataclasses import dataclass, field  # 설정 클래스를 간단히 정의하기 위한 도구
from pathlib import Path                   # 운영체제와 무관한 경로 처리

PROJECT_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 최상위 폴더 (code/의 상위)


@dataclass
class Config:
    # ── 데이터 ────────────────────────────────────────────────
    data_path: Path = PROJECT_DIR / "data" / "IFE_term_stock_data.csv"  # 과제 제공 CRSP 월별 데이터
    output_dir: Path = PROJECT_DIR / "output"                             # 결과 저장 폴더
    exchanges: tuple = (1, 2, 3)            # 과제 지침: NYSE(1), AMEX(2), NASDAQ(3)만 사용
    price_error_threshold: float = 1e5      # 이 값을 넘는 ADJ_PRC가 있는 PERMNO는 데이터 오류로 제외
    reversal_threshold: float = 3.0         # 급등-반전 오류 판정: 월 수익률 > 300% 이고
    reversal_cum: float = 0.5               #   다음 달까지 2개월 누적 수익률 < 50%면 두 달 모두 오류 (Ince & Porter 2006)
    nasdaq_volume_adjust: bool = True       # NASDAQ 거래량 이중집계 보정 (Gao & Ritter 2010)
    # 최소 주가 필터는 쓰지 않는다: ADJ_PRC는 이후 분할·배당으로 소급 조정되어 미래 정보가 섞임

    # ── 유니버스 ──────────────────────────────────────────────
    lookback: int = 60                      # 추정 창 길이(개월) [월별 조정: K24는 75거래일]
    halflife: float | None = 18.0           # 창 안 관측치의 지수가중 반감기(개월), None이면 균등 가중 [K24 4.6]
                                            # 60개월·반감기 18 → 유효 관측치 약 43, 최근 1개월 가중치 약 4.2%
    universe_size: int = 500                # 창 내 평균 VOL 상위 N개 종목 (VOL은 활동성 지표로만 사용)
    winsor_q: float = 0.01                  # 추정용 수익률의 횡단면 윈저라이징 분위 (실현 수익률에는 미적용)

    # ── 클러스터링 ────────────────────────────────────────────
    use_residual: bool = True               # True면 시장 요인 제거 잔차로 상관행렬 계산 [C23 2.2.1]
    cluster_algo: str = "sponge_sym"        # "sponge_sym" [K24 3.4] 또는 "signed_laplacian" [K24 3.2]
    tau_pos: float = 1.0                    # SPONGE 정규화 계수 τ+ (SigNet 기본값)
    tau_neg: float = 1.0                    # SPONGE 정규화 계수 τ− (SigNet 기본값)
    k_method: str = "mp"                    # "mp"(Marchenko–Pastur) | "var"(누적 분산 비율) | "fixed" [C23 2.3]
                                            # MP의 T에는 유효 관측치 수를 사용 (지수가중이면 근사)
    k_fixed: int = 10                       # 고정 K (k_method="fixed"일 때만 사용)
    k_var_threshold: float = 0.9            # "var" 방식의 누적 분산 기준 P [C23 표 1: 90%]
    k_bounds: tuple = (3, 20)               # 동적 K의 하한/상한 (반감기 18 기준 MP의 K는 약 3~8)
    n_repeats: int = 10                     # 클러스터링 반복 횟수 R, 최종 종목 비중을 평균 [K24 표 2: R = 10]
    kmeans_n_init: int = 3                  # 반복 1회당 k-means++ 초기화 횟수 (관성 최소 결과 채택)

    # ── 클러스터 안 비중 β ────────────────────────────────────
    intra_methods: tuple = ("gaussian", "ivp", "periphery")  # 비교할 β 방식 (②③⑥)
    gauss_sigma_scale: float = 1.0          # σ² = scale × 클러스터 안 평균 거리² [월별 조정: K24는 교차검증]

    # ── 클러스터 간 최적화 α ──────────────────────────────────
    objectives: tuple = ("gmvp", "msrp")    # 클러스터 간 최적화 방식 (롱온리) [월별 조정: K24는 목표수익률 제약, 롱숏]
    max_weight: float | None = 0.05         # 종목별 비중 상한 5% (쏠림 방지, 모든 전략에 동일 적용)

    # ── 백테스트 ──────────────────────────────────────────────
    # 상장폐지: 가정 수익률 없이 데이터의 마지막 가격으로 평가 (그 달 0%) — data.realized_next_returns 참고
    cost_bps_list: tuple = (0.0, 1.0, 10.0) # 성과표에 함께 보고할 편도 거래비용(bp): 순수익률 = 총수익률 − bp × 회전율
    cost_bps_main: float = 10.0             # 그래프에 쓸 대표 거래비용(bp)
    run_benchmarks: bool = True             # 동일가중, 전체 종목 GMVP(Ledoit–Wolf) 벤치마크 계산 여부
    seed: int = 42                          # 난수 시드 (k-means 재현성)
    split_dates: tuple = field(default_factory=lambda: ("2012-12-31",))  # 하위 구간 성과 분리 기준일
