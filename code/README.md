# Cluster-driven Hierarchical Portfolio — 프레임워크

> 모든 설정값과 근거, 대표 전략 선택 규칙, 민감도 계획, 한계는 [SETTINGS.md](SETTINGS.md)에 정리되어 있다.

과제 제공 데이터(`data/IFE_term_stock_data.csv`)만으로 동작하는 월별 리밸런싱 백테스트.

## 참고 논문의 파이프라인

### [K24] Khelifa, Allier & Cucuringu (ICAIF 2024)
1. **그래프 구성**: 수익률 Pearson 상관행렬 Γ를 부호 그래프 인접행렬로 해석 (§2)
2. **클러스터링**: Spectral / Signed Laplacian / SPONGE / SPONGE_sym → k-means++ (§3)
3. **클러스터 안 비중 β**: Gaussian kernel, 중심(평균 수익률 벡터)에 가까울수록 큼 (§4.2)
4. **가상 자산**: r*_k = Σ β_i r_i, μ*와 Σ*(K×K) 계산 (§4.1-ii)
5. **클러스터 간 비중 α**: K차원 Markowitz (목표수익률 하 분산 최소) (§1.1, 식 1)
6. **종목 비중**: w_i = α_k β_i (§4.1-ii)
7. **안정화**: 1~6을 R번 반복해 비중 평균 (§4.1-iii)
8. **평가**: 보유 후 한 창 이동, 반복 (§4.1-iv), EWA 공분산 (§4.6), K = 10/17/24 비교 (§5.3)

### [C23] Cartea, Cucuringu & Jin (ICAIF 2023)
1. **시장 잔차**: R_res = R − β R_mkt (§2.2.1)
2. **잔차 상관행렬** → 부호 그래프 클러스터링 (§2.2.2)
3. **K 결정** (§2.3)
   - Marchenko–Pastur: λ₊ = (1 + √(N/T))²를 넘는 고유값 개수
   - 누적 분산: 상위 고유값 누적 비율이 P(90%)에 도달하는 개수

## 전처리 (백테스트 전 1회, data.py)
| 단계 | 규칙 | 결과 |
|---|---|---|
| 거래소 | EXCHCD ∈ {1, 2, 3} | 1,207,153 → 1,200,831행 |
| 가격 오류 | ADJ_PRC > 1e5인 PERMNO 제외 (소급 조정 가격은 낮아지기만 하므로 큰 값은 오류) | 34개 PERMNO |
| 수익률 | 연속된 두 달 가격으로만 계산 | 빈 달 뒤 수익률은 결측 |
| 급등-반전 오류 | r_t > 300% 이고 (1+r_t)(1+r_{t+1}) − 1 < 50%면 두 달 결측 (Ince & Porter 2006) | 2건, 4개 수익률 |
| NASDAQ 거래량 | 2001-01 이전 ÷2.0, 2001-02~12 ÷1.8, 2002~03 ÷1.6 (Gao & Ritter 2010) | 거래량 상위 500의 NASDAQ 비중 약 32% |
| 최소 주가 | **적용 안 함**: ADJ_PRC는 분할·배당으로 소급 조정(예: AAPL 2010-12 = 9.97)되어 미래 정보가 섞임 | – |

**다음 달 수익률 결측 처리** (`realized_next_returns`)
- 가정 수익률을 넣지 않고 데이터 그대로 반영: 마지막 관측 가격으로 평가 → 그 달 0%, 다음 리밸런싱에서 자동 제외
- 데이터에 상장폐지 수익률(청산·인수 대금)이 없어 실제 상장폐지 손익은 반영 불가 → 보고서에 한계로 명시
- 2005~2020 유니버스에서 약 390건 (월평균 약 2건, 보유 종목의 약 0.4%)

**수익률 성격**: ADJ_PRC가 배당까지 반영하므로 수익률은 총수익률에 가깝다 → S&P 500도 **총수익(배당 포함) 지수**와 비교.

## 이 프레임워크의 월별 파이프라인
```
매월 t (t까지의 데이터만 사용)
① 유니버스   EXCHCD 1/2/3, 과거 lookback(60)개월 수익률 완비, 평균 VOL 상위 N(500)   data.py
② 전처리     횡단면 1%/99% 윈저라이징(추정용), 시장 잔차(유니버스 EW 평균 기준) [C23]  data.py
③ K 결정     매월 동적 결정: Marchenko–Pastur(기본) 또는 누적 분산 — 모든 후보를 진단으로 기록 [C23]  clustering.py
④ 클러스터링 SPONGE_sym 또는 Signed Laplacian 임베딩 → k-means++ [K24]               clustering.py
⑤ β          gaussian ② / ivp ③ / periphery ⑥  (같은 라벨을 공유해 β 효과만 비교)     allocation.py
⑥ α          가상 ETF 수익률 → Σ*(시간 가중) → 롱온리 GMVP 또는 MSRP                  optimization.py

※ 창 안의 모든 추정(시장 베타, 상관행렬, K, β, Σ*)은 반감기 18개월 지수가중을 사용한다 [K24 §4.6].
   60개월 창 → 유효 관측치 약 43개, 최근 1개월 가중치 약 4.2%, 최근 12개월 합 약 41%.
   Marchenko–Pastur의 T에는 유효 관측치 수를 넣는다 (균등 가중을 가정한 이론의 근사).
⑦ 종목 비중  w = α × β, R회 평균 [K24], (선택) 종목 상한 투영                         portfolio.py
⑧ 보유       t+1월 실현 수익률(원시), 회전율·비용 반영                                 backtest.py
```

## 단계별 출처
| 단계 | 출처 | 구현 위치 |
|---|---|---|
| 상관행렬을 부호 그래프로 해석 | [K24] §2 | `clustering.correlation` |
| SPONGE_sym 클러스터링 | [K24] §3.4 (원출처: Cucuringu et al. 2019) | `clustering.embedding_sponge_sym` |
| Signed Laplacian (대안) | [K24] §3.2 (원출처: Kunegis et al. 2010) | `clustering.embedding_signed_laplacian` |
| k-means++ | [K24] §3.1 (원출처: Arthur & Vassilvitskii 2007) | `clustering.kmeans_pp` |
| Gaussian kernel 클러스터 안 비중 β | [K24] §4.2 | `allocation.beta_gaussian` |
| 가상 자산 수익률, w = α × β | [K24] §4.1 | `portfolio.build_portfolios` |
| 지수가중(EWA) 공분산 | [K24] §4.6 | `data.time_weights`, `optimization.covariance` |
| R회 반복 후 비중 평균 | [K24] §4.1-iii, 표 2 | `portfolio.build_portfolios` |
| 워크포워드 평가 | [K24] §4.1-iv, 그림 2 | `backtest.run_backtest` |
| 회전율·거래비용 | [K24] §4.5 | `backtest._turnover`, `metrics.net_returns` |
| 시장 잔차 수익률 | [C23] §2.2.1 식 (2) | `data.market_residuals` |
| K 결정: Marchenko–Pastur | [C23] §2.3 식 (5) | `clustering.choose_k` |
| K 결정: 누적 분산 비율(대안) | [C23] §2.3 식 (6) | `clustering.choose_k` |
| 역분산 β | López de Prado (2016) HRP의 클러스터 안 배분 | `allocation.beta_ivp` |
| 주변부 가중 β | Pozzi, Di Matteo & Aste (2013) | `allocation.beta_periphery` |
| Ledoit–Wolf 수축 (벤치마크) | Ledoit & Wolf (2004) | `optimization.ledoit_wolf` |
| 평균-분산 최적화 (GMVP/MSRP) | Markowitz (1952) | `optimization.gmvp`, `optimization.msrp` |
| 급등-반전 오류 필터 | Ince & Porter (2006) | `data.load_panel` |
| NASDAQ 거래량 보정 | Gao & Ritter (2010) | `data._nasdaq_volume_divisor` |
| 무위험 수익률 | Ken French 데이터 라이브러리 | `benchmarks.load_ff_rf` |
| Sortino 비율 | Sortino & Price (1994) | `metrics.performance` |

**논문 근거 없이 팀이 정한 부분**: 유니버스(거래량 상위 500), 최소 주가 필터 미사용, 상장폐지 처리, 반감기 18개월, K 범위 3~20, 종목 상한 5%, 거래비용 수준, 대표 전략 선택 규칙 → 자세한 근거는 [SETTINGS.md](SETTINGS.md).

## 논문 대비 조정 사항
| 항목 | 논문 | 이 프레임워크 | 이유 |
|---|---|---|---|
| 데이터 주기 | 일별 | 월별 | 과제 데이터 |
| 추정 창 | 75거래일 | 60개월, 반감기 18개월 지수가중 | 관측치 확보 + 최근 구조 변화 반영 (반감기는 K와 시장 변동성의 연동성으로 선택, 수익률 성과로 고르지 않음) |
| 기대수익 μ* | 미래 수익률 + 노이즈(η) [K24] | 과거 평균 (MSRP에서만 사용) | 미래 정보 누출 방지 |
| 포지션 | 롱숏 [K24] | 롱온리 | 실행 가능성 |
| Gaussian σ² | 교차검증 [K24] | scale × 클러스터 안 평균 거리² | 팀 설정값 |
| 시장 수익률 | SPY [C23] | 유니버스 동일가중 평균 | 주어진 데이터만 사용 |
| 유니버스 | 시가총액 상위 [C23] | 평균 VOL 상위 | 시가총액 정보 없음 |
| 오류 가격·수익률 | – | 위 전처리 표 참고 | 데이터 오류 제거 (전체 기간 기준 판단, 보고서에 명시) |
| 상장폐지 | – | 마지막 관측 가격으로 평가 (0%) | 데이터에 상장폐지 수익률 없음, 가정 수익률 미사용 |
| 거래비용 | 1bp [K24 §4.5] | 10bp × 회전율 | 월별 주식 리밸런싱의 보수적 가정 |

## 실행
```
cd code
python benchmarks.py   # 최초 1회: RF(Ken French)와 S&P 500 총수익지수 준비
python main.py         # 백테스트 + 성과표·그래프
python report.py       # (선택) 저장된 결과로 성과표·그래프만 다시 생성
```
설정은 `config.py`의 `Config`에서 변경한다. 노트북에서는:
```python
from config import Config
from main import main
main(Config(halflife=12, output_dir=Config().output_dir.parent / "output_hl12"))
```

## 외부 데이터 (평가용, 전략 계산에는 미사용)
- `data/F-F_Research_Data_Factors_CSV.zip`: Ken French 월별 RF
- `data/benchmarks_monthly.csv`: RF, S&P 500 총수익지수(^SP500TR), SPY 월 수익률 (`benchmarks.py`로 생성)
- 연간 수익률을 12등분한 S&P 500 근사치는 변동성과 MDD를 왜곡하므로 사용하지 않는다.
