# Cluster-driven Hierarchical Portfolio — 전략 설명

## 참고 논문

| 약칭 | 논문 |
|------|------|
| [K24] | Khelifa, Allier & Cucuringu (ICAIF '24) — *Cluster-driven Hierarchical Representation of Large Asset Universes for Optimal Portfolio Construction* |
| [C23] | Cartea, Cucuringu & Jin (ICAIF '23) — *Correlation Matrix Clustering for Statistical Arbitrage Portfolios* |
| [T08] | Tola, Lillo, Gallegati & Mantegna (J. Econ. Dyn. Control, 2008) — *Cluster Analysis for Portfolio Optimization* |

---

## 데이터

- **출처**: CRSP 월간 미국 주식 데이터 (NYSE / NASDAQ)
- **기간**: 2000–2020 (워밍업 36개월 제외 시 실질 백테스트 2003–2020)
- **리밸런싱**: 매월

---

## Asset Universe 필터링 (매월 적용)

매월 리밸런싱 시점마다 아래 세 가지 조건을 모두 충족하는 종목만 후보군에 포함합니다.

| 조건 | 기준 |
|------|------|
| 최소 관측 기간 | 36개월 이상 데이터 존재 |
| 최소 주가 | $5 이상 |
| 거래량 | 전체 분포 하위 20% 제외 |

---

## 포트폴리오 구성 로직

### Step 1. CAPM 잔차 수익률 계산 [C23]

클러스터링 전에 시장 공통 인자를 제거합니다.

```
β_i  = cov(r_i, r_mkt) / var(r_mkt)
ε_i  = r_i − β_i × r_mkt
```

원시 수익률 대신 잔차 수익률 ε_i 로 클러스터링하면,
시장 베타 차이가 아닌 **순수 섹터·산업 단위 공동 움직임**으로 종목을 그룹화합니다.

---

### Step 2. 동적 K 결정 [C23]

매월 잔차 수익률 상관행렬의 고유값 누적 비율이 **70% 이상**인 최소 K를 클러스터 수로 선택합니다.

```
K = min { k : Σ_{j=1}^{k} λ_j / Σ λ_j ≥ 0.70 }
K ∈ [5, 20]  (안전 범위)
```

시장 상황에 따라 적절한 클러스터 수를 자동으로 조정합니다.

---

### Step 3. Signed Spectral Clustering [K24]

잔차 수익률 상관행렬 Γ 를 부호 그래프 인접행렬로 해석해 Signed Spectral Clustering을 수행합니다.

```
L_N = I − D^{-½} Γ D^{-½}     (정규화 Laplacian)
```

L_N 의 최소 K개 고유벡터를 추출 → 행 정규화 → **K-means++** 으로 K개 클러스터 결정.

- 양의 상관 종목 → 같은 클러스터
- 음의 상관 종목 → 다른 클러스터

K-means 초기값 민감도를 줄이기 위해 **R=5회 반복 후 평균 비중**을 사용합니다.

---

### Step 4. 클러스터 내 종목 가중치 β (Gaussian 가중치) [K24]

클러스터 k 안에서 중심 μ_k 에 가까운 종목일수록 높은 가중치를 부여합니다.

```
β_{i,k} ∝ exp( −‖r_i − μ_k‖² / 2σ² )
```

σ² 는 클러스터 내 평균 거리에 비례해 적응적으로 결정됩니다.
아웃라이어 종목은 자동으로 낮은 가중치를 받습니다.

---

### Step 5. 합성자산 수익률 생성 [K24]

각 클러스터를 하나의 가상 자산으로 압축합니다.

```
r*_k = Σ_i β_{i,k} × r_i
```

수백 개 종목 → K개 합성자산으로 차원 축소.

---

### Step 6. EWA 공분산 추정 [K24]

K개 합성자산에 대해 지수가중 공분산(EWA)을 계산합니다.

```
Σ*(K×K),  감쇠율 β = 0.94  (반감기 ≈ 12개월)
```

최근 데이터에 더 높은 가중치를 부여해 시장 상황 변화에 빠르게 반응합니다.

---

### Step 7. 클러스터 레벨 GMVP (비중 α)

K×K 공분산으로 **최소분산 포트폴리오(GMVP)**를 풀어 클러스터 간 배분 비중을 결정합니다.

```
α = argmin  α^T Σ* α
     s.t.   Σα_k = 1,  α_k ≥ 0
```

변동성이 낮고 클러스터 간 상관이 낮을수록 높은 비중을 받습니다.
최적화는 Projected Gradient Descent (scipy 미사용, NumPy 구현)로 수행합니다.

---

### Step 8. 최종 종목 비중 결정 및 상한 적용

```
w_i = α_k × β_{i,k}        (클러스터 배분 × 클러스터 내 배분)
w   = w / Σw                (정규화)
w_i = min(w_i, 0.05)        (종목별 최대 5% 상한)
w   = w / Σw                (재정규화)
```

---

### 전체 흐름 요약

```
전체 CRSP 종목
    ↓  가격·거래량·관측기간 필터
Asset Universe (매월 갱신)
    ↓  CAPM 잔차 수익률 계산  [C23]
    ↓  동적 K 결정 (PCA 고유값)  [C23]
    ↓  Signed Spectral Clustering  [K24]
K개 클러스터
    ↓  Gaussian β 가중치  [K24]
    ↓  합성자산 수익률 생성  [K24]
    ↓  EWA 공분산 추정  [K24]
    ↓  K×K GMVP (클러스터 배분 α)  [K24]
최종 종목 비중  w_i = α_k × β_{i,k}
    ↓  종목별 5% 상한 적용
    ↓  매월 리밸런싱
```

---

## 비교 전략

| 전략 | 수익률 입력 | K 결정 |
|------|------------|--------|
| **Cluster v2** (메인) | CAPM 잔차 [C23] | 동적 (PCA) [C23] |
| Cluster v1 (베이스라인) | 원시 수익률 | 고정 K=10 |
| Naive GMVP-LW | — | — (Ledoit-Wolf만 사용) |
| EW Market | — | — |

---

## 성과 검증 지표 [T08]

Tola et al. (2008)의 방법론에 따라 공분산 추정 품질을 정량화합니다.

```
Risk Ratio = σ_pred / σ_real
```

- σ_pred : 추정 공분산으로 계산한 예측 변동성
- σ_real : 실제 실현 변동성
- **1에 가까울수록** 공분산 추정이 정확함
- 클러스터링 전략이 Naive 대비 1에 더 가까운 비율을 보임

---

## 주요 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| COV_LOOKBACK | 36개월 | 상관행렬 추정 롤링 윈도우 |
| K_FIXED | 10 | v1 고정 클러스터 수 |
| K_MIN / K_MAX | 5 / 20 | v2 동적 K 범위 |
| PCA_THRESHOLD | 70% | 동적 K 결정 누적 분산 임계값 |
| R_REPEAT | 5 | K-means 반복 횟수 |
| EWA_BETA | 0.94 | 지수가중 감쇠율 |
| MAX_WEIGHT | 5% | 종목별 비중 상한 |
