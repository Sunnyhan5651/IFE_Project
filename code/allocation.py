"""
클러스터 안 비중 β (각 클러스터 안에서 합 = 1). 모든 통계량은 시간 가중치 w를 사용한다.

  gaussian  ② [K24 4.2] 클러스터 중심(평균 수익률 시계열)에 가까운 종목일수록 큰 비중
  ivp       ③ [López de Prado 2016] 분산의 역수에 비례
  periphery ⑥ [Pozzi, Di Matteo, Aste 2013] 클러스터 안 평균 상관이 낮은(주변부) 종목일수록 큰 비중
"""

import numpy as np  # 수치 계산


def _normalize_within(raw, labels, K):
    """클러스터마다 합이 1이 되도록 정규화한다."""
    beta = np.zeros_like(raw)                                         # 결과 벡터
    for k in range(K):                                                # 클러스터별로
        m = labels == k                                               # 소속 종목 마스크
        total = raw[m].sum()                                          # 원시 점수 합
        beta[m] = raw[m] / total if total > 0 else 1.0 / m.sum()      # 정규화 (합 0이면 동일가중)
    return beta                                                       # β 반환


def beta_gaussian(X, w, labels, K, sigma_scale):
    """β_i ∝ exp(−d_i² / 2σ²), d_i² = Σ_t w_t (r_it − μ_kt)², σ² = scale × 클러스터 안 평균 d²."""
    raw = np.ones(X.shape[1])                                         # 기본값 (단일 종목 클러스터용)
    for k in range(K):                                                # 클러스터별로
        m = labels == k                                               # 소속 종목 마스크
        if m.sum() < 2:                                               # 종목이 1개면
            continue                                                  # 그대로 1
        R = X[:, m]                                                   # 소속 종목 수익률 (T, n_k)
        mu = R.mean(axis=1, keepdims=True)                            # 클러스터 중심 시계열 (T, 1)
        d2 = w @ (R - mu) ** 2                                        # 종목별 가중 거리²
        sigma2 = sigma_scale * d2.mean()                              # 적응형 커널 폭
        raw[m] = np.exp(-d2 / (2.0 * sigma2)) if sigma2 > 0 else 1.0  # 가우시안 점수
    return _normalize_within(raw, labels, K)                          # 클러스터 안 정규화


def beta_ivp(X, w, labels, K):
    """β_i ∝ 1 / σ_i² (가중 분산)."""
    var = w @ (X - w @ X) ** 2                                        # 종목별 가중 분산
    raw = 1.0 / np.maximum(var, 1e-12)                                # 분산의 역수 (0 나눗셈 방지)
    return _normalize_within(raw, labels, K)                          # 클러스터 안 정규화


def beta_periphery(C_raw, labels, K):
    """β_i ∝ 1 − c̄_i, c̄_i = 같은 클러스터 다른 종목과의 평균 (가중) 상관."""
    raw = np.ones(C_raw.shape[0])                                     # 기본값 (단일 종목 클러스터용)
    for k in range(K):                                                # 클러스터별로
        idx = np.where(labels == k)[0]                                # 소속 종목 인덱스
        if len(idx) < 2:                                              # 종목이 1개면
            continue                                                  # 그대로 1
        sub = C_raw[np.ix_(idx, idx)]                                 # 클러스터 안 상관행렬
        c_bar = (sub.sum(axis=1) - 1.0) / (len(idx) - 1)              # 대각선(자기 자신) 제외 평균 상관
        raw[idx] = np.maximum(1.0 - c_bar, 1e-6)                      # 주변부일수록 큰 점수
    return _normalize_within(raw, labels, K)                          # 클러스터 안 정규화


def intra_cluster_beta(method, X, w, C_raw, labels, K, cfg):
    """설정 이름으로 β 계산 함수를 고른다."""
    if method == "gaussian":                                          # ② Gaussian kernel
        return beta_gaussian(X, w, labels, K, cfg.gauss_sigma_scale)  # 중심 가중
    if method == "ivp":                                               # ③ 역분산
        return beta_ivp(X, w, labels, K)                              # 변동성 반비례
    if method == "periphery":                                         # ⑥ 주변부 가중
        return beta_periphery(C_raw, labels, K)                       # 평균 상관 반비례
    raise ValueError(f"unknown intra method: {method}")               # 잘못된 설정 방지
