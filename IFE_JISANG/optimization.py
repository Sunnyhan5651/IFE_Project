"""
공분산 추정과 롱온리 평균-분산 최적화 (NumPy만 사용).

  covariance      시간 가중 공분산 (균등 또는 지수가중) [K24 4.6]
  ledoit_wolf     항등행렬 목표 수축 (Ledoit & Wolf 2004) — 전체 종목 벤치마크용
  gmvp            min wᵀΣw  s.t. Σw = 1, 0 ≤ w (≤ cap)
  msrp            max μᵀw / √(wᵀΣw)  s.t. Σw = 1, w ≥ 0
"""

import numpy as np  # 수치 계산


def covariance(X, w):
    """(T, n) 수익률의 시간 가중 공분산 (w 합 1, 균등이면 1/T 표본 공분산)."""
    Xc = X - w @ X                                                    # 가중 평균 제거
    return (Xc * w[:, None]).T @ Xc                                   # 가중 공분산 (n, n)


def ledoit_wolf(X, w):
    """Ledoit & Wolf (2004) 항등행렬 목표 수축 공분산. 시간 가중은 관측치를 √(T·w_t)로 재조정해 근사."""
    T, N = X.shape                                                    # 관측 수, 자산 수
    Xc = (X - w @ X) * np.sqrt(T * w)[:, None]                        # 가중 평균 제거 후 가중치 반영 (균등이면 그대로)
    S = Xc.T @ Xc / T                                                 # 가중 공분산
    mu = np.trace(S) / N                                              # 목표 행렬 스케일
    d2 = ((S - mu * np.eye(N)) ** 2).sum()                            # ‖S − μI‖²
    b2_bar = ((Xc ** 2).sum(1) ** 2).sum() / T**2 - (S ** 2).sum() / T  # (1/T²) Σ_t ‖x_t x_tᵀ − S‖²
    b2 = min(b2_bar, d2)                                              # 상한 적용
    delta = b2 / d2 if d2 > 0 else 1.0                                # 수축 강도
    return delta * mu * np.eye(N) + (1.0 - delta) * S                 # 수축 공분산


def project_simplex(v, cap=None):
    """{w ≥ 0, Σw = 1, (w ≤ cap)} 위로의 유클리드 투영 (이분법)."""
    upper = np.inf if cap is None else cap                            # 상한 (없으면 무한대)
    lo, hi = v.min() - 1.0, v.max()                                   # 이동량 τ의 탐색 구간
    for _ in range(100):                                              # 이분법 반복
        tau = (lo + hi) / 2.0                                         # 중간값
        s = np.clip(v - tau, 0.0, upper).sum()                        # 투영 후 합
        lo, hi = (tau, hi) if s > 1.0 else (lo, tau)                  # 합이 1보다 크면 τ 증가
    return np.clip(v - (lo + hi) / 2.0, 0.0, upper)                   # 투영 결과


def _fista(grad, project, x0, L, n_iter=3000, tol=1e-10):
    """가속 투영 경사하강법 (FISTA)."""
    x, y, t = x0.copy(), x0.copy(), 1.0                               # 초기값
    for _ in range(n_iter):                                           # 반복
        x_new = project(y - grad(y) / L)                              # 경사 이동 후 제약 집합으로 투영
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0              # 가속 계수 갱신
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)                 # 모멘텀 적용
        if np.abs(x_new - x).max() < tol:                             # 변화가 충분히 작으면
            return x_new                                              # 수렴
        x, t = x_new, t_new                                           # 상태 갱신
    return x                                                          # 최대 반복 도달 시 결과


def gmvp(Sigma, cap=None):
    """롱온리 최소분산 포트폴리오."""
    n = len(Sigma)                                                    # 자산 수
    if cap is not None and cap * n < 1.0:                             # 상한이 너무 작아 실현 불가능하면
        cap = None                                                    # 상한 해제
    L = 2.0 * np.linalg.eigvalsh(Sigma)[-1] + 1e-12                   # 기울기의 립시츠 상수
    grad = lambda w: 2.0 * Sigma @ w                                  # 분산의 기울기
    proj = lambda v: project_simplex(v, cap)                          # 제약 집합 투영
    return _fista(grad, proj, np.full(n, 1.0 / n), L)                 # 동일가중에서 출발


def _project_budget(v, mu):
    """{y ≥ 0, μᵀy = 1} 위로의 투영: y = max(v + νμ, 0), ν는 이분법."""
    g = lambda nu: mu @ np.maximum(v + nu * mu, 0.0)                  # ν에 대해 단조 증가
    lo, hi = -1.0, 1.0                                                # 초기 탐색 구간
    while g(lo) > 1.0:                                                # 하한이 부족하면
        lo *= 2.0                                                     # 구간 확장
    while g(hi) < 1.0:                                                # 상한이 부족하면
        hi *= 2.0                                                     # 구간 확장
    for _ in range(100):                                              # 이분법
        mid = (lo + hi) / 2.0                                         # 중간값
        lo, hi = (mid, hi) if g(mid) < 1.0 else (lo, mid)             # 구간 축소
    return np.maximum(v + (lo + hi) / 2.0 * mu, 0.0)                  # 투영 결과


def msrp(mu, Sigma):
    """롱온리 최대 샤프 포트폴리오. min yᵀΣy s.t. μᵀy = 1, y ≥ 0 → w = y / Σy."""
    if mu.max() <= 0:                                                 # 양의 기대수익 자산이 없으면
        return gmvp(Sigma), False                                     # 정의 불가 → GMVP로 대체
    n = len(Sigma)                                                    # 자산 수
    L = 2.0 * np.linalg.eigvalsh(Sigma)[-1] + 1e-12                   # 립시츠 상수
    y0 = _project_budget(np.full(n, 1.0 / n), mu)                     # 실현 가능한 시작점
    y = _fista(lambda z: 2.0 * Sigma @ z, lambda z: _project_budget(z, mu), y0, L)  # 볼록 문제 풀이
    return y / y.sum(), True                                          # 비중으로 정규화
