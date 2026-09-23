"""
클러스터 수 K 결정 [C23 2.3]과 부호 그래프 클러스터링 [K24 3장].

상관행렬 Γ를 부호 있는 그래프의 인접행렬로 보고(대각선 0),
스펙트럴 임베딩을 만든 뒤 k-means++로 K개 그룹을 찾는다.
"""

import numpy as np  # 수치 계산


def correlation(X, w):
    """(T, N) 수익률과 시간 가중치 w로 N×N 가중 상관행렬을 계산한다."""
    X_c = X - w @ X                                                   # 종목별 가중 평균 제거
    S = (X_c * w[:, None]).T @ X_c                                    # 가중 공분산
    sd = np.sqrt(np.clip(np.diag(S), 1e-18, None))                    # 종목별 가중 표준편차
    C = S / np.outer(sd, sd)                                          # 상관행렬로 변환
    return np.clip(C, -1.0, 1.0)                                      # 수치 오차로 범위를 넘는 값 보정


def choose_k(C, T_eff, cfg):
    """세 가지 K 후보를 모두 계산하고, 설정된 방식의 값을 반환한다. T_eff는 유효 관측치 수."""
    N = C.shape[0]                                                    # 종목 수
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]                        # 고유값 내림차순
    lam_plus = (1.0 + np.sqrt(N / T_eff)) ** 2                        # Marchenko–Pastur 상한 λ+ (가중이면 근사)
    k_mp = int((eig > lam_plus).sum())                                # λ+를 넘는 고유값 개수 [C23 식 (5)]
    share = np.cumsum(eig) / eig.sum()                                # 누적 분산 비율
    k_var = int(np.searchsorted(share, cfg.k_var_threshold) + 1)      # 기준 P에 도달하는 최소 개수 [C23 식 (6)]
    candidates = {"fixed": cfg.k_fixed, "mp": k_mp, "var": k_var}     # 진단용으로 세 값 모두 보관
    k = candidates[cfg.k_method]                                      # 설정된 방식의 K 선택
    if cfg.k_method != "fixed":                                       # 동적 K일 때만
        k = int(np.clip(k, *cfg.k_bounds))                            # 하한/상한 적용
    diag = {"K_fixed": cfg.k_fixed, "K_mp": k_mp, "K_var": k_var,     # 진단 기록
            "T_eff": float(T_eff),                                    # 유효 관측치 수
            "top_eig_share": float(eig[0] / eig.sum())}               # 첫 고유값 비중
    return k, diag                                                    # 사용할 K와 진단 정보


def _inv_sqrt_degree(d):
    """차수 벡터의 −1/2 제곱 (차수 0이면 0)."""
    out = np.zeros_like(d)                                            # 결과 벡터 초기화
    pos = d > 1e-12                                                   # 차수가 양수인 노드
    out[pos] = 1.0 / np.sqrt(d[pos])                                  # D^{-1/2} 대각 원소
    return out                                                        # 반환


def _sym_laplacian(A):
    """비음수 인접행렬 A의 대칭 정규화 라플라시안 I − D^{-1/2} A D^{-1/2}."""
    s = _inv_sqrt_degree(A.sum(axis=1))                               # D^{-1/2}
    return np.eye(len(A)) - s[:, None] * A * s[None, :]               # 대칭 정규화 라플라시안


def embedding_signed_laplacian(C, K):
    """[K24 3.2] Signed Laplacian: 차수는 |Γ|의 합, 최소 K개 고유벡터를 행 정규화."""
    A = C.copy()                                                      # 원본 보호
    np.fill_diagonal(A, 0.0)                                          # 자기 자신과의 연결 제거
    s = _inv_sqrt_degree(np.abs(A).sum(axis=1))                       # D̄^{-1/2}, D̄ = Σ|Γ_ij|
    L = np.eye(len(A)) - s[:, None] * A * s[None, :]                  # 부호 정규화 라플라시안
    _, vecs = np.linalg.eigh(L)                                       # 고유값 오름차순 고유벡터
    U = vecs[:, :K]                                                   # 가장 작은 K개
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)  # 행 정규화


def embedding_sponge_sym(C, K, tau_pos, tau_neg):
    """[K24 3.4] SPONGE_sym: (L+_sym + τ−I, L−_sym + τ+I)의 최소 K개 일반화 고유벡터."""
    A = C.copy()                                                      # 원본 보호
    np.fill_diagonal(A, 0.0)                                          # 자기 연결 제거
    A_pos = np.maximum(A, 0.0)                                        # 양의 상관 그래프 Γ+
    A_neg = np.maximum(-A, 0.0)                                       # 음의 상관 그래프 Γ−
    M1 = _sym_laplacian(A_pos) + tau_neg * np.eye(len(A))             # 분자 행렬
    M2 = _sym_laplacian(A_neg) + tau_pos * np.eye(len(A))             # 분모 행렬 (양정치)
    Lc = np.linalg.cholesky(M2)                                       # M2 = Lc Lcᵀ
    Lc_inv = np.linalg.inv(Lc)                                        # Lc의 역행렬
    S = Lc_inv @ M1 @ Lc_inv.T                                        # 일반화 문제를 표준 대칭 문제로 변환
    S = (S + S.T) / 2.0                                               # 수치 대칭성 보정
    _, Y = np.linalg.eigh(S)                                          # 고유값 오름차순 고유벡터
    return Lc_inv.T @ Y[:, :K]                                        # 원래 좌표로 되돌린 K차원 임베딩


def kmeans_pp(Z, K, rng, n_init=3, max_iter=200):
    """k-means++ 초기화를 n_init번 시도하고 관성이 가장 작은 라벨을 반환한다."""
    n = len(Z)                                                        # 점 개수
    best_labels, best_inertia = None, np.inf                          # 최적 결과 저장용
    for _ in range(n_init):                                           # 초기화 반복
        centers = [Z[rng.integers(n)]]                                # 첫 중심 무작위 선택
        for _ in range(K - 1):                                        # 나머지 중심 선택
            d2 = ((Z[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(1)  # 가장 가까운 중심까지 거리²
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)            # 거리² 비례 확률
            centers.append(Z[rng.choice(n, p=p)])                    # 확률적으로 다음 중심 선택
        centers = np.array(centers)                                   # 배열로 변환
        labels = np.full(n, -1)                                       # 라벨 초기화
        for _ in range(max_iter):                                     # Lloyd 반복
            d2 = ((Z[:, None, :] - centers[None]) ** 2).sum(-1)       # 점-중심 거리² (n, K)
            new = d2.argmin(1)                                        # 가장 가까운 중심 배정
            for k in range(K):                                        # 빈 클러스터 처리
                if not (new == k).any():                              # 배정된 점이 없으면
                    far = d2[np.arange(n), new].argmax()              # 현재 가장 먼 점을
                    new[far] = k                                      # 그 클러스터로 옮김
            if np.array_equal(new, labels):                           # 변화가 없으면
                break                                                 # 수렴
            labels = new                                              # 라벨 갱신
            centers = np.array([Z[labels == k].mean(0) for k in range(K)])  # 중심 갱신
        inertia = ((Z - centers[labels]) ** 2).sum()                  # 군집 내 제곱합
        if inertia < best_inertia:                                    # 더 좋은 결과면
            best_labels, best_inertia = labels.copy(), inertia        # 저장
    return best_labels                                                # 최적 라벨 반환


def cluster_embedding(C, K, cfg):
    """설정된 알고리즘으로 스펙트럴 임베딩을 만든다 (반복마다 재사용)."""
    if cfg.cluster_algo == "sponge_sym":                              # SPONGE_sym 선택 시
        return embedding_sponge_sym(C, K, cfg.tau_pos, cfg.tau_neg)   # 일반화 고유벡터 임베딩
    if cfg.cluster_algo == "signed_laplacian":                        # Signed Laplacian 선택 시
        return embedding_signed_laplacian(C, K)                       # 부호 라플라시안 임베딩
    raise ValueError(f"unknown cluster_algo: {cfg.cluster_algo}")     # 잘못된 설정 방지
