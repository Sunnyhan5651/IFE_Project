"""
한 시점 t의 포트폴리오 구성 [K24 4.1 (i)~(iii)].

  ① 추정용 수익률 준비 (윈저라이징, 필요 시 시장 잔차)
  ② 상관행렬 → K 결정 → 스펙트럴 임베딩
  ③ R번 반복: k-means++ 라벨 → β 방식별로
        가상 ETF 수익률 r*_k = Σ β_i r_i → Σ*, μ* → α (GMVP/MSRP) → w_i = α_k β_i
  ④ R번의 w 평균 (β 방식마다 따로)

세 β 방식은 같은 반복에서 같은 클러스터 라벨을 공유한다 (β 효과만 비교하기 위함).
"""

import numpy as np                          # 수치 계산
from data import winsorize_cross_section, market_residuals, time_weights, effective_n  # 전처리 함수
from clustering import correlation, choose_k, cluster_embedding, kmeans_pp  # 클러스터링 함수
from allocation import intra_cluster_beta   # 클러스터 안 비중 β
from optimization import covariance, gmvp, msrp, project_simplex  # 최적화 함수


def build_portfolios(window, cfg, rng):
    """window: (lookback × N) 수익률 DataFrame. β 방식별 종목 비중, 추정용 수익률, 시간 가중치, 진단 정보를 반환."""
    X = winsorize_cross_section(window, cfg.winsor_q).to_numpy()      # 추정용 수익률 (T, N)
    T, N = X.shape                                                    # 관측 개월 수, 종목 수
    w = time_weights(T, cfg.halflife)                                 # 시간 가중치 (최근일수록 큼)
    X_clu = market_residuals(X, w) if cfg.use_residual else X         # 클러스터링 입력 수익률
    C_clu = correlation(X_clu, w)                                     # 클러스터링용 가중 상관행렬
    C_raw = correlation(X, w)                                         # 원시 수익률 가중 상관 (주변부 β용)
    K, diag = choose_k(C_clu, effective_n(w), cfg)                    # 클러스터 수 결정 (유효 관측치 사용)
    Z = cluster_embedding(C_clu, K, cfg)                              # 스펙트럴 임베딩 (반복 간 공유)

    keys = [(m, o) for m in cfg.intra_methods for o in cfg.objectives]  # (β 방식, 최적화 방식) 조합
    sums = {k: np.zeros(N) for k in keys}                             # 조합별 비중 누적합
    msrp_fallback = 0                                                 # MSRP가 GMVP로 대체된 횟수
    sizes = []                                                        # 클러스터 크기 기록
    first_labels = None                                               # 첫 반복의 클러스터 라벨 (구성 확인용으로 저장)

    for rep in range(cfg.n_repeats):                                  # 반복 R회 [K24 4.1-iii]
        labels = kmeans_pp(Z, K, rng, n_init=cfg.kmeans_n_init)       # 클러스터 라벨
        if rep == 0:                                                  # 첫 반복이면
            first_labels = labels.copy()                              # 라벨 보관
        sizes.append(np.bincount(labels, minlength=K))                # 클러스터 크기 저장
        for method in cfg.intra_methods:                              # β 방식별로
            beta = intra_cluster_beta(method, X, w, C_raw, labels, K, cfg)  # 클러스터 안 비중
            B = np.zeros((N, K))                                      # 종목→클러스터 가중 행렬
            B[np.arange(N), labels] = beta                            # B[i, label_i] = β_i
            X_syn = X @ B                                             # 가상 ETF 수익률 (T, K)
            Sigma = covariance(X_syn, w)                              # K×K 가중 공분산 Σ*
            for obj in cfg.objectives:                                # 최적화 방식별로
                if obj == "gmvp":                                     # 최소분산
                    alpha = gmvp(Sigma)                               # 클러스터 비중 α
                else:                                                 # 최대 샤프
                    alpha, ok = msrp(w @ X_syn, Sigma)                # μ* = 과거 가중 평균 (미래 정보 미사용)
                    msrp_fallback += int(not ok)                      # 대체 횟수 기록
                sums[(method, obj)] += B @ alpha                      # w_i = α_{label_i} × β_i 누적

    weights = {}                                                      # 최종 비중 저장
    for method, obj in keys:                                          # 조합별로
        port_w = sums[(method, obj)] / cfg.n_repeats                  # R회 평균 (시간 가중치 w와 이름 구분)
        if cfg.max_weight is not None:                                # 종목 상한이 있으면
            port_w = project_simplex(port_w, cfg.max_weight)          # 상한 포함 심플렉스로 투영
        weights[f"{method}-{obj}"] = port_w / port_w.sum()            # 합 1로 정규화, 이름 예: "ivp-gmvp"

    sizes = np.concatenate(sizes)                                     # 모든 반복의 클러스터 크기
    diag.update({"K": K, "N": N,                                      # 진단 정보 추가
                 "cluster_size_min": int(sizes.min()),                # 최소 클러스터 크기
                 "cluster_size_max": int(sizes.max()),                # 최대 클러스터 크기
                 "msrp_fallback": msrp_fallback})                     # MSRP 대체 횟수
    return weights, X, w, diag, first_labels                          # 비중, 추정용 수익률, 시간 가중치, 진단, 라벨
