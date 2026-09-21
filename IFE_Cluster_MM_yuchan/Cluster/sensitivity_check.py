"""
변동성 관리 레이어 강건성 · 과최적화 점검
==========================================
IFE_Cluster_fast.py 실행 후, 저장된 월별 수익률만 가지고 몇 초 만에 돌아간다.
(백테스트 루프 불필요)

[1] 파라미터 민감도   : 레버리지 상한 × c 추정 시작(burn-in)
[2] 변동성 신호 정의   : 분산 / 표본분산 / 표준편차 / 3개월 평균
[3] 표본 분할         : 전반(2003~2011) / 후반(2012~2020)
[4] 위기 구간 제외     : 금융위기·코로나를 빼도 Sharpe 우위가 남는가
[5] 연도별 jackknife  : 한 해씩 빼면서 Sharpe 차이가 어느 해에 의존하는지
[6] 최악의 달 제외     : 원본 기준 최악 k개월을 빼면 어떻게 되는가
[7] 블록 부트스트랩    : Sharpe 차이의 분포 (전체 / 금융위기 제외)
[8] 그래프            : cluster_sensitivity_mm.png

주의: 이 스크립트의 Sharpe 는 산술평균 기준
      (평균 초과수익 × 12) / (표준편차 × √12).
      월을 빼고 계산하는 검정이 많아 기하평균 대신 산술평균을 쓴다.
      cluster_metrics_mm.csv 의 기하평균 기준 값과 소수점 차이가 날 수 있다.

출력: cluster_sensitivity_mm.csv, cluster_sensitivity_mm.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)

_installed = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if _f in _installed:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

CAP, BURN = 1.5, 24          # 본 분석에서 채택한 설정
BOOT_N, BOOT_BLOCK = 2000, 12
GFC    = ("2008-09", "2009-03")
COVID  = ("2020-02", "2020-04")

base = pd.read_csv(os.path.join(_DIR, "cluster_monthly_returns_base_mm.csv"),
                   index_col=0, parse_dates=True)
mrv  = pd.read_csv(os.path.join(_ROOT, "data", "market_rv_monthly.csv"), index_col=0, parse_dates=True)
tb   = pd.read_csv(os.path.join(_ROOT, "data", "tbill3m_monthly.csv"), index_col=0,
                   parse_dates=True)["DTB3_month_end_pct"]
mrv.index, tb.index = mrv.index.to_period("M"), tb.index.to_period("M")

P  = base.index.to_period("M")
rf = pd.Series((((1 + tb.shift(1) / 100) ** (1 / 12) - 1).reindex(P)).values, index=base.index)
V2 = base.columns[0]

SIGNALS = {
    "RV2 (기본, 분산)":   mrv["RV2"],
    "표본분산":           mrv["RV2_var"],
    "표준편차 (1/σ)":     np.sqrt(mrv["RV2"]),
    "최근 3개월 평균 RV":  mrv["RV2"].rolling(3).mean(),
}


def prev(sig):
    return pd.Series(sig.reindex(P - 1).values, index=base.index)


def vm_weights(x, s, cap=CAP, burn=BURN):
    """실시간 c (과거 자료만) → w_t = clip(c_t / s_{t-1}, 0, cap)"""
    xv, sv = x.values, s.values
    w = np.ones(len(xv))
    for j in range(burn, len(xv)):
        past = xv[:j]
        w[j] = np.clip(past.std(ddof=1) / (past / sv[:j]).std(ddof=1) / sv[j], 0.0, cap)
    return pd.Series(w, index=x.index)


def sharpe(x):
    """산술 Sharpe (초과수익률 시리즈 입력)"""
    x = x.dropna()
    return x.mean() / x.std(ddof=1) * np.sqrt(12) if len(x) > 2 else np.nan


def mdd(r):
    cum = (1 + r).cumprod()
    return (cum / cum.cummax() - 1).min()


x_v2  = base[V2] - rf
s_rv  = prev(SIGNALS["RV2 (기본, 분산)"])
w_v2  = vm_weights(x_v2, s_rv)
vm_x  = w_v2 * x_v2                      # 관리 후 초과수익률
vm_r  = rf + vm_x
rows  = {}

print(f"기준선: 원본 Sharpe {sharpe(x_v2):.3f} / MDD {mdd(base[V2]):.1%}   "
      f"VM Sharpe {sharpe(vm_x):.3f} / MDD {mdd(vm_r):.1%}\n")

# [1] 파라미터 민감도 ------------------------------------------------
print("[1] 레버리지 상한 × burn-in (Sharpe / MDD, Cluster v2)")
hdr = "".join(f"{'burn ' + str(b):>24s}" for b in (24, 36, 60))
print(f"{'':10s}{hdr}")
for cap in (1.0, 1.25, 1.5, 2.0, 3.0, np.inf):
    cells = []
    for burn in (24, 36, 60):
        w = vm_weights(x_v2, s_rv, cap, burn)
        cells.append(f"{sharpe(w * x_v2):5.2f} / {mdd(rf + w * x_v2) * 100:6.1f}%")
        rows[f"[1] cap={cap} burn={burn}"] = {"Sharpe": sharpe(w * x_v2), "MDD": mdd(rf + w * x_v2)}
    print(f"cap {cap:<6}" + "".join(f"{c:>24s}" for c in cells))

# [2] 신호 정의 ------------------------------------------------------
print("\n[2] 변동성 신호 정의 (cap 1.5, burn 24)")
for nm, sig in SIGNALS.items():
    w = vm_weights(x_v2, prev(sig))
    print(f"    {nm:20s} Sharpe {sharpe(w * x_v2):5.2f}   MDD {mdd(rf + w * x_v2) * 100:6.1f}%")
    rows[f"[2] {nm}"] = {"Sharpe": sharpe(w * x_v2), "MDD": mdd(rf + w * x_v2)}

# [3] 표본 분할 ------------------------------------------------------
print("\n[3] 표본 분할")
for nm, a, b in [("전체 2003~2020", "2003", "2020"), ("전반 2003~2011", "2003", "2011"),
                 ("후반 2012~2020", "2012", "2020")]:
    s0, s1 = sharpe(x_v2.loc[a:b]), sharpe(vm_x.loc[a:b])
    m0, m1 = mdd(base[V2].loc[a:b]), mdd(vm_r.loc[a:b])
    print(f"    {nm:16s} 원본 {s0:5.2f} / {m0*100:6.1f}%   →   VM {s1:5.2f} / {m1*100:6.1f}%"
          f"   (ΔSharpe {s1-s0:+.2f})")
    rows[f"[3] {nm}"] = {"Sharpe": s1, "MDD": m1, "원본 Sharpe": s0, "원본 MDD": m0}

# [4] 위기 구간 제외 --------------------------------------------------
print("\n[4] 위기 구간 제외 (해당 월을 표본에서 빼고 계산)")
def drop(idx_periods, *wins):
    keep = pd.Series(True, index=idx_periods)
    for a, b in wins:
        keep.loc[a:b] = False
    return keep.values

for nm, wins in [("전체 표본", []), ("금융위기 제외", [GFC]), ("코로나 제외", [COVID]),
                 ("둘 다 제외", [GFC, COVID])]:
    k = drop(P, *wins)
    s0, s1 = sharpe(x_v2[k]), sharpe(vm_x[k])
    print(f"    {nm:14s} (n={k.sum():3d})  원본 {s0:5.2f}  →  VM {s1:5.2f}   (ΔSharpe {s1-s0:+.2f})")
    rows[f"[4] {nm}"] = {"Sharpe": s1, "원본 Sharpe": s0, "ΔSharpe": s1 - s0, "n": int(k.sum())}

# [5] 연도별 jackknife -----------------------------------------------
print("\n[5] 연도별 제외 — ΔSharpe(VM - 원본)")
years = sorted(set(base.index.year))
jk = {}
for y in years:
    k = base.index.year != y
    jk[y] = sharpe(vm_x[k]) - sharpe(x_v2[k])
jk_s = pd.Series(jk)
full_d = sharpe(vm_x) - sharpe(x_v2)
print(f"    전체 ΔSharpe {full_d:+.3f} | 한 해 제외 시 범위 {jk_s.min():+.3f} ~ {jk_s.max():+.3f}")
print(f"    가장 크게 줄어드는 해: {jk_s.idxmin()} 제외 시 {jk_s.min():+.3f} "
      f"(전체 대비 {jk_s.min()-full_d:+.3f})")
print(f"    ΔSharpe 가 0 이하가 되는 해: "
      f"{[int(y) for y in jk_s[jk_s <= 0].index] or '없음'}")
for y, v in jk_s.items():
    rows[f"[5] {y} 제외"] = {"ΔSharpe": v}

# [6] 최악의 달 제외 --------------------------------------------------
print("\n[6] 원본 기준 최악의 달 k개 제외")
order = x_v2.sort_values().index
for k_ in (0, 1, 3, 5, 10):
    keep = ~base.index.isin(order[:k_])
    s0, s1 = sharpe(x_v2[keep]), sharpe(vm_x[keep])
    print(f"    최악 {k_:2d}개월 제외  원본 {s0:5.2f}  →  VM {s1:5.2f}   (ΔSharpe {s1-s0:+.2f})")
    rows[f"[6] 최악 {k_}개월 제외"] = {"Sharpe": s1, "원본 Sharpe": s0, "ΔSharpe": s1 - s0}

# [7] 블록 부트스트랩 -------------------------------------------------
def boot_dsharpe(a, b, n_boot=BOOT_N, block=BOOT_BLOCK, seed=0):
    rs = np.random.RandomState(seed)
    n, nb = len(a), int(np.ceil(len(a) / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        st = rs.randint(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        aa, bb = a[idx], b[idx]
        out[i] = (bb.mean() / bb.std(ddof=1) - aa.mean() / aa.std(ddof=1)) * np.sqrt(12)
    return out

print("\n[7] 블록 부트스트랩 — ΔSharpe 분포")
d_all = boot_dsharpe(x_v2.values, vm_x.values)
k_gfc = drop(P, GFC)
d_exg = boot_dsharpe(x_v2[k_gfc].values, vm_x[k_gfc].values)
for nm, d in [("전체 표본", d_all), ("금융위기 제외", d_exg)]:
    print(f"    {nm:12s} P(VM 우위) {np.mean(d > 0):5.1%}   중앙값 {np.median(d):+.2f}   "
          f"90% 구간 {np.percentile(d, 5):+.2f} ~ {np.percentile(d, 95):+.2f}")
    rows[f"[7] 부트스트랩 {nm}"] = {"P(VM 우위)": np.mean(d > 0), "ΔSharpe 중앙값": np.median(d),
                                 "5%": np.percentile(d, 5), "95%": np.percentile(d, 95)}

pd.DataFrame(rows).T.to_csv(os.path.join(_DIR, "cluster_sensitivity_mm.csv"),
                            encoding="utf-8-sig", float_format="%.4f")

# [8] 그래프 ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("변동성 관리 레이어 — Sharpe 개선이 특정 구간에 의존하는가 (Cluster v2)",
             fontsize=13, fontweight="bold", y=1.02)

ax = axes[0]
roll = (vm_x.rolling(36).mean() / vm_x.rolling(36).std(ddof=1)
        - x_v2.rolling(36).mean() / x_v2.rolling(36).std(ddof=1)) * np.sqrt(12)
ax.axhspan(-10, 0, color="#d62728", alpha=0.06)
ax.fill_between(roll.index, 0, roll.values, where=roll.values >= 0, color="#7b2cbf", alpha=0.6)
ax.fill_between(roll.index, 0, roll.values, where=roll.values < 0, color="#d62728", alpha=0.6)
for a, b in (GFC, COVID):
    ax.axvline(pd.Period(a).start_time, color="#555555", lw=0.8, ls=":")
ax.axhline(0, color="black", lw=1)
ax.set_ylim(roll.min() * 1.15, roll.max() * 1.15)
ax.set_title(f"(a) 36개월 롤링 ΔSharpe\nVM 우위 구간 {np.mean(roll.dropna() > 0):.0%}", fontsize=11, loc="left")
ax.set_ylabel("ΔSharpe (VM - 원본)")
ax.grid(alpha=0.3)

ax = axes[1]
cols = ["#d62728" if v == jk_s.min() else "#7b2cbf" for v in jk_s.values]
ax.bar(jk_s.index.astype(str), jk_s.values, color=cols, alpha=0.85)
ax.axhline(full_d, color="black", lw=1.2, ls="--", label=f"전체 표본 {full_d:+.2f}")
ax.axhline(0, color="black", lw=1)
ax.set_title("(b) 한 해씩 빼고 다시 계산한 ΔSharpe", fontsize=11, loc="left")
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax = axes[2]
ax.hist(d_all, bins=45, color="#7b2cbf", alpha=0.6, label=f"전체 표본 (P {np.mean(d_all>0):.0%})")
ax.hist(d_exg, bins=45, color="#f58231", alpha=0.6, label=f"금융위기 제외 (P {np.mean(d_exg>0):.0%})")
ax.axvline(0, color="black", lw=1.2)
ax.set_title(f"(c) 블록 부트스트랩 {BOOT_N:,}회 — ΔSharpe 분포", fontsize=11, loc="left")
ax.set_xlabel("ΔSharpe (VM - 원본)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_DIR, "cluster_sensitivity_mm.png"), dpi=140, bbox_inches="tight")
plt.close()
print("\n완료: cluster_sensitivity_mm.csv, cluster_sensitivity_mm.png")
