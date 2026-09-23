"""
성과 평가용 외부 데이터 준비 (전략 계산에는 사용하지 않음).

  RF      Ken French 데이터 라이브러리 F-F_Research_Data_Factors의 월별 RF (1개월 T-bill, %)
  SP500TR S&P 500 총수익지수(^SP500TR, 배당 재투자) — ADJ_PRC가 배당을 반영하므로 총수익지수와 비교
  SPY     SPY ETF 배당 조정 가격 (^SP500TR 검증용)

월 수익률은 일별 데이터에서 각 월 마지막 거래일 종가로 계산해 CRSP 월말 날짜와 맞춘다.
실행:  python benchmarks.py   → data/benchmarks_monthly.csv
"""

import io                            # 메모리 안 텍스트 처리
import zipfile                       # zip 파일 읽기
from pathlib import Path             # 경로 처리
import pandas as pd                  # 표 형태 데이터 처리
import yfinance as yf                # Yahoo Finance 데이터 다운로드
from config import PROJECT_DIR       # 프로젝트 폴더 경로

DATA_DIR = PROJECT_DIR / "data"                                       # 데이터 폴더
FF_ZIP = DATA_DIR / "F-F_Research_Data_Factors_CSV.zip"               # Ken French 팩터 zip (미리 다운로드)
OUT_CSV = DATA_DIR / "benchmarks_monthly.csv"                         # 결과 파일


def load_ff_rf(path=FF_ZIP):
    """Ken French 월별 RF를 소수(예: 0.0025)로 읽는다."""
    with zipfile.ZipFile(path) as z:                                  # zip 열기
        text = z.read(z.namelist()[0]).decode("latin-1")              # 안의 CSV 텍스트
    lines = text.splitlines()                                         # 줄 단위로 분리
    rows = []                                                         # 월별 행 저장
    for line in lines:                                                # 한 줄씩
        parts = [p.strip() for p in line.split(",")]                  # 쉼표로 분리
        if len(parts) == 5 and len(parts[0]) == 6 and parts[0].isdigit():  # "YYYYMM, Mkt-RF, SMB, HML, RF" 형태만
            rows.append((parts[0], float(parts[4]) / 100.0))          # (연월, RF 소수) 저장
        elif rows and "Annual" in line:                               # 연간 표가 시작되면
            break                                                     # 월별 구간 종료
    rf = pd.Series(dict(rows), name="RF")                             # 연월 → RF
    rf.index = pd.PeriodIndex(rf.index, freq="M")                     # 월 단위 인덱스
    return rf                                                         # 월별 RF


def monthly_from_daily(ticker, start="2004-11-01", end="2021-01-10"):
    """일별 종가에서 월말(마지막 거래일) 기준 월 수익률을 만든다."""
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]  # 배당 조정 일별 종가
    px = px.squeeze("columns") if isinstance(px, pd.DataFrame) else px  # 단일 열이면 Series로
    month_end = px.groupby(px.index.to_period("M")).last()            # 각 월 마지막 거래일 종가
    return month_end.pct_change().rename(ticker)                      # 월 수익률


def build():
    """RF, S&P 500 총수익지수, SPY 월 수익률을 합쳐 저장한다."""
    rf = load_ff_rf()                                                 # 월별 RF
    sp = monthly_from_daily("^SP500TR")                               # S&P 500 총수익지수 월 수익률
    spy = monthly_from_daily("SPY")                                   # SPY 월 수익률
    df = pd.concat([rf, sp.rename("SP500TR"), spy.rename("SPY")], axis=1)  # 연월 기준 합치기
    df = df.loc["2005-01":"2020-12"]                                  # 백테스트 기간
    df.index.name = "month"                                           # 인덱스 이름
    df.to_csv(OUT_CSV)                                                # 저장
    return df                                                         # 반환


def load_benchmarks(dates):
    """CRSP 월말 날짜(DatetimeIndex)에 맞춰 RF, SP500TR, SPY를 정렬해 반환한다."""
    df = pd.read_csv(OUT_CSV, index_col="month")                      # 저장된 파일 읽기
    df.index = pd.PeriodIndex(df.index, freq="M")                     # 월 단위 인덱스
    out = df.reindex(dates.to_period("M"))                            # 백테스트 날짜의 연월에 맞추기
    out.index = dates                                                 # 원래 월말 날짜로 복원
    return out                                                        # 정렬된 외부 데이터


if __name__ == "__main__":                                            # 스크립트로 실행할 때만
    bench = build()                                                   # 데이터 생성
    print(bench.head(3).round(5).to_string())                         # 앞부분 확인
    print(bench.tail(3).round(5).to_string())                         # 뒷부분 확인
    print("결측:", bench.isna().sum().to_dict())                       # 결측 확인
    diff = (bench["SP500TR"] - bench["SPY"])                          # 두 S&P 500 대리변수 차이
    print(f"SP500TR − SPY 월평균 차이 {diff.mean()*1e4:.1f}bp, 연환산 약 {diff.mean()*12*100:.2f}%p, 상관 {bench['SP500TR'].corr(bench['SPY']):.4f}")  # 검증
    ann = lambda r: (1 + r).prod() ** (12 / len(r)) - 1               # 연환산 수익률
    print(f"2005-2020 연환산: SP500TR {ann(bench['SP500TR']):.2%}, SPY {ann(bench['SPY']):.2%}, RF {ann(bench['RF']):.2%}")  # 수준 확인
