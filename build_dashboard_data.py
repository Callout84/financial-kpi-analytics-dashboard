"""Pipeline de consolidação financeira (screener universe -> dashboard JSON).

Lê os 10 CSVs fragmentados, consolida via ``join_key`` (relação 1:1),
trata outliers/NaN de forma vetorizada e exporta um payload JSON compacto
(agregados de universo, setor, rankings e amostra de risco) consumido pelo
componente React do dashboard.

Premissa central: a base é um *corte transversal* de ~4.668 empresas listadas
(BSE/NSE, valores em INR Cr), e não uma série temporal por entidade. As séries
"temporais" são reconstruídas a partir de colunas-snapshot (10y/7y/5y/3y/atual).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

UPLOADS: Final[Path] = Path("/mnt/user-data/uploads")
OUTPUT: Final[Path] = Path("/home/claude/dashboard_data.json")
MIN_INDUSTRY_SIZE: Final[int] = 5      # nº mínimo de empresas p/ agregado setorial estável
MIN_MCAP_RANKING: Final[float] = 500.0  # corte (Cr) p/ remover ruído de micro-caps em rankings
SCATTER_CAP: Final[int] = 220           # teto de pontos no scatter de risco


def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Limita uma série aos quantis informados (clipping vetorizado).

    Args:
        series: Série numérica a tratar.
        lower: Quantil inferior de corte.
        upper: Quantil superior de corte.

    Returns:
        Série com caudas extremas aparadas, preservando o índice original.
    """
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def load_universe() -> pd.DataFrame:
    """Carrega e consolida os 10 arquivos em um único DataFrame por ``join_key``.

    Returns:
        DataFrame consolidado (uma linha por empresa) com colunas desduplicadas.

    Raises:
        FileNotFoundError: Se algum CSV esperado não existir no diretório.
    """
    files = [
        "ratios_1_final.csv", "ratios_2_final.csv", "Balance_Sheet_final.csv",
        "Annual_P_L_1_final.csv", "Annual_P_L_2_final.csv",
        "cash_flow_statments_final.csv", "other_metrics_final.csv",
        "price_final.csv", "Quarter_P_L_1_final.csv", "t1_prices.csv",
    ]
    try:
        frames = [pd.read_csv(UPLOADS / f) for f in files]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Arquivo de entrada ausente: {exc.filename}") from exc

    merged = frames[0]
    meta_cols = {"Name", "BSE Code", "NSE Code", "Industry", "Current Price",
                 "Market Capitalization"}
    for frame in frames[1:]:
        drop = [c for c in frame.columns if c in merged.columns and c != "join_key"
                and c in meta_cols]
        # mantém colunas meta apenas na primeira ocorrência; demais idênticas removidas
        dup = [c for c in frame.columns if c in merged.columns and c != "join_key"]
        merged = merged.merge(frame.drop(columns=dup), on="join_key", how="inner")
    return merged


def build_universe_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs agregados do universo e séries-snapshot reconstruídas.

    Args:
        df: DataFrame consolidado.

    Returns:
        Dicionário de KPIs com valor corrente, delta (%) e série temporal.
    """
    mcap_series = [
        df["Market Capitalization 10years back"].sum(),
        df["Market Capitalization 7years back"].sum(),
        df["Market Capitalization 5years back"].sum(),
        df["Market Capitalization 3years back"].sum(),
        df["Market Capitalization"].sum(),
    ]
    debt_series = [
        df["Debt 10Years back"].sum(), df["Debt 7Years back"].sum(),
        df["Debt 5Years back"].sum(), df["Debt 3Years back"].sum(),
        df["Debt preceding year"].sum(), df["Debt"].sum(),
    ]
    sales_now, sales_prev = df["Sales last year"].sum(), df["Sales preceding year"].sum()
    profit_now = df["Net profit"].sum()
    profit_prev = df["Net Profit preceding year"].sum()
    npm_now = 100.0 * profit_now / sales_now
    npm_prev = 100.0 * profit_prev / sales_prev

    def _delta(curr: float, prev: float) -> float:
        return round(100.0 * (curr - prev) / abs(prev), 1) if prev else 0.0

    labels5 = ["-10a", "-7a", "-5a", "-3a", "Atual"]
    labels6 = ["-10a", "-7a", "-5a", "-3a", "-1a", "Atual"]
    return {
        "receita": {
            "label": "Receita Agregada (Sales)", "unit": "Cr",
            "value": round(sales_now), "delta": _delta(sales_now, sales_prev),
            "series": [{"t": "-1a", "v": round(sales_prev)},
                       {"t": "Atual", "v": round(sales_now)}],
        },
        "margem": {
            "label": "Margem Líquida do Universo (NPM)", "unit": "%",
            "value": round(npm_now, 1), "delta": round(npm_now - npm_prev, 1),
            "series": [{"t": "-1a", "v": round(npm_prev, 1)},
                       {"t": "Atual", "v": round(npm_now, 1)}],
        },
        "alavancagem": {
            "label": "Dívida Agregada", "unit": "Cr",
            "value": round(debt_series[-1]), "delta": _delta(debt_series[-1], debt_series[-2]),
            "series": [{"t": t, "v": round(v)} for t, v in zip(labels6, debt_series)],
        },
        "valuation": {
            "label": "Market Cap Agregado", "unit": "Cr",
            "value": round(mcap_series[-1]), "delta": _delta(mcap_series[-1], mcap_series[-2]),
            "series": [{"t": t, "v": round(v)} for t, v in zip(labels5, mcap_series)],
        },
        "pe_mediano": round(df["Price to Earning"].median(), 1),
        "roe_mediano": round(df["Return on equity"].median(), 1),
        "de_mediano": round(df["Debt to equity"].median(), 2),
        "n_empresas": int(len(df)),
        "n_setores": int(df["Industry"].nunique()),
    }


def build_industry_agg(df: pd.DataFrame) -> list[dict]:
    """Agrega métricas medianas por setor (groupby vetorizado).

    Args:
        df: DataFrame consolidado.

    Returns:
        Lista de setores (count >= MIN_INDUSTRY_SIZE) com métricas medianas.
    """
    grp = df.groupby("Industry")
    agg = pd.DataFrame({
        "n": grp.size(),
        "mcap": grp["Market Capitalization"].sum(),
        "sales_g5": grp["Sales growth 5Years"].median(),
        "profit_g5": grp["Profit growth 5Years"].median(),
        "roe": grp["Return on equity"].median(),
        "de": grp["Debt to equity"].median(),
        "pe": grp["Price to Earning"].median(),
    }).reset_index()
    agg = agg[agg["n"] >= MIN_INDUSTRY_SIZE].round(2)
    agg = agg.replace({np.nan: None})
    return agg.sort_values("mcap", ascending=False).to_dict("records")


def build_rankings(df: pd.DataFrame) -> dict:
    """Top/bottom empresas por crescimento, filtrando micro-caps.

    Args:
        df: DataFrame consolidado.

    Returns:
        Dicionário com listas de maiores expansões e contrações (5 anos).
    """
    base = df[df["Market Capitalization"] >= MIN_MCAP_RANKING].copy()
    cols = ["Name", "Industry", "Sales growth 5Years", "Profit growth 5Years",
            "Return on equity", "Market Capitalization"]
    clean = base.dropna(subset=["Sales growth 5Years"])[cols].round(2)

    def _fmt(frame: pd.DataFrame) -> list[dict]:
        renamed = frame.rename(columns={
            "Name": "name", "Industry": "industry", "Sales growth 5Years": "sales_g5",
            "Profit growth 5Years": "profit_g5", "Return on equity": "roe",
            "Market Capitalization": "mcap"})
        return renamed.replace({np.nan: None}).to_dict("records")

    top = clean.nlargest(5, "Sales growth 5Years")
    bottom = clean.nsmallest(5, "Sales growth 5Years")
    return {"top_growth": _fmt(top), "bottom_growth": _fmt(bottom)}


def build_risk_scatter(df: pd.DataFrame) -> list[dict]:
    """Amostra de risco: D/E x ROE, cor por zona de Altman, tamanho por mcap.

    Estratégia de amostragem: maiores caps por valor de mercado + empresas em
    zona de distress (Altman < 1.8) com mcap relevante, para que anomalias não
    sejam diluídas pelo volume de micro-caps.

    Args:
        df: DataFrame consolidado.

    Returns:
        Lista de pontos para o scatter plot de risco.
    """
    cols = ["Name", "Industry", "Debt to equity", "Return on equity",
            "Altman Z Score", "Market Capitalization", "Price to Earning"]
    sub = df[cols].dropna(subset=["Debt to equity", "Return on equity", "Altman Z Score"])

    distress = sub[(sub["Altman Z Score"] < 1.8) &
                   (sub["Market Capitalization"] >= 1000)]
    largest = sub.nlargest(SCATTER_CAP, "Market Capitalization")
    sample = pd.concat([largest, distress]).drop_duplicates(subset=["Name"]).head(SCATTER_CAP)

    sample = sample.assign(
        de_clip=_winsorize(sample["Debt to equity"]).round(2),
        roe_clip=_winsorize(sample["Return on equity"]).round(1),
    )
    zone = np.select(
        [sample["Altman Z Score"] < 1.8, sample["Altman Z Score"] < 3.0],
        ["distress", "grey"], default="safe")
    sample = sample.assign(zone=zone)
    out = sample.rename(columns={
        "Name": "name", "Industry": "industry", "de_clip": "de", "roe_clip": "roe",
        "Altman Z Score": "altman", "Market Capitalization": "mcap",
        "Price to Earning": "pe"})[
        ["name", "industry", "de", "roe", "altman", "mcap", "pe", "zone"]]
    return out.round(2).replace({np.nan: None}).to_dict("records")


def build_attention_points(df: pd.DataFrame) -> dict:
    """Quantifica pontos de atenção (outliers de risco) no universo.

    Args:
        df: DataFrame consolidado.

    Returns:
        Contagens de empresas por categoria de risco.
    """
    return {
        "distress": int((df["Altman Z Score"] < 1.8).sum()),
        "grey": int(((df["Altman Z Score"] >= 1.8) & (df["Altman Z Score"] < 3)).sum()),
        "safe": int((df["Altman Z Score"] >= 3).sum()),
        "high_leverage": int((df["Debt to equity"] > 3).sum()),
        "neg_roe_largecap": int(((df["Return on equity"] < 0) &
                                 (df["Market Capitalization"] >= 1000)).sum()),
        "extreme_pe": int((df["Price to Earning"] > 100).sum()),
        "high_pledge": int((df["Pledged percentage"] > 50).sum()),
    }


def build_correlation(df: pd.DataFrame) -> dict:
    """Correlação cross-file: retorno 1 ano x variação de ROE (winsorizada).

    Args:
        df: DataFrame consolidado.

    Returns:
        Coeficiente, N e amostra (bucketizada) para gráfico de dispersão leve.
    """
    sub = df[["Return over 1year", "Return on equity",
              "Return on equity preceding year"]].dropna()
    sub = sub.assign(roe_delta=sub["Return on equity"] -
                     sub["Return on equity preceding year"])
    x = _winsorize(sub["roe_delta"])
    y = _winsorize(sub["Return over 1year"])
    return {"coef": round(float(x.corr(y)), 3), "n": int(len(sub))}


def main() -> None:
    """Orquestra o pipeline e persiste o JSON consolidado."""
    df = load_universe()
    payload = {
        "meta": {"source": "Universo BSE/NSE (screener export)", "currency": "INR Cr",
                 "n_companies": int(len(df))},
        "kpis": build_universe_kpis(df),
        "industries": build_industry_agg(df),
        "rankings": build_rankings(df),
        "scatter": build_risk_scatter(df),
        "attention": build_attention_points(df),
        "correlation": build_correlation(df),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"OK -> {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KB)")
    print("KPIs:", {k: v for k, v in payload["kpis"].items() if not isinstance(v, dict)})
    print("Setores:", len(payload["industries"]), "| Scatter pts:", len(payload["scatter"]))
    print("Atenção:", payload["attention"])
    print("Correlação:", payload["correlation"])


if __name__ == "__main__":
    main()
