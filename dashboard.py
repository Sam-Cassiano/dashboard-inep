"""
dashboard.py — Dashboard Interativo: Infraestrutura Escolar e Rendimento em Alagoas
Censo Escolar 2023 + IDEB 2023 (INEP)

Uso após rodar o pipeline:
    streamlit run dashboard.py -- --dados dados_tratados/

Ou com caminho padrão:
    streamlit run dashboard.py
"""
 
import os
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

# ── Configuração da página ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Infraestrutura Escolar — Alagoas 2023",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de cores ───────────────────────────────────────────────────────────

CORES_DEP = {
    "Federal": "#1f77b4",
    "Estadual": "#ff7f0e",
    "Municipal": "#2ca02c",
    "Privada": "#d62728",
}
CORES_LOC = {"Urbana": "#636EFA", "Rural": "#EF553B"}
CORES_SIM_NAO = {"Sim": "#00CC96", "Não": "#EF553B"}

# ── Carregamento de dados ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Carregando dados…")
def carregar(dados_dir: str) -> dict[str, pd.DataFrame]:
    dfs = {}
    nomes = {
        "a1": "analise1_computadores.parquet",
        "a2": "analise2_banda_larga.parquet",
        "a3": "analise3_equipamentos_av.parquet",
        "a4": "analise4_idxinfra_ideb.parquet",
        "a5": "analise5_aprovacao_dep_loc.parquet",
        "a6": "analise6_vulnerabilidade.parquet",
        "a7": "analise7_inclusao_digital.parquet",
        "a8": "analise8_saneamento.parquet",
        "a9": "analise9_leitura_rendimento.parquet",
        "censo": "censo_al.parquet",
    }
    for key, fname in nomes.items():
        fpath = os.path.join(dados_dir, fname)
        if os.path.exists(fpath):
            dfs[key] = pd.read_parquet(fpath)
        else:
            dfs[key] = pd.DataFrame()
    return dfs


def dados_dir_from_args() -> str:
    """Lê --dados do argv para compatibilidade com 'streamlit run dashboard.py -- --dados ...'"""
    try:
        idx = sys.argv.index("--dados")
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return "dados_tratados"


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def stat_card(col, titulo: str, valor: str, delta: str = ""):
    with col:
        st.metric(titulo, valor, delta)


def rodape_analise(integrante: str, metodo: str):
    st.caption(f"**Integrante:** {integrante} · **Método:** {metodo}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(dfs: dict) -> dict:
    st.sidebar.image(
        "https://www.gov.br/inep/pt-br/@@/image/logo-inep.png/@@images/image",
        width=160,
    )
    st.sidebar.title("Filtros Globais")

    censo = dfs.get("censo", pd.DataFrame())
    filtros = {}

    if not censo.empty:
        deps = sorted(censo["DEPENDENCIA"].dropna().unique())
        filtros["dependencias"] = st.sidebar.multiselect(
            "Dependência Administrativa", deps, default=deps
        )
        locs = sorted(censo["LOCALIZACAO"].dropna().unique())
        filtros["localizacoes"] = st.sidebar.multiselect(
            "Localização", locs, default=locs
        )
    else:
        filtros["dependencias"] = []
        filtros["localizacoes"] = []

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Fonte:** Microdados Censo Escolar 2023 + IDEB 2023 — INEP  \n"
        "**Escopo:** Estado de Alagoas  \n"
        "**Grupo:** Guilherme · Samuel · Matheus"
    )
    return filtros


def aplicar_filtros(df: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    if df.empty:
        return df
    if "DEPENDENCIA" in df.columns and filtros.get("dependencias"):
        df = df[df["DEPENDENCIA"].isin(filtros["dependencias"])]
    if "LOCALIZACAO" in df.columns and filtros.get("localizacoes"):
        df = df[df["LOCALIZACAO"].isin(filtros["localizacoes"])]
    return df


# ── Visão Geral ───────────────────────────────────────────────────────────────

def aba_visao_geral(dfs: dict, filtros: dict):
    st.header("📊 Visão Geral — Escolas de Alagoas (2023)")

    censo = aplicar_filtros(dfs.get("censo", pd.DataFrame()), filtros)
    if censo.empty:
        st.warning("Base principal não carregada. Execute o pipeline primeiro.")
        return

    c1, c2, c3, c4 = st.columns(4)
    stat_card(c1, "Total de Escolas", f"{len(censo):,}")
    stat_card(c2, "Matrículas", f"{censo['QT_MAT_BAS'].sum():,.0f}")
    stat_card(c3, "Com Internet", fmt_pct(censo["IN_INTERNET"].mean() * 100))
    stat_card(c4, "Com IDEB", f"{censo['IDEB_2023'].notna().sum():,}")

    col1, col2 = st.columns(2)
    with col1:
        dep_cnt = censo["DEPENDENCIA"].value_counts().reset_index()
        dep_cnt.columns = ["Dependência", "Escolas"]
        fig = px.pie(dep_cnt, names="Dependência", values="Escolas",
                     color="Dependência", color_discrete_map=CORES_DEP,
                     title="Escolas por Dependência Administrativa")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        loc_cnt = censo["LOCALIZACAO"].value_counts().reset_index()
        loc_cnt.columns = ["Localização", "Escolas"]
        fig = px.bar(loc_cnt, x="Localização", y="Escolas",
                     color="Localização", color_discrete_map=CORES_LOC,
                     text="Escolas",
                     title="Escolas por Localização")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Mapa de calor: infraestrutura por dependência
    st.subheader("Disponibilidade de Infraestrutura por Dependência (%)")
    indicadores = {
        "Internet": "IN_INTERNET",
        "Banda Larga": "IN_BANDA_LARGA",
        "Lab. Informática": "IN_LABORATORIO_INFORMATICA",
        "Lab. Ciências": "IN_LABORATORIO_CIENCIAS",
        "Energia": "IN_ENERGIA_DISPONIVEL",
        "Água Potável": "IN_AGUA_POTAVEL",
        "Esgoto": "IN_ESGOTO_DISPONIVEL",
        "Quadra Esportes": "IN_QUADRA_ESPORTES",
        "Biblioteca/Leitura": "IN_ESPACO_LEITURA",
        "Proj. Multimídia": "IN_EQUIP_MULTIMIDIA",
        "Lousa Digital": "IN_EQUIP_LOUSA_DIGITAL",
    }
    cols_disp = {k: v for k, v in indicadores.items() if v in censo.columns}
    heat_df = censo.groupby("DEPENDENCIA")[[v for v in cols_disp.values()]].mean() * 100
    heat_df = heat_df.rename(columns={v: k for k, v in cols_disp.items()})
    fig = px.imshow(
        heat_df.T,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=100,
        text_auto=".0f",
        title="% de Escolas com Infraestrutura — por Dependência",
        labels={"color": "%"},
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Análise 1 ─────────────────────────────────────────────────────────────────

def aba_analise1(dfs: dict, filtros: dict):
    st.header("A1 · Distribuição de Computadores por Dependência Administrativa")
    st.markdown(
        "**Objetivo:** Identificar disparidades de recursos tecnológicos (desktop + portátil + tablet) "
        "entre redes de ensino em Alagoas."
    )

    df = aplicar_filtros(dfs.get("a1", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    # Estatísticas
    stats_df = df.groupby("DEPENDENCIA")["QT_COMP_ALUNO_TOTAL"].agg(
        Média="mean", Mediana="median", Desvio_Padrão="std", N="count"
    ).round(2).reset_index()
    stats_df.columns = ["Dependência", "Média", "Mediana", "Desvio Padrão", "N de Escolas"]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.violin(
            df, x="DEPENDENCIA", y="QT_COMP_ALUNO_TOTAL_VIS",
            color="DEPENDENCIA", color_discrete_map=CORES_DEP,
            box=True, points=False,
            title="Distribuição de Computadores por Escola (violino + boxplot)",
            labels={"DEPENDENCIA": "Dependência", "QT_COMP_ALUNO_TOTAL_VIS": "Qtd. Computadores (clip 99º pct)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Destaque
    med_mun = stats_df.loc[stats_df["Dependência"] == "Municipal", "Mediana"].values
    med_fed = stats_df.loc[stats_df["Dependência"] == "Federal", "Mediana"].values
    if len(med_mun) and len(med_fed):
        st.info(
            f"📌 **Mediana Federal:** {med_fed[0]:.0f} computadores vs "
            f"**Mediana Municipal:** {med_mun[0]:.0f} — diferença de "
            f"**{med_fed[0]-med_mun[0]:.0f} equipamentos** por escola."
        )

    # Barras: média por dependência
    fig2 = px.bar(
        stats_df, x="Dependência", y="Média",
        color="Dependência", color_discrete_map=CORES_DEP,
        text="Média", title="Média de Computadores por Escola",
        labels={"Média": "Média de Computadores"},
    )
    fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)
    rodape_analise("Guilherme Henrique Costa Lima", "Estatística descritiva · Gráfico de violino")


# ── Análise 2 ─────────────────────────────────────────────────────────────────

def aba_analise2(dfs: dict, filtros: dict):
    st.header("A2 · Internet Banda Larga × Porte da Escola")
    st.markdown(
        "**Objetivo:** Verificar se escolas maiores (mais matrículas) tendem a ter melhor conectividade."
    )

    df = aplicar_filtros(dfs.get("a2", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    df_valido = df[df["QT_MAT_BAS"] > 0].copy()

    # Estatísticas por banda larga
    grp = df_valido.groupby("BANDA_LARGA_LABEL")["QT_MAT_BAS"].agg(
        Média="mean", Mediana="median", N="count"
    ).round(1).reset_index()
    grp.columns = ["Banda Larga", "Matrículas — Média", "Matrículas — Mediana", "N de Escolas"]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.box(
            df_valido, x="BANDA_LARGA_LABEL", y="QT_MAT_BAS",
            color="BANDA_LARGA_LABEL", color_discrete_map={"Com Banda Larga": "#00CC96", "Sem Banda Larga": "#EF553B"},
            points=False,
            title="Distribuição de Matrículas por Disponibilidade de Banda Larga",
            labels={"BANDA_LARGA_LABEL": "Conectividade", "QT_MAT_BAS": "Total de Matrículas"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(grp, use_container_width=True, hide_index=True)

    # Teste t de Welch
    sim = df_valido[df_valido["IN_BANDA_LARGA"] == 1]["QT_MAT_BAS"]
    nao = df_valido[df_valido["IN_BANDA_LARGA"] == 0]["QT_MAT_BAS"]
    if len(sim) > 1 and len(nao) > 1:
        t_stat, p_val = stats.ttest_ind(sim, nao, equal_var=False)
        sig = "✅ Significativa (p < 0,05)" if p_val < 0.05 else "❌ Não significativa"
        p_fmt = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
        st.info(f"**Teste t de Welch:** t = {t_stat:.2f} · p = {p_fmt} · {sig}")

    # Proporção por porte
    st.subheader("Proporção com Banda Larga por Porte da Escola")
    porte_grp = df.groupby("PORTE")["IN_BANDA_LARGA"].mean().mul(100).reset_index()
    porte_grp.columns = ["Porte", "% com Banda Larga"]
    ordem_porte = ["Sem matrícula", "Pequena (<50)", "Média (50-199)", "Grande (200-499)", "Muito grande (≥500)"]
    porte_grp["Porte"] = pd.Categorical(porte_grp["Porte"], categories=ordem_porte, ordered=True)
    porte_grp = porte_grp.sort_values("Porte")
    fig2 = px.bar(
        porte_grp, x="Porte", y="% com Banda Larga",
        text="% com Banda Larga",
        color="% com Banda Larga", color_continuous_scale="RdYlGn",
        title="% de Escolas com Banda Larga por Porte",
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)
    rodape_analise("Guilherme Henrique Costa Lima", "Teste t de Welch · Boxplot")


# ── Análise 3 ─────────────────────────────────────────────────────────────────

def aba_analise3(dfs: dict, filtros: dict):
    st.header("A3 · Projetor Multimídia e Lousa Digital por Localização")
    st.markdown(
        "**Objetivo:** Comparar a proporção de escolas com equipamentos audiovisuais entre zonas urbana e rural."
    )

    df = aplicar_filtros(dfs.get("a3", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    equip = {
        "Projetor Multimídia": "IN_EQUIP_MULTIMIDIA",
        "Lousa Digital": "IN_EQUIP_LOUSA_DIGITAL",
    }

    rows = []
    for nome, col in equip.items():
        for loc in ["Urbana", "Rural"]:
            sub = df[df["LOCALIZACAO"] == loc]
            if sub.empty:
                continue
            pct = sub[col].mean() * 100
            n = len(sub)
            rows.append({"Equipamento": nome, "Localização": loc, "% de Escolas": round(pct, 1), "N": n})
    res = pd.DataFrame(rows)

    # Teste qui-quadrado
    testes = []
    for nome, col in equip.items():
        ct = pd.crosstab(df["LOCALIZACAO"], df[col])
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
            testes.append({"Equipamento": nome, "χ²": round(chi2, 2), "p-valor": f"{p:.4f}",
                           "Conclusão": "Associação significativa" if p < 0.05 else "Não significativa"})

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.bar(
            res, x="Equipamento", y="% de Escolas",
            color="Localização", barmode="group",
            color_discrete_map=CORES_LOC,
            text="% de Escolas",
            title="% de Escolas com Equipamentos por Localização",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Tabela de Resultados**")
        st.dataframe(res, use_container_width=True, hide_index=True)
        if testes:
            st.markdown("**Testes Qui-Quadrado**")
            st.dataframe(pd.DataFrame(testes), use_container_width=True, hide_index=True)

    # Diferença percentual
    if not res.empty:
        for nome in res["Equipamento"].unique():
            urb = res[(res["Equipamento"] == nome) & (res["Localização"] == "Urbana")]["% de Escolas"].values
            rur = res[(res["Equipamento"] == nome) & (res["Localização"] == "Rural")]["% de Escolas"].values
            if len(urb) and len(rur):
                st.metric(f"Gap Urbano-Rural — {nome}", f"{urb[0]:.1f}% vs {rur[0]:.1f}%",
                          delta=f"-{urb[0]-rur[0]:.1f} p.p. no rural", delta_color="inverse")
    rodape_analise("Guilherme Henrique Costa Lima", "Teste Qui-Quadrado · Barras agrupadas")


# ── Análise 4 ─────────────────────────────────────────────────────────────────

def aba_analise4(dfs: dict, filtros: dict):
    st.header("A4 · Índice de Infraestrutura Tecnológica × IDEB")
    st.markdown(
        "**Objetivo:** Avaliar se o índice composto de infraestrutura (energia + banda larga + lab. ciências) "
        "está associado ao IDEB, controlando por dependência e localização."
    )

    df = aplicar_filtros(dfs.get("a4", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.scatter(
            df, x="IDX_INFRA_TEC", y="IDEB_2023",
            color="DEPENDENCIA", color_discrete_map=CORES_DEP,
            size="QT_MAT_BAS", size_max=30,
            opacity=0.6,
            trendline="ols",
            title="IDEB × Índice de Infraestrutura Tecnológica",
            labels={"IDX_INFRA_TEC": "Índice Infra. Tec. (0–3)", "IDEB_2023": "IDEB 2023"},
            hover_data=["NO_ENTIDADE"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_c = df.dropna(subset=["IDX_INFRA_TEC", "IDEB_2023"])
        r, p = (0.0, 1.0)
        if len(df_c) > 2:
            r, p = stats.pearsonr(df_c["IDX_INFRA_TEC"], df_c["IDEB_2023"])
        p_fmt = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        st.metric("Correlação de Pearson (r)", f"{r:.3f}")
        st.metric("p-valor", p_fmt)
        sig = "✅ Significativa" if p < 0.05 else "❌ Não significativa"
        st.markdown(f"**Significância:** {sig}")
        st.markdown("---")
        st.markdown(
            "**Atenção:** A correlação simples pode ser explicada pela "
            "dependência administrativa (variável de confusão)."
        )

    # IDEB médio por nível do índice
    st.subheader("IDEB Médio por Nível do Índice de Infraestrutura")
    grp = df.groupby("IDX_INFRA_TEC")["IDEB_2023"].agg(["mean", "median", "count"]).round(2).reset_index()
    grp.columns = ["Índice (0-3)", "IDEB Médio", "IDEB Mediana", "N Escolas"]
    st.dataframe(grp, use_container_width=True, hide_index=True)

    fig2 = px.box(
        df, x=df["IDX_INFRA_TEC"].astype(str), y="IDEB_2023",
        color="DEPENDENCIA", color_discrete_map=CORES_DEP,
        title="Distribuição do IDEB por Nível de Infraestrutura e Dependência",
        labels={"x": "Índice Infra. Tec.", "IDEB_2023": "IDEB 2023"},
    )
    st.plotly_chart(fig2, use_container_width=True)
    rodape_analise("Samuel Cassiano dos Santos", "Correlação de Pearson · Regressão linear · Dispersão")


# ── Análise 5 ─────────────────────────────────────────────────────────────────

def aba_analise5(dfs: dict, filtros: dict):
    st.header("A5 · Heterogeneidade da Taxa de Aprovação por Dependência e Localização")
    st.markdown(
        "**Objetivo:** Avaliar se a disparidade na taxa de aprovação entre dependências "
        "varia conforme a localização da escola."
    )

    df = aplicar_filtros(dfs.get("a5", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    # Usar IDEB como proxy se TAXA_APROVACAO não disponível
    col_aprov = "TAXA_APROVACAO" if "TAXA_APROVACAO" in df.columns and df["TAXA_APROVACAO"].notna().sum() > 50 else "IDEB_2023"
    label_y = "Taxa de Aprovação (%)" if col_aprov == "TAXA_APROVACAO" else "IDEB 2023"

    df_v = df.dropna(subset=[col_aprov]).copy()
    # Converter escala 0–1 → 0–100 se necessário
    if col_aprov == "TAXA_APROVACAO" and df_v[col_aprov].mean() < 2.0:
        df_v[col_aprov] = df_v[col_aprov] * 100
    if df_v.empty:
        st.warning("Sem dados de aprovação disponíveis neste filtro.")
        return

    # IC 95% por grupo
    rows = []
    for loc in df_v["LOCALIZACAO"].dropna().unique():
        for dep in df_v["DEPENDENCIA"].dropna().unique():
            sub = df_v[(df_v["LOCALIZACAO"] == loc) & (df_v["DEPENDENCIA"] == dep)][col_aprov]
            if len(sub) < 3:
                continue
            m = sub.mean()
            se = stats.sem(sub)
            ic_inf = m - 1.96 * se
            ic_sup = m + 1.96 * se
            rows.append({"Localização": loc, "Dependência": dep, "Média": round(m, 2),
                         "IC_inf": round(ic_inf, 2), "IC_sup": round(ic_sup, 2), "N": len(sub)})
    ic_df = pd.DataFrame(rows)

    if ic_df.empty:
        st.warning("Dados insuficientes para calcular intervalos de confiança.")
        return

    # Painel duplo
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Zona Urbana", "Zona Rural"])
    for i, loc in enumerate(["Urbana", "Rural"], 1):
        sub = ic_df[ic_df["Localização"] == loc]
        for dep in sub["Dependência"].unique():
            row = sub[sub["Dependência"] == dep].iloc[0]
            fig.add_trace(go.Bar(
                name=dep, x=[dep], y=[row["Média"]],
                error_y=dict(type="data", array=[row["IC_sup"] - row["Média"]],
                             arrayminus=[row["Média"] - row["IC_inf"]], visible=True),
                marker_color=CORES_DEP.get(dep, "gray"),
                showlegend=(i == 1),
            ), row=1, col=i)
    fig.update_layout(title_text=f"{label_y} por Dependência e Localização (IC 95%)", barmode="group")
    fig.update_yaxes(title_text=label_y)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela IC
    st.subheader("Dados com Intervalo de Confiança de 95%")
    st.dataframe(
        ic_df.rename(columns={"Média": f"{label_y} Média", "IC_inf": "IC 95% Inf.", "IC_sup": "IC 95% Sup."}),
        use_container_width=True, hide_index=True,
    )

    # ANOVA por zona
    st.subheader("Resultado da ANOVA por Localização")
    for loc in ["Urbana", "Rural"]:
        grupos = [
            df_v[(df_v["LOCALIZACAO"] == loc) & (df_v["DEPENDENCIA"] == dep)][col_aprov].dropna()
            for dep in df_v["DEPENDENCIA"].unique()
        ]
        grupos = [g for g in grupos if len(g) >= 3]
        if len(grupos) >= 2:
            f_stat, p_val = stats.f_oneway(*grupos)
            sig = "✅" if p_val < 0.05 else "❌"
            st.markdown(f"**{loc}:** F = {f_stat:.2f} · p = {p_val:.4f} {sig}")
    rodape_analise("Samuel Cassiano dos Santos", "ANOVA · IC 95% · Barras empilhadas")


# ── Análise 6 ─────────────────────────────────────────────────────────────────

def aba_analise6(dfs: dict, filtros: dict):
    st.header("A6 · Escolas em Situação de Vulnerabilidade Combinada")
    st.markdown(
        "**Objetivo:** Identificar escolas com IDEB abaixo da mediana estadual **e** "
        "ao menos 2 déficits de infraestrutura (sem quadra, sem lab. ciências, sem banda larga)."
    )

    df = aplicar_filtros(dfs.get("a6", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    total = len(df)
    vuln = df[df["VULNERAVEL"] == 1]
    n_vuln = len(vuln)
    pct_vuln = n_vuln / total * 100 if total > 0 else 0

    c1, c2, c3 = st.columns(3)
    stat_card(c1, "Escolas com IDEB disponível", f"{total:,}")
    stat_card(c2, "Escolas Vulneráveis", f"{n_vuln:,}")
    stat_card(c3, "% Vulnerável", fmt_pct(pct_vuln))

    st.info(f"**Mediana estadual do IDEB:** {df['IDEB_2023'].median():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        grp = vuln.groupby(["DEPENDENCIA", "LOCALIZACAO"]).size().reset_index(name="N de Escolas Vulneráveis")
        fig = px.bar(
            grp, x="DEPENDENCIA", y="N de Escolas Vulneráveis",
            color="LOCALIZACAO", barmode="group",
            color_discrete_map=CORES_LOC,
            text="N de Escolas Vulneráveis",
            title="Escolas Vulneráveis por Dependência e Localização",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Treemap
        grp2 = df.groupby(["DEPENDENCIA", "LOCALIZACAO", "VULNERAVEL"]).size().reset_index(name="N")
        grp2["Status"] = grp2["VULNERAVEL"].map({1: "Vulnerável", 0: "Não Vulnerável"})
        fig2 = px.treemap(
            grp2, path=["DEPENDENCIA", "LOCALIZACAO", "Status"], values="N",
            color="Status",
            color_discrete_map={"Vulnerável": "#EF553B", "Não Vulnerável": "#00CC96"},
            title="Proporção de Vulnerabilidade (Treemap)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tabela
    st.subheader("Tabela de Vulnerabilidade")
    tbl = grp.rename(columns={"DEPENDENCIA": "Dependência", "LOCALIZACAO": "Localização"})
    tbl["% do total"] = (tbl["N de Escolas Vulneráveis"] / total * 100).round(1)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Qui-quadrado
    st.subheader("Testes Qui-Quadrado")
    for var, label in [("DEPENDENCIA", "Dependência"), ("LOCALIZACAO", "Localização")]:
        ct = pd.crosstab(df[var], df["VULNERAVEL"])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        sig = "✅ Significativa" if p < 0.05 else "❌ Não significativa"
        st.markdown(f"**{label}:** χ² = {chi2:.2f} · p = {p:.4f} · {sig}")
    rodape_analise("Samuel Cassiano dos Santos", "Qui-Quadrado · Barras agrupadas · Treemap")


# ── Análise 7 ─────────────────────────────────────────────────────────────────

def aba_analise7(dfs: dict, filtros: dict):
    st.header("A7 · Inclusão Digital nas Escolas Estaduais")
    st.markdown(
        "**Objetivo:** Mapear o percentual de escolas estaduais com internet, banda larga e "
        "laboratório de informática — revelando a 'inclusão digital administrativa vs. pedagógica'."
    )

    df = dfs.get("a7", pd.DataFrame())
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    # Aplicar filtro de localização
    if filtros.get("localizacoes"):
        df = df[df["LOCALIZACAO"].isin(filtros["localizacoes"])]

    n = len(df)
    indicadores = {
        "Internet (qualquer)": "IN_INTERNET",
        "Banda Larga": "IN_BANDA_LARGA",
        "Lab. Informática": "IN_LABORATORIO_INFORMATICA",
        "Acesso Internet p/ Alunos": "IN_ACESSO_INTERNET_COMPUTADOR",
    }
    rows = []
    for nome, col in indicadores.items():
        if col in df.columns:
            # Valor 9 = "não se aplica" → tratar como 0
            col_vals = df[col].replace(9, 0)
            total_sim = (col_vals == 1).sum()
            pct = total_sim / n * 100 if n > 0 else 0
            rows.append({"Indicador": nome, "Com": int(total_sim), "Sem": int(n - total_sim), "% com": round(pct, 1)})
    res = pd.DataFrame(rows)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            res, x="Indicador", y="% com",
            color="% com", color_continuous_scale="RdYlGn",
            text="% com",
            title="Inclusão Digital — Escolas Estaduais de Alagoas (%)",
            labels={"% com": "% de Escolas"},
            range_y=[0, 110],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.metric("Total de escolas estaduais", f"{n:,}")

    # Gap: banda larga vs lab informática
    banda = res[res["Indicador"] == "Banda Larga"]["% com"].values
    lab = res[res["Indicador"] == "Lab. Informática"]["% com"].values
    if len(banda) and len(lab):
        gap = banda[0] - lab[0]
        st.warning(
            f"⚠️ **Gap de Inclusão Pedagógica:** {banda[0]:.1f}% das escolas têm banda larga, "
            f"mas apenas {lab[0]:.1f}% têm laboratório de informática. "
            f"Diferença de **{gap:.1f} p.p.** — conectividade sem equipamentos para os alunos."
        )

    # Por localização
    st.subheader("Inclusão Digital por Localização (Escolas Estaduais)")
    loc_rows = []
    for loc in df["LOCALIZACAO"].dropna().unique():
        sub = df[df["LOCALIZACAO"] == loc]
        for nome, col in indicadores.items():
            if col in sub.columns:
                loc_rows.append({
                    "Localização": loc,
                    "Indicador": nome,
                    "% com": round(sub[col].mean() * 100, 1),
                    "N": len(sub),
                })
    if loc_rows:
        fig2 = px.bar(
            pd.DataFrame(loc_rows), x="Indicador", y="% com",
            color="Localização", barmode="group",
            color_discrete_map=CORES_LOC,
            text="% com",
            title="Inclusão Digital por Localização",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    rodape_analise("Matheus Ferreira de Lima", "Estatística descritiva · Barras simples")


# ── Análise 8 ─────────────────────────────────────────────────────────────────

def aba_analise8(dfs: dict, filtros: dict):
    st.header("A8 · Saneamento Básico nas Escolas de Alagoas por Zona")
    st.markdown(
        "**Objetivo:** Analisar a disponibilidade de água potável e esgotamento sanitário "
        "por localização (Urbana vs. Rural)."
    )

    df = aplicar_filtros(dfs.get("a8", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    indicadores = {
        "Água Potável":   "IN_AGUA_POTAVEL",
        "Esgoto":         "IN_ESGOTO_DISPONIVEL",
    }

    rows = []
    for loc in ["Urbana", "Rural"]:
        sub = df[df["LOCALIZACAO"] == loc]
        n = len(sub)
        for nome, col in indicadores.items():
            if col in sub.columns:
                sim = sub[col].sum()
                rows.append({
                    "Localização": loc,
                    "Indicador": nome,
                    "Possui (%)": round(sim / n * 100, 1) if n > 0 else 0,
                    "Não Possui (%)": round((n - sim) / n * 100, 1) if n > 0 else 0,
                    "N": n,
                })
    res = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    for idx, (nome, col) in enumerate(indicadores.items()):
        sub_res = res[res["Indicador"] == nome]
        with (col1 if idx == 0 else col2):
            fig = px.bar(
                sub_res, x="Localização",
                y=["Possui (%)", "Não Possui (%)"],
                barmode="stack",
                color_discrete_map={"Possui (%)": "#00CC96", "Não Possui (%)": "#EF553B"},
                title=f"{nome} por Localização",
                labels={"value": "%", "variable": ""},
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tabela resumo
    st.subheader("Resumo — Saneamento Básico por Zona")
    st.dataframe(
        res[["Localização", "Indicador", "Possui (%)", "Não Possui (%)", "N"]],
        use_container_width=True, hide_index=True,
    )

    # Alertas dinâmicos baseados nos dados
    agua_rural_sem = res[(res["Indicador"] == "Água Potável") & (res["Localização"] == "Rural")]["Não Possui (%)"]
    agua_rural_sim = res[(res["Indicador"] == "Água Potável") & (res["Localização"] == "Rural")]["Possui (%)"]
    if not agua_rural_sem.empty:
        pct_sem = agua_rural_sem.values[0]
        pct_sim = agua_rural_sim.values[0] if not agua_rural_sim.empty else 0
        if pct_sem > 10:
            st.error(
                f"🚨 **{pct_sem:.1f}% das escolas rurais ativas** não têm acesso à água potável — "
                "condição que viola os ODS 3 e 6 e compromete a saúde e permanência dos alunos."
            )
        else:
            st.success(
                f"✅ **{pct_sim:.1f}% das escolas rurais ativas** têm acesso à água potável. "
                f"Apenas {pct_sem:.1f}% sem cobertura — referente ao conjunto de escolas em funcionamento."
            )

    # Qui-quadrado
    st.subheader("Testes Qui-Quadrado — Saneamento × Localização")
    for nome, col in indicadores.items():
        if col not in df.columns:
            continue
        ct = pd.crosstab(df["LOCALIZACAO"], df[col])
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
            sig = "✅" if p < 0.05 else "❌"
            st.markdown(f"**{nome}:** χ² = {chi2:.2f} · p = {p:.4f} {sig}")
    rodape_analise("Matheus Ferreira de Lima", "Qui-Quadrado · Barras empilhadas")


# ── Análise 9 ─────────────────────────────────────────────────────────────────

def aba_analise9(dfs: dict, filtros: dict):
    st.header("A9 · Espaços de Leitura × Rendimento Escolar")
    st.markdown(
        "**Objetivo:** Analisar a correlação entre a presença de biblioteca/sala de leitura "
        "e o IDEB — identificando o **Paradoxo de Simpson** quando todas as etapas são agregadas."
    )

    df = aplicar_filtros(dfs.get("a9", pd.DataFrame()), filtros)
    if df.empty:
        st.warning("Dataset não disponível.")
        return

    if "TAXA_APROVACAO" not in df.columns or df["TAXA_APROVACAO"].notna().sum() < 10:
        st.warning("TAXA_APROVACAO não disponível neste dataset.")
        return
    df_v = df.dropna(subset=["TAXA_APROVACAO"]).copy()
    # Converter escala 0–1 → 0–100 se necessário
    if df_v["TAXA_APROVACAO"].mean() < 2.0:
        df_v["TAXA_APROVACAO"] = df_v["TAXA_APROVACAO"] * 100
    df_v["Espaço de Leitura"] = df_v["IN_ESPACO_LEITURA"].map({1: "Com Biblioteca", 0: "Sem Biblioteca"})

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.box(
            df_v, x="Espaço de Leitura", y="TAXA_APROVACAO",
            color="Espaço de Leitura",
            color_discrete_map={
                "Com Biblioteca": "#636EFA",
                "Sem Biblioteca": "#EF553B",
            },
            points=False,
            title="Distribuição da Taxa de Aprovação por Presença de Biblioteca",
            labels={"TAXA_APROVACAO": "Taxa de Aprovação (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        grp = df_v.groupby("Espaço de Leitura")["TAXA_APROVACAO"].agg(
            Mediana="median", Média="mean", N="count"
        ).round(2).reset_index()
        st.dataframe(grp, use_container_width=True, hide_index=True)

    # Paradoxo de Simpson: por dependência
    st.subheader("Paradoxo de Simpson: Resultado por Dependência Administrativa")
    st.markdown(
        "Quando estratificado por dependência (proxy de etapa de ensino), "
        "o resultado se inverte ou desaparece — confirmando o viés de variável de confusão."
    )
    dep_rows = []
    for dep in df_v["DEPENDENCIA"].dropna().unique():
        sub = df_v[df_v["DEPENDENCIA"] == dep]
        for label in ["Com Biblioteca", "Sem Biblioteca"]:
            val = sub[sub["Espaço de Leitura"] == label]["TAXA_APROVACAO"]
            dep_rows.append({
                "Dependência": dep,
                "Espaço de Leitura": label,
                "Aprovação Mediana (%)": round(val.median(), 2) if len(val) else None,
                "N": len(val),
            })
    dep_df = pd.DataFrame(dep_rows).dropna(subset=["Aprovação Mediana (%)"])
    fig2 = px.bar(
        dep_df, x="Dependência", y="Aprovação Mediana (%)",
        color="Espaço de Leitura", barmode="group",
        color_discrete_map={
            "Com Biblioteca": "#636EFA",
            "Sem Biblioteca": "#EF553B",
        },
        text="Aprovação Mediana (%)",
        title="Taxa de Aprovação Mediana por Dependência e Presença de Biblioteca",
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Nota metodológica:** Escolas sem biblioteca tendem a ser de Anos Iniciais "
        "(rede municipal/pequeno porte), etapa com taxas de aprovação naturalmente mais altas. "
        "A análise desagregada por etapa é necessária para conclusões corretas."
    )
    rodape_analise("Matheus Ferreira de Lima", "Boxplot · Análise de confundimento · Paradoxo de Simpson")


# ── App principal ─────────────────────────────────────────────────────────────

def main():
    dados_dir = dados_dir_from_args()
    dfs = carregar(dados_dir)

    # Header
    st.title("🏫 Infraestrutura Escolar e Rendimento Educacional — Alagoas 2023")
    st.markdown(
        "Dashboard baseado nos **Microdados do Censo Escolar 2023** e nos **Resultados do IDEB 2023** (INEP).  \n"        "Análises: **Guilherme Henrique Costa Lima · Samuel Cassiano dos Santos · Matheus Ferreira de Lima**"
    )
    st.divider()

    filtros = sidebar(dfs)

    abas = st.tabs([
        "📊 Visão Geral",
        "A1 · Computadores",
        "A2 · Banda Larga",
        "A3 · Equip. AV",
        "A4 · Infra × IDEB",
        "A5 · Aprovação",
        "A6 · Vulnerabilidade",
        "A7 · Inclusão Digital",
        "A8 · Saneamento",
        "A9 · Leitura",
    ])

    with abas[0]:
        aba_visao_geral(dfs, filtros)
    with abas[1]:
        aba_analise1(dfs, filtros)
    with abas[2]:
        aba_analise2(dfs, filtros)
    with abas[3]:
        aba_analise3(dfs, filtros)
    with abas[4]:
        aba_analise4(dfs, filtros)
    with abas[5]:
        aba_analise5(dfs, filtros)
    with abas[6]:
        aba_analise6(dfs, filtros)
    with abas[7]:
        aba_analise7(dfs, filtros)
    with abas[8]:
        aba_analise8(dfs, filtros)
    with abas[9]:
        aba_analise9(dfs, filtros)

if __name__ == "__main__":
    main()
else:
    # Streamlit executa o módulo como script, não como __main__
    main()
