"""
pipeline.py — Tratamento de dados para o Dashboard de Infraestrutura Escolar de Alagoas
Fonte: Microdados do Censo Escolar 2023 (INEP) + IDEB 2023 (INEP)

Uso:
    python pipeline.py --censo microdados_ed_basica_2023.csv \
                       --ideb_medio divulgacao_ensino_medio_escolas_2023.xlsx \
                       --ideb_iniciais divulgacao_anos_iniciais_escolas_2023.xlsx \
                       --ideb_finais divulgacao_anos_finais_escolas_2023.xlsx \
                       --output dados_tratados/

Saída: 9 arquivos parquet + censo_al.parquet + ideb_al.parquet
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Mapeamentos ──────────────────────────────────────────────────────────────

DEPENDENCIA_MAP = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
LOCALIZACAO_MAP = {1: "Urbana", 2: "Rural"}

COLUNAS_CENSO = [
    # Identificação
    "CO_ENTIDADE", "NO_ENTIDADE", "SG_UF", "NO_MUNICIPIO", "CO_MUNICIPIO",
    "TP_DEPENDENCIA", "TP_LOCALIZACAO",
    # Infraestrutura tecnológica
    "IN_ENERGIA_REDE_PUBLICA", "IN_ENERGIA_INEXISTENTE",
    "IN_INTERNET", "IN_BANDA_LARGA", "IN_INTERNET_ALUNOS",
    "IN_ACESSO_INTERNET_COMPUTADOR",
    "IN_LABORATORIO_INFORMATICA", "IN_LABORATORIO_CIENCIAS",
    "IN_EQUIP_MULTIMIDIA", "IN_EQUIP_LOUSA_DIGITAL",
    # Equipamentos
    "QT_DESKTOP_ALUNO", "QT_COMP_PORTATIL_ALUNO", "QT_TABLET_ALUNO",
    # Infraestrutura física
    "IN_AGUA_POTAVEL", "IN_AGUA_INEXISTENTE",
    "IN_ESGOTO_REDE_PUBLICA", "IN_ESGOTO_FOSSA_SEPTICA",
    "IN_ESGOTO_FOSSA_COMUM", "IN_ESGOTO_FOSSA", "IN_ESGOTO_INEXISTENTE",
    "IN_QUADRA_ESPORTES", "IN_QUADRA_ESPORTES_COBERTA", "IN_QUADRA_ESPORTES_DESCOBERTA",
    "IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_SALA_LEITURA",
    # Matrículas
    "QT_MAT_BAS", "QT_MAT_FUND_AI", "QT_MAT_FUND_AF", "QT_MAT_MED",
]


# ── Carregamento ─────────────────────────────────────────────────────────────

def carregar_censo(path: str) -> pd.DataFrame:
    """Carrega o CSV do Censo Escolar filtrando para Alagoas e colunas relevantes."""
    print(f"[1/4] Carregando Censo Escolar: {path}")
    # Lê somente as colunas necessárias para economizar memória
    df = pd.read_csv(
        path,
        sep=";",
        encoding="latin-1",
        usecols=lambda c: c in COLUNAS_CENSO + ["SG_UF"],
        low_memory=False,
    )
    # Filtro Alagoas
    df = df[df["SG_UF"] == "AL"].copy()
    print(f"    → {len(df):,} escolas em AL encontradas.")
    return df


def _limpar_ideb_sheet(df_raw: pd.DataFrame, etapa: str) -> pd.DataFrame:
    """
    Parseia uma planilha do IDEB (formato INEP com cabeçalho multi-linha).
    Retorna DataFrame com colunas: CO_ENTIDADE, IDEB_2023, APROVACAO_2023, etapa.
    """
    # O INEP usa as 4 primeiras linhas como cabeçalho descritivo.
    # A linha de índice 3 contém os nomes de colunas reais (SG_UF, CO_MUNICIPIO…).
    # As colunas sem nome ficam como Unnamed: N.
    # Estratégia: localizar a linha que contém "ID_ESCOLA" ou "CO_ENTIDADE".

    # Encontrar linha-cabeçalho
    header_row = None
    for i, row in df_raw.iterrows():
        vals = row.astype(str).str.upper().tolist()
        if any("ID_ESCOLA" in v or "CO_ENTIDADE" in v for v in vals):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Não foi possível localizar o cabeçalho na planilha {etapa}.")

    # Reconstruir com cabeçalho correto
    new_header = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = new_header
    df = df.reset_index(drop=True)

    # Renomear colunas chave
    rename = {}
    for col in df.columns:
        col_str = str(col).strip().upper()
        if "ID_ESCOLA" in col_str or "CO_ENTIDADE" in col_str:
            rename[col] = "CO_ENTIDADE"
        elif col_str == "VL_OBSERVADO_2023":
            rename[col] = "IDEB_2023"
        elif col_str in ("VL_APROVACAO_2023_SI_4", "VL_APROVACAO_2023_TOTAL"):
            rename[col] = "APROVACAO_2023_RAW"
    df = df.rename(columns=rename)

    # Garantir que CO_ENTIDADE existe
    if "CO_ENTIDADE" not in df.columns:
        raise ValueError(f"Coluna CO_ENTIDADE não encontrada na planilha {etapa}. Colunas: {df.columns.tolist()[:10]}")

    # Converter para numérico
    df["CO_ENTIDADE"] = pd.to_numeric(df["CO_ENTIDADE"], errors="coerce")
    if "IDEB_2023" in df.columns:
        df["IDEB_2023"] = pd.to_numeric(df["IDEB_2023"], errors="coerce")
    if "APROVACAO_2023_RAW" in df.columns:
        df["APROVACAO_2023_RAW"] = pd.to_numeric(df["APROVACAO_2023_RAW"], errors="coerce")

    df = df.dropna(subset=["CO_ENTIDADE"])
    df["CO_ENTIDADE"] = df["CO_ENTIDADE"].astype(int)
    df["ETAPA"] = etapa
    return df[["CO_ENTIDADE", "IDEB_2023", "APROVACAO_2023_RAW", "ETAPA"] if "APROVACAO_2023_RAW" in df.columns else ["CO_ENTIDADE", "IDEB_2023", "ETAPA"]]


def carregar_ideb(path_medio: str, path_iniciais: str, path_finais: str) -> pd.DataFrame:
    """Carrega e combina os 3 arquivos IDEB em um único DataFrame."""
    print("[2/4] Carregando bases IDEB...")
    frames = []
    for path, etapa in [
        (path_medio, "Ensino Médio"),
        (path_iniciais, "Anos Iniciais"),
        (path_finais, "Anos Finais"),
    ]:
        if not os.path.exists(path):
            print(f"    ⚠ Arquivo não encontrado: {path} — etapa '{etapa}' ignorada.")
            continue
        xls = pd.ExcelFile(path)
        sheet = xls.sheet_names[0]
        df_raw = pd.read_excel(path, sheet_name=sheet, header=None)
        df_limpo = _limpar_ideb_sheet(df_raw, etapa)
        frames.append(df_limpo)
        print(f"    → {etapa}: {len(df_limpo):,} registros.")

    if not frames:
        raise RuntimeError("Nenhum arquivo IDEB foi carregado.")
    return pd.concat(frames, ignore_index=True)


# ── Tratamento base ──────────────────────────────────────────────────────────

def tratar_censo(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza, tipagem e criação de variáveis derivadas."""
    print("[3/4] Tratando dados do Censo...")

    # Mapeamentos
    df["DEPENDENCIA"] = df["TP_DEPENDENCIA"].map(DEPENDENCIA_MAP)
    df["LOCALIZACAO"] = df["TP_LOCALIZACAO"].map(LOCALIZACAO_MAP)

    # Binários IN_ → 0/1 numérico (tratar NaN como 0)
    cols_in = [c for c in df.columns if c.startswith("IN_")]
    for c in cols_in:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Variáveis de quantidade
    cols_qt = [c for c in df.columns if c.startswith("QT_")]
    for c in cols_qt:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ── Variáveis derivadas ──────────────────────────────────────────────────

    # Total de computadores para alunos
    df["QT_COMP_ALUNO_TOTAL"] = (
        df["QT_DESKTOP_ALUNO"].fillna(0)
        + df["QT_COMP_PORTATIL_ALUNO"].fillna(0)
        + df["QT_TABLET_ALUNO"].fillna(0)
    )

    # Presença de energia (qualquer fonte)
    df["IN_ENERGIA_DISPONIVEL"] = np.where(df["IN_ENERGIA_INEXISTENTE"] == 1, 0, 1)

    # Presença de água potável (já existe IN_AGUA_POTAVEL)
    # Presença de esgoto (qualquer tipo exceto inexistente)
    df["IN_ESGOTO_DISPONIVEL"] = np.where(df["IN_ESGOTO_INEXISTENTE"] == 1, 0, 1)

    # Índice de Infraestrutura Tecnológica (0-3): energia + banda larga + lab informática
    df["IDX_INFRA_TEC"] = (
        df["IN_ENERGIA_DISPONIVEL"]
        + df["IN_BANDA_LARGA"]
        + df["IN_LABORATORIO_CIENCIAS"]
    )

    # Espaço de leitura (biblioteca OU sala de leitura)
    df["IN_ESPACO_LEITURA"] = np.where(
        (df["IN_BIBLIOTECA"] == 1) | (df["IN_BIBLIOTECA_SALA_LEITURA"] == 1) | (df["IN_SALA_LEITURA"] == 1),
        1, 0
    )

    # Porte da escola
    def porte(x):
        if pd.isna(x) or x == 0:
            return "Sem matrícula"
        elif x < 50:
            return "Pequena (<50)"
        elif x < 200:
            return "Média (50-199)"
        elif x < 500:
            return "Grande (200-499)"
        else:
            return "Muito grande (≥500)"

    df["PORTE"] = df["QT_MAT_BAS"].apply(porte)

    print(f"    → {len(df):,} escolas tratadas.")
    return df


def merge_censo_ideb(censo: pd.DataFrame, ideb: pd.DataFrame) -> pd.DataFrame:
    """Merge entre censo e IDEB via CO_ENTIDADE (left join)."""
    print("[4/4] Fazendo merge Censo × IDEB...")
    censo["CO_ENTIDADE"] = pd.to_numeric(censo["CO_ENTIDADE"], errors="coerce").astype("Int64")
    ideb["CO_ENTIDADE"] = ideb["CO_ENTIDADE"].astype("Int64")

    # Pegar o melhor IDEB por escola (maior valor, caso haja múltiplas etapas)
    ideb_agg = ideb.groupby("CO_ENTIDADE").agg(
        IDEB_2023=("IDEB_2023", "max"),
        APROVACAO_2023_RAW=("APROVACAO_2023_RAW", "max") if "APROVACAO_2023_RAW" in ideb.columns else ("IDEB_2023", "max"),
    ).reset_index()

    df = censo.merge(ideb_agg, on="CO_ENTIDADE", how="left")
    n_com_ideb = df["IDEB_2023"].notna().sum()
    print(f"    → {n_com_ideb:,} escolas com IDEB ({n_com_ideb/len(df)*100:.1f}%).")

    # Taxa de aprovação a partir do IDEB bruto (fallback)
    if "APROVACAO_2023_RAW" in df.columns:
        df["TAXA_APROVACAO"] = df["APROVACAO_2023_RAW"]
    else:
        df["TAXA_APROVACAO"] = np.nan

    return df


# ── Geração dos 9 datasets de análise ────────────────────────────────────────

def gerar_analises(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ── ANÁLISE 1 (Guilherme): Computadores por Dependência ─────────────────
    a1 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "QT_DESKTOP_ALUNO", "QT_COMP_PORTATIL_ALUNO", "QT_TABLET_ALUNO",
             "QT_COMP_ALUNO_TOTAL", "QT_MAT_BAS"]].copy()
    a1 = a1[a1["DEPENDENCIA"].notna()]
    # Recorte no 99º percentil para visualização
    p99 = a1["QT_COMP_ALUNO_TOTAL"].quantile(0.99)
    a1["QT_COMP_ALUNO_TOTAL_VIS"] = a1["QT_COMP_ALUNO_TOTAL"].clip(upper=p99)
    a1.to_parquet(os.path.join(output_dir, "analise1_computadores.parquet"), index=False)

    # ── ANÁLISE 2 (Guilherme): Banda Larga × Porte ──────────────────────────
    a2 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "IN_BANDA_LARGA", "IN_INTERNET", "QT_MAT_BAS", "PORTE"]].copy()
    a2["BANDA_LARGA_LABEL"] = a2["IN_BANDA_LARGA"].map({1: "Com Banda Larga", 0: "Sem Banda Larga"})
    a2.to_parquet(os.path.join(output_dir, "analise2_banda_larga.parquet"), index=False)

    # ── ANÁLISE 3 (Guilherme): Projetor + Lousa por Localização ─────────────
    a3 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "IN_EQUIP_MULTIMIDIA", "IN_EQUIP_LOUSA_DIGITAL"]].copy()
    a3 = a3[a3["LOCALIZACAO"].notna()]
    a3.to_parquet(os.path.join(output_dir, "analise3_equipamentos_av.parquet"), index=False)

    # ── ANÁLISE 4 (Samuel): Índice Infraestrutura Tec × IDEB ─────────────────
    a4 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "IDX_INFRA_TEC", "IN_ENERGIA_DISPONIVEL", "IN_BANDA_LARGA",
             "IN_LABORATORIO_CIENCIAS", "IDEB_2023", "QT_MAT_BAS"]].copy()
    a4 = a4[a4["IDEB_2023"].notna() & a4["DEPENDENCIA"].notna()]
    a4.to_parquet(os.path.join(output_dir, "analise4_idxinfra_ideb.parquet"), index=False)

    # ── ANÁLISE 5 (Samuel): Taxa de Aprovação × Dependência × Localização ───
    a5 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "TAXA_APROVACAO", "IDEB_2023", "QT_MAT_BAS"]].copy()
    # Se TAXA_APROVACAO está vazia, usar proxy via IDEB
    a5 = a5[a5["DEPENDENCIA"].notna() & a5["LOCALIZACAO"].notna()]
    # Remover privadas (sem IDEB público consistente)
    a5 = a5[a5["DEPENDENCIA"] != "Privada"]
    a5.to_parquet(os.path.join(output_dir, "analise5_aprovacao_dep_loc.parquet"), index=False)

    # ── ANÁLISE 6 (Samuel): Escolas em Vulnerabilidade Combinada ────────────
    a6 = df.copy()
    mediana_ideb = a6["IDEB_2023"].median()
    a6["IDEB_ABAIXO_MEDIANA"] = (a6["IDEB_2023"] < mediana_ideb).astype(int)
    # Déficits: sem quadra, sem lab ciências, sem banda larga
    a6["N_DEFICITS"] = (
        (1 - a6["IN_QUADRA_ESPORTES"])
        + (1 - a6["IN_LABORATORIO_CIENCIAS"])
        + (1 - a6["IN_BANDA_LARGA"])
    )
    a6["VULNERAVEL"] = ((a6["IDEB_ABAIXO_MEDIANA"] == 1) & (a6["N_DEFICITS"] >= 2)).astype(int)
    a6 = a6[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "IDEB_2023", "IDEB_ABAIXO_MEDIANA", "N_DEFICITS", "VULNERAVEL",
             "IN_QUADRA_ESPORTES", "IN_LABORATORIO_CIENCIAS", "IN_BANDA_LARGA",
             "QT_MAT_BAS"]].copy()
    a6 = a6[a6["DEPENDENCIA"].notna() & a6["LOCALIZACAO"].notna() & a6["IDEB_2023"].notna()]
    a6.to_parquet(os.path.join(output_dir, "analise6_vulnerabilidade.parquet"), index=False)

    # ── ANÁLISE 7 (Matheus): Inclusão Digital — Escolas Estaduais ───────────
    a7 = df[df["DEPENDENCIA"] == "Estadual"][
        ["CO_ENTIDADE", "NO_ENTIDADE", "LOCALIZACAO",
         "IN_INTERNET", "IN_BANDA_LARGA", "IN_LABORATORIO_INFORMATICA",
         "IN_ACESSO_INTERNET_COMPUTADOR"]
    ].copy()
    a7.to_parquet(os.path.join(output_dir, "analise7_inclusao_digital.parquet"), index=False)

    # ── ANÁLISE 8 (Matheus): Saneamento Básico por Zona ─────────────────────
    a8 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "IN_AGUA_POTAVEL", "IN_AGUA_INEXISTENTE",
             "IN_ESGOTO_DISPONIVEL", "IN_ESGOTO_INEXISTENTE",
             "QT_MAT_BAS"]].copy()
    a8 = a8[a8["LOCALIZACAO"].notna()]
    a8.to_parquet(os.path.join(output_dir, "analise8_saneamento.parquet"), index=False)

    # ── ANÁLISE 9 (Matheus): Espaços de Leitura × Rendimento ────────────────
    a9 = df[["CO_ENTIDADE", "NO_ENTIDADE", "DEPENDENCIA", "LOCALIZACAO",
             "IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_SALA_LEITURA",
             "IN_ESPACO_LEITURA", "IDEB_2023", "TAXA_APROVACAO",
             "QT_MAT_BAS", "QT_MAT_FUND_AI", "QT_MAT_FUND_AF", "QT_MAT_MED"]].copy()
    a9 = a9[a9["DEPENDENCIA"].notna()]
    a9.to_parquet(os.path.join(output_dir, "analise9_leitura_rendimento.parquet"), index=False)

    print(f"\n✅ 9 datasets de análise salvos em '{output_dir}'.")
    for i in range(1, 10):
        fname = [f for f in os.listdir(output_dir) if f.startswith(f"analise{i}_")]
        if fname:
            path = os.path.join(output_dir, fname[0])
            rows = len(pd.read_parquet(path))
            print(f"   analise{i}: {rows:,} linhas")


# ── Execução principal ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline de dados — Censo Escolar AL 2023")
    parser.add_argument("--censo", required=True, help="Caminho para microdados_ed_basica_2023.csv")
    parser.add_argument("--ideb_medio", required=True, help="Caminho para divulgacao_ensino_medio_escolas_2023.xlsx")
    parser.add_argument("--ideb_iniciais", required=True, help="Caminho para divulgacao_anos_iniciais_escolas_2023.xlsx")
    parser.add_argument("--ideb_finais", required=True, help="Caminho para divulgacao_anos_finais_escolas_2023.xlsx")
    parser.add_argument("--output", default="dados_tratados", help="Diretório de saída (default: dados_tratados/)")
    args = parser.parse_args()

    # 1. Carregar
    censo_raw = carregar_censo(args.censo)
    ideb_raw = carregar_ideb(args.ideb_medio, args.ideb_iniciais, args.ideb_finais)

    # 2. Tratar
    censo_limpo = tratar_censo(censo_raw)

    # 3. Merge
    df_final = merge_censo_ideb(censo_limpo, ideb_raw)

    # 4. Salvar base completa
    censo_path = os.path.join(args.output, "censo_al.parquet")
    os.makedirs(args.output, exist_ok=True)
    df_final.to_parquet(censo_path, index=False)
    print(f"\n📦 Base completa salva: {censo_path} ({len(df_final):,} escolas)")

    # 5. Gerar 9 datasets
    gerar_analises(df_final, args.output)


if __name__ == "__main__":
    main()
