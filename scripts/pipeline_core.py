from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem import RDConfig

    _sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if _sa_path not in sys.path:
        sys.path.append(_sa_path)
    import sascorer
except Exception:
    sascorer = None


REACTIVE_SMARTS = [
    ("azide", "[N-]=[N+]=[N-]"),
    ("diazo", "C=[N+]=[N-]"),
    ("acyl_halide", "C(=O)[F,Cl,Br,I]"),
    ("epoxide", "C1OC1"),
    ("peroxide", "O-O"),
    ("nitro_aryl", "c[N+](=O)[O-]"),
]


def _compile_reactive_patterns():
    out = []
    for name, smarts in REACTIVE_SMARTS:
        p = Chem.MolFromSmarts(smarts)
        if p is not None:
            out.append((name, p))
    return out


REACTIVE_PATTERNS = _compile_reactive_patterns()


def load_pains_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


def load_brenk_catalog():
    params = FilterCatalog.FilterCatalogParams()
    fc = FilterCatalog.FilterCatalogParams.FilterCatalogs
    brenk = getattr(fc, "BRENK", None)
    if brenk is None:
        return None
    params.AddCatalog(brenk)
    return FilterCatalog.FilterCatalog(params)


PAINS_CATALOG = None
BRENK_CATALOG = None


def _get_pains():
    global PAINS_CATALOG
    if PAINS_CATALOG is None:
        PAINS_CATALOG = load_pains_catalog()
    return PAINS_CATALOG


def _get_brenk():
    global BRENK_CATALOG
    if BRENK_CATALOG is None:
        BRENK_CATALOG = load_brenk_catalog()
    return BRENK_CATALOG


def largest_fragment_mol(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())


def canonicalize_smiles(smi: str) -> Optional[str]:
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    s = str(smi).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    mol = largest_fragment_mol(mol)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def clean_generated_pool(df: pd.DataFrame, smiles_col: str = "smiles") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Канонизация, удаление невалидных и дубликатов."""
    stats: Dict[str, Any] = {}
    if smiles_col not in df.columns:
        for alt in ("SMILES", "smiles", "canonical_smiles"):
            if alt in df.columns:
                smiles_col = alt
                break
    raw_n = len(df)
    stats["rows_input"] = raw_n
    canon = []
    for s in df[smiles_col].astype(str):
        canon.append(canonicalize_smiles(s))
    out = df.copy()
    out["canonical_smiles"] = canon
    out = out[out["canonical_smiles"].notna()].drop_duplicates(subset=["canonical_smiles"])
    stats["valid_unique"] = len(out)
    stats["invalid_or_duplicate_dropped"] = raw_n - len(out)
    return out[["canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"}), stats


def morgan_fp(mol: Chem.Mol, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)


def max_tanimoto_to_references(mol: Chem.Mol, ref_fps) -> float:
    if mol is None or not ref_fps:
        return 0.0
    fp = morgan_fp(mol)
    best = 0.0
    for rfp in ref_fps:
        best = max(best, DataStructs.TanimotoSimilarity(fp, rfp))
    return float(best)


def load_reference_fps(smi_path: Path, n_bits: int = 2048) -> List:
    fps = []
    text = smi_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    for line in text:
        line = line.strip()
        if not line or line.upper() == "SMILES":
            continue
        smi = line.split()[0] if line.split() else line
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps.append(morgan_fp(m, n_bits))
    return fps


def pains_hit(mol: Chem.Mol) -> bool:
    cat = _get_pains()
    return cat.GetFirstMatch(mol) is not None


def brenk_hit(mol: Chem.Mol) -> bool:
    cat = _get_brenk()
    if cat is None:
        return False
    return cat.GetFirstMatch(mol) is not None


def reactive_matches(mol: Chem.Mol) -> List[str]:
    hits = []
    for name, pat in REACTIVE_PATTERNS:
        if mol.HasSubstructMatch(pat):
            hits.append(name)
    return hits


def compute_row(mol: Chem.Mol, ref_fps) -> Dict[str, Any]:
    pains = pains_hit(mol)
    brenk = brenk_hit(mol)
    reactive = reactive_matches(mol)
    structural_alert = brenk or len(reactive) > 0
    tox_proxy_count = int(pains) + int(brenk) + len(reactive)
    tox_flag = 1 if tox_proxy_count > 0 else 0

    sa = float(sascorer.calculateScore(mol)) if sascorer is not None else float("nan")
    qed_v = float(QED.qed(mol))
    mw = float(Descriptors.MolWt(mol))
    clogp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    rings = int(mol.GetRingInfo().NumRings())
    fsp3 = float(CalcFractionCSP3(mol))
    charge = int(Chem.GetFormalCharge(mol))
    target_rel = max_tanimoto_to_references(mol, ref_fps)

    return {
        "qed": qed_v,
        "sa_score": sa,
        "mw": mw,
        "clogp": clogp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot,
        "ring_count": rings,
        "fraction_csp3": fsp3,
        "formal_charge": charge,
        "pains_alert": int(pains),
        "structural_alert": int(structural_alert),
        "tox_flag_or_score": tox_proxy_count,
        "tox_flag": tox_flag,
        "target_relevance_score": target_rel,
        "reactive_tags": ";".join(reactive) if reactive else "",
    }


def score_dataframe(df: pd.DataFrame, ref_fps, smiles_col: str = "smiles") -> pd.DataFrame:
    rows = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        r = compute_row(mol, ref_fps)
        r["smiles"] = smi
        rows.append(r)
    return pd.DataFrame(rows)


FILTER_DEFAULTS = {
    "qed_min": 0.5,
    "sa_max": 6.0,
    "mw_min": 200,
    "mw_max": 550,
    "clogp_min": -0.5,
    "clogp_max": 5.5,
    "hbd_max": 5,
    "hba_max": 10,
    "tpsa_max": 140,
    "rot_max": 10,
}


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    f = FILTER_DEFAULTS
    m = (
        (df["qed"] >= f["qed_min"])
        & (df["sa_score"] <= f["sa_max"])
        & (df["mw"] >= f["mw_min"])
        & (df["mw"] <= f["mw_max"])
        & (df["clogp"] >= f["clogp_min"])
        & (df["clogp"] <= f["clogp_max"])
        & (df["hbd"] <= f["hbd_max"])
        & (df["hba"] <= f["hba_max"])
        & (df["tpsa"] <= f["tpsa_max"])
        & (df["rotatable_bonds"] <= f["rot_max"])
        & (df["pains_alert"] == 0)
        & (df["structural_alert"] == 0)
        & (df["tox_flag"] == 0)
    )
    return df.loc[m].copy()


def composite_score(row: pd.Series) -> float:
    """Прозрачная линейная комбинация (после жёстких фильтров по токс-прокси)."""
    qed = row["qed"]
    sa = row["sa_score"]
    sa_norm = max(0.0, 1.0 - (sa / 10.0)) if not np.isnan(sa) else 0.5
    targ = row["target_relevance_score"]
    rot_pen = min(1.0, row["rotatable_bonds"] / 15.0)
    return float(0.28 * qed + 0.22 * sa_norm + 0.42 * targ + 0.08 * (1.0 - rot_pen))


def greedy_diverse_top_k(
    df: pd.DataFrame,
    k: int = 10,
    pool_mult: int = 8,
    fp_bits: int = 2048,
    max_tanimoto_sim: float = 0.82,
) -> pd.DataFrame:
    """Топ по final_score с жадным отбором: Tanimoto similarity с уже выбранными ≤ порога."""
    if df.empty:
        return df
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    pool_n = min(len(df), max(k * pool_mult, k))
    pool = df.iloc[:pool_n].copy()
    fps: List = []
    for smi in pool["canonical_smiles"]:
        m = Chem.MolFromSmiles(smi) if smi else None
        fps.append(morgan_fp(m, fp_bits) if m is not None else None)

    selected_idx: List[int] = []
    for i in range(len(pool)):
        if len(selected_idx) >= k:
            break
        if fps[i] is None:
            continue
        if not selected_idx:
            selected_idx.append(i)
            continue
        sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in selected_idx if fps[j] is not None]
        if sims and max(sims) > max_tanimoto_sim:
            continue
        selected_idx.append(i)

    for i in range(len(pool)):
        if len(selected_idx) >= k:
            break
        if i not in selected_idx and fps[i] is not None:
            selected_idx.append(i)

    selected_idx = selected_idx[:k]
    out = pool.iloc[selected_idx].copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def russian_short_comment(row: pd.Series) -> str:
    parts = []
    if row["target_relevance_score"] >= 0.35:
        parts.append("высокая близость к референсным лигандам EGFR")
    elif row["target_relevance_score"] >= 0.22:
        parts.append("умеренная близость к референсным лигандам EGFR")
    else:
        parts.append("относительно низкая близость к референсу (остальные метрики сильнее)")

    if row["qed"] >= 0.65:
        parts.append("хороший QED")
    if row["sa_score"] <= 4.0:
        parts.append("низкая SA, проще синтез")
    if row["rotatable_bonds"] <= 6:
        parts.append("мало ротируемых связей")
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "." if text else ""


def draw_candidates_grid(smiles_list: List[str], out_png: Path, mols_per_row: int = 5):
    from rdkit.Chem import Draw

    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
    if not mols:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        molsPerRow=mols_per_row,
        subImgSize=(320, 280),
        legends=[f"#{i+1}" for i in range(len(mols))],
    )
    # В части сборок RDKit MolsToGridImage возвращает не PIL.Image (нет .save).
    # Надёжный путь — запросить PNG как bytes (returnPNG=True, где поддерживается).
    try:
        png = Draw.MolsToGridImage(mols, returnPNG=True, **kwargs)
        if isinstance(png, (bytes, bytearray)):
            out_png.write_bytes(png)
            return
    except TypeError:
        pass
    img = Draw.MolsToGridImage(mols, **kwargs)
    if isinstance(img, (bytes, bytearray)):
        out_png.write_bytes(img)
        return
    if hasattr(img, "save"):
        img.save(str(out_png))
        return
    try:
        from PIL import Image as PILImage
        if isinstance(img, PILImage.Image):
            img.save(str(out_png))
            return
    except ImportError:
        pass
    raise TypeError(
        f"MolsToGridImage вернул тип {type(img)!r} без .save(); "
        "обновите RDKit или установите Pillow."
    )


def resolve_checkpoint_dir(
    druggen_root: Path, submodel: str, inference_model_rel: str
) -> Optional[Path]:
    """Папка с весами: ожидается файл {submodel}-G.ckpt."""
    ckpt = druggen_root / inference_model_rel / f"{submodel}-G.ckpt"
    if ckpt.is_file():
        return ckpt.parent
    for folder in (druggen_root / "experiments" / "models").glob("*"):
        if not folder.is_dir():
            continue
        cand = folder / f"{submodel}-G.ckpt"
        if cand.is_file():
            return folder
    return None


def inference_ready(druggen_root: Path, max_atom: int, inf_smiles_basename: str = "chembl_test") -> bool:
    """Нужен конкретный .pt для инференса (как в DruggenDataset) и любой чекпоинт генератора."""
    pt = druggen_root / "data" / f"{inf_smiles_basename}{max_atom}.pt"
    if not pt.is_file():
        return False
    return any((druggen_root / "experiments" / "models").rglob("*-G.ckpt"))


def run_inference_subprocess(
    druggen_root: Path,
    submodel: str,
    inference_model_dir: str,
    sample_num: int,
    max_atom: int,
    inf_smiles: str,
    train_smiles: str,
    train_drug_smiles: str,
    disable_correction: bool = True,
) -> Path:
    """
    Запускает DrugGEN inference.py из каталога druggen_root.
    Возвращает путь к CSV с SMILES (inference_drugs.csv).
    """
    cmd = [
        sys.executable,
        "inference.py",
        "--submodel",
        submodel,
        "--inference_model",
        inference_model_dir,
        "--inf_smiles",
        inf_smiles,
        "--train_smiles",
        train_smiles,
        "--train_drug_smiles",
        train_drug_smiles,
        "--sample_num",
        str(sample_num),
        "--max_atom",
        str(max_atom),
        "--set_seed",
        "--seed",
        "42",
    ]
    if disable_correction:
        cmd.append("--disable_correction")
    print("Запуск:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(druggen_root), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:] if r.stdout else "", r.stderr[-2000:] if r.stderr else "", flush=True)
        raise RuntimeError(f"inference.py завершился с кодом {r.returncode}")
    out_csv = druggen_root / "experiments" / "inference" / submodel / "inference_drugs.csv"
    if not out_csv.is_file():
        raise FileNotFoundError(f"Не найден выход: {out_csv}")
    return out_csv


def copy_raw_from_inference_csv(src_csv: Path, dst_csv: Path):
    df = pd.read_csv(src_csv)
    col = "SMILES" if "SMILES" in df.columns else "smiles"
    out = pd.DataFrame({"smiles": df[col].astype(str)})
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_csv, index=False)


def copy_raw_from_fallback(fallback_csv: Path, dst_csv: Path, n: int, seed: int = 42):
    df = pd.read_csv(fallback_csv)
    col = "SMILES" if "SMILES" in df.columns else "smiles"
    take = df[[col]].rename(columns={col: "smiles"})
    take = take.dropna().sample(n=min(n, len(take)), random_state=seed)
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    take.to_csv(dst_csv, index=False)


@dataclass
class PipelineResult:
    generation_mode: str
    stats: Dict[str, Any] = field(default_factory=dict)


def run_full_pipeline(
    repo_root: Path,
    sample_num: int = 3000,
    submodel: str = "DrugGEN",
    inference_model_rel: str = "experiments/models/DrugGEN-akt1",
    max_atom: int = 60,
    fallback_csv: Optional[Path] = None,
) -> PipelineResult:
    druggen = repo_root / "DrugGEN"
    artifacts = repo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    egfr_smi = repo_root / "data" / "egfr_reference_ligands.smi"
    ref_fps = load_reference_fps(egfr_smi)

    raw_path = artifacts / "generated_raw.csv"
    clean_path = artifacts / "generated_clean.csv"
    scored_path = artifacts / "scored_molecules.csv"
    final_path = artifacts / "final_candidates.csv"
    png_path = artifacts / "final_candidates.png"
    report_path = artifacts / "report.md"

    stats: Dict[str, Any] = {}
    generation_mode = "fallback_csv"
    train_drug_rel = ensure_egfr_smi_in_druggen_data(egfr_smi, druggen)
    ckpt_ok = resolve_checkpoint_dir(druggen, submodel, inference_model_rel) is not None
    ready = inference_ready(druggen, max_atom)

    if ready and ckpt_ok:
        try:
            run_inference_subprocess(
                druggen,
                submodel=submodel,
                inference_model_dir=inference_model_rel,
                sample_num=sample_num,
                max_atom=max_atom,
                inf_smiles="data/chembl_test.smi",
                train_smiles="data/chembl_train.smi",
                train_drug_smiles=train_drug_rel,
            )
            inf_out = druggen / "experiments" / "inference" / submodel / "inference_drugs.csv"
            copy_raw_from_inference_csv(inf_out, raw_path)
            generation_mode = "inference_py"
        except Exception as e:  # noqa: BLE001
            print("Inference не удался, используем fallback CSV:", e, flush=True)
            generation_mode = "fallback_csv"

    if generation_mode == "fallback_csv":
        fb = fallback_csv or (
            druggen / "results" / "generated_molecules" / "DrugGEN_generated_molecules_AKT1.csv"
        )
        if not fb.is_file():
            raise FileNotFoundError(
                f"Нет весов/.pt для inference и нет fallback CSV: {fb}\n"
                "Выполните DrugGEN/setup.sh или положите чекпоинты."
            )
        copy_raw_from_fallback(fb, raw_path, n=sample_num)

    df_raw = pd.read_csv(raw_path)
    stats["generated_raw_count"] = len(df_raw)

    df_clean, cstats = clean_generated_pool(df_raw)
    df_clean.to_csv(clean_path, index=False)
    stats.update(cstats)
    stats["clean_unique_count"] = len(df_clean)

    scored = score_dataframe(df_clean, ref_fps)
    scored["passes_filters"] = False
    filtered = apply_filters(scored)
    scored.loc[filtered.index, "passes_filters"] = True
    filtered = filtered.copy()
    filtered["passes_filters"] = True
    scored.to_csv(scored_path, index=False)
    stats["scored_count"] = len(scored)
    stats["passed_filters_count"] = len(filtered)

    if filtered.empty:
        # смягчить только SA / QED слегка для отчёта, если всё отвалилось
        relaxed = scored[(scored["pains_alert"] == 0) & (scored["tox_flag"] == 0)].copy()
        relaxed = relaxed[(relaxed["qed"] >= 0.35) & (relaxed["sa_score"] <= 7.0)]
        filtered = relaxed.head(200)
        stats["note"] = "Жёсткие фильтры дали 0 молекул; использован ослабленный набор для демонстрации."

    filtered = filtered.copy()
    filtered["canonical_smiles"] = [canonicalize_smiles(s) for s in filtered["smiles"]]
    filtered["final_score"] = filtered.apply(composite_score, axis=1)
    final_df = greedy_diverse_top_k(filtered, k=10)
    final_df["canonical_smiles"] = final_df["canonical_smiles"].fillna(final_df["smiles"])
    final_df["short_comment"] = final_df.apply(russian_short_comment, axis=1)

    cols = [
        "rank",
        "smiles",
        "canonical_smiles",
        "qed",
        "sa_score",
        "mw",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "ring_count",
        "fraction_csp3",
        "formal_charge",
        "pains_alert",
        "structural_alert",
        "tox_flag_or_score",
        "target_relevance_score",
        "final_score",
        "short_comment",
    ]
    for c in cols:
        if c not in final_df.columns and c != "rank":
            if c == "canonical_smiles":
                final_df[c] = final_df["smiles"]
            else:
                final_df[c] = np.nan
    final_df[cols].to_csv(final_path, index=False)
    draw_candidates_grid(final_df["canonical_smiles"].tolist(), png_path)

    write_report_md(
        report_path,
        generation_mode=generation_mode,
        stats=stats,
        final_df=final_df,
        submodel=submodel,
    )
    return PipelineResult(generation_mode=generation_mode, stats=stats)


def write_report_md(
    path: Path,
    generation_mode: str,
    stats: Dict[str, Any],
    final_df: pd.DataFrame,
    submodel: str,
):
    brenk = _get_brenk() is not None
    lines = [
        "# Отчёт: мини-пайплайн de novo под мишень EGFR",
        "",
        "## 1. Генеративная модель и запуск",
        "",
        "Использована модель **DrugGEN** (Graph Transformer + GAN) из каталога `DrugGEN/`, скрипт `DrugGEN/inference.py`. "
        "Вариант чекпоинта задаётся флагом `--submodel` (`DrugGEN` или `NoTarget`) и путём `--inference_model`.",
        "",
        f"**Режим генерации в этом прогоне:** `{generation_mode}`.",
        "- `inference_py` — молекулы получены свежим запуском `inference.py`.",
        "- `fallback_csv` — чекпоинты и/или предобработанные графы (`.pt`) отсутствуют; взяты SMILES из публикуемого набора "
        "`DrugGEN/results/generated_molecules/` (демонстрация пост-обработки без локального GPU/весов).",
        "",
        "**Важно:** на этапе инференса модель **не принимает target id** и не условится явным вектором мишени. "
        "У DrugGEN «таргетность» заложена в **обучении** дискриминатора на наборе ингибиторов конкретного белка; "
        "при выборе другого `.ckpt` распределение сэмплов меняется. Для мишени **EGFR** в открытом репозитории нет готового чекпоинта, "
        "поэтому специфичность к EGFR достигается **пост-hoc**: максимальная Tanimoto-близость (Morgan, r=2) к набору известных лигандов EGFR (`data/egfr_reference_ligands.smi`).",
        "",
        "## 2. Выбор мишени",
        "",
        "**EGFR** (рецептор эпидермального фактора роста) — клинически важная мишень в онкологии (некоторые ТКИ: gefitinib, erlotinib, osimertinib). "
        "Много известных малых молекул-ингибиторов, хорошо изученная медицинская химия — удобно для учебного baseline по сходству с референсными лигандами.",
        "",
        "## 3. Target-conditioned generation",
        "",
        "**Прямой conditioning по ID мишени на инференсе не поддерживается** (нет аргументов CLI для target embedding / prompt). "
        "Условность возможна косвенно через использование весов, обученных на ингибиторах конкретного белка (AKT1, CDK2 в оригинальной статье).",
        "",
        "## 4. Фильтрация",
        "",
        "После канонизации и дедупликации применялись пороги (medchem baseline):",
        f"- QED ≥ {FILTER_DEFAULTS['qed_min']}",
        f"- SA score ≤ {FILTER_DEFAULTS['sa_max']}",
        f"- {FILTER_DEFAULTS['mw_min']} ≤ MW ≤ {FILTER_DEFAULTS['mw_max']}",
        f"- cLogP ∈ [{FILTER_DEFAULTS['clogp_min']}, {FILTER_DEFAULTS['clogp_max']}]",
        f"- HBD ≤ {FILTER_DEFAULTS['hbd_max']}, HBA ≤ {FILTER_DEFAULTS['hba_max']}",
        f"- TPSA ≤ {FILTER_DEFAULTS['tpsa_max']}, ротируемые связи ≤ {FILTER_DEFAULTS['rot_max']}",
        "- PAINS (каталоги A+B+C RDKit): совпадений нет",
        f"- Structural alerts: BRENK ({'да, RDKit' if brenk else 'недоступен в этой сборке RDKit'}) + эвристики реактивных подструктур (azide, acyl_halide, epoxide, …)",
        "- Токс-прокси: суммарный счётчик срабатываний PAINS + BRENK + реактивных SMARTS; кандидаты с счётчиком > 0 отсекаются",
        "",
        "Для ТКИ EGFR выбранные окна по MW/cLogP/TPSA согласуются с типичными оральными ингибиторами; при необходимости их можно сузить под конкретный химотип.",
        "",
        "## 5. Свойства",
        "",
        "QED (RDKit), SA score (`RDKit Contrib sascorer`), cLogP (Crippen), MW, TPSA, HBD/HBA (Lipinski), ротируемые связи, число колец, FractionCSP3, формальный заряд, "
        "флаги PAINS/structural, прокси токсичности, `target_relevance_score` (max Tanimoto к референсу EGFR), итоговый `final_score` (взвешенная сумма QED, SA, таргет-близости, штраф за гибкость).",
        "",
        "## 6. Токсичность",
        "",
        "**Это не полноценное предсказание in vivo-токсичности.** Используется **best-effort прокси**: PAINS + (если доступно) фильтр BRENK + набор простых реактивных/проблемных подструктур. "
        "Результат — целочисленный `tox_flag_or_score` и жёсткий отсев при ненулевом флаге после фильтрации.",
        "",
        "## 7. Счётчики",
        "",
        f"- Строк в сыром CSV: **{stats.get('generated_raw_count', '—')}**",
        f"- Валидных уникальных после очистки: **{stats.get('clean_unique_count', '—')}**",
        f"- Проскорено: **{stats.get('scored_count', '—')}**",
        f"- Прошло фильтры: **{stats.get('passed_filters_count', '—')}**",
        f"- В топ-10 (после diversity selection): **{len(final_df)}**",
        "",
    ]
    if stats.get("note"):
        lines.append(f"_Примечание:_ {stats['note']}")
        lines.append("")
    lines.append("## 8. Таблица финальных кандидатов")
    lines.append("")
    report_cols = [
        "rank",
        "smiles",
        "canonical_smiles",
        "qed",
        "sa_score",
        "mw",
        "clogp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "ring_count",
        "fraction_csp3",
        "formal_charge",
        "pains_alert",
        "structural_alert",
        "tox_flag_or_score",
        "target_relevance_score",
        "final_score",
        "short_comment",
    ]
    rep = final_df[[c for c in report_cols if c in final_df.columns]]
    try:
        lines.append(rep.to_markdown(index=False))
    except (ImportError, ValueError):
        lines.append("```")
        lines.append(rep.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## 9. Комментарии")
    lines.append("")
    for _, r in final_df.iterrows():
        lines.append(f"- **Ранг {int(r['rank'])}:** {r.get('short_comment', '')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_egfr_smi_in_druggen_data(egfr_repo: Path, druggen_root: Path) -> str:
    """Копирует референсные SMILES в DrugGEN/data/ (inference запускается из DrugGEN/)."""
    import shutil

    alt = druggen_root / "data" / "egfr_reference_ligands.smi"
    alt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(egfr_repo.resolve(), alt)
    return "data/egfr_reference_ligands.smi"
