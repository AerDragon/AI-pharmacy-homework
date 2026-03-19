"""Однокомандный запуск полного пайплайна (из корня репозитория)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline_core import run_full_pipeline


def main():
    p = argparse.ArgumentParser(description="DrugGEN → очистка → скоринг → топ-10 EGFR")
    p.add_argument("--sample-num", type=int, default=3000)
    p.add_argument(
        "--submodel",
        default="DrugGEN",
        choices=["DrugGEN", "NoTarget"],
        help="Должен совпадать с именем чекпоинта (DrugGEN-G.ckpt / NoTarget-G.ckpt)",
    )
    p.add_argument(
        "--inference-model",
        default="experiments/models/DrugGEN-akt1",
        help="Каталог относительно DrugGEN/ с файлом {submodel}-G.ckpt",
    )
    p.add_argument("--max-atom", type=int, default=60)
    args = p.parse_args()

    res = run_full_pipeline(
        REPO_ROOT,
        sample_num=args.sample_num,
        submodel=args.submodel,
        inference_model_rel=args.inference_model,
        max_atom=args.max_atom,
    )
    art = REPO_ROOT / "artifacts"
    print("\n=== Итог ===", flush=True)
    print("Мишень: EGFR (пост-hoc: similarity к data/egfr_reference_ligands.smi)", flush=True)
    print(f"Режим генерации: {res.generation_mode}", flush=True)
    for k, v in res.stats.items():
        print(f"  {k}: {v}", flush=True)
    print(f"Артефакты: {art}", flush=True)


if __name__ == "__main__":
    main()
