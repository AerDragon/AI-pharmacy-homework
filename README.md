# AI-pharmacy-homework: мини-пайплайн под EGFR (DrugGEN)

Обертка вокруг генеративной модели **DrugGEN** из каталога `DrugGEN/`: генерация молекул (через `DrugGEN/inference.py`), очистка SMILES, расчёт свойств, medchem-фильтры, приоритизация под **мишень EGFR** (пост-hoc: максимальная Tanimoto-близость к `data/egfr_reference_ligands.smi`), отбор **до 10** кандидатов с учётом разнообразия структур.

## Модель

- **DrugGEN** — GAN с graph transformer для генерации молекул по графовому представлению (см. `DrugGEN/README.md` и статью в репозитории).
- Инференс: `DrugGEN/inference.py` (аргументы `--submodel`, `--inference_model`, `--inf_smiles`, `--train_smiles`, `--train_drug_smiles`, `--sample_num`, `--max_atom`, опционально `--disable_correction`).

## Мишень: EGFR

**EGFR** (рецептор эпидермального фактора роста) — проверенная онкологическая мишень, много известных малых молекул (ТКИ: gefitinib, erlotinib, osimertinib и др.). Референсный набор SMILES для скоринга — файл `data/egfr_reference_ligands.smi` (копируется в `DrugGEN/data/` при запуске инференса как `--train_drug_smiles`).

## Предварительная настройка DrugGEN

Перед запуском пайплайна с **реальной генерацией** через `inference.py` нужны веса модели, энкодеры/декодеры и предобработанные графы (`.pt`). Для этого **в корне проекта** выполните:

```bash
git clone -q https://github.com/HUBioDataLab/DrugGEN.git
cd DrugGEN
chmod +x setup.sh
./setup.sh
```

После успешного `setup.sh` вернитесь в корень учебного репозитория (`cd ..`), если будете запускать `scripts/run_pipeline.py` оттуда.

Если каталог `DrugGEN/` уже есть (например, он добавлен в ваш форк), шаг `git clone` можно пропустить; при необходимости обновите код и всё равно выполните `chmod +x setup.sh` и `./setup.sh` внутри `DrugGEN/`.

> **Примечание:** запуск скрипта — `./setup.sh` (из текущего каталога `DrugGEN`), а не `/.setup.sh`.

## Быстрый запуск (одна команда)

Из **корня репозитория** (рядом с `DrugGEN/` и `scripts/`):

```bash
python scripts/run_pipeline.py --sample-num 3000
```

Параметры:

| Аргумент | По умолчанию | Описание |
|----------|----------------|----------|
| `--sample-num` | 3000 | Число молекул (инференс или размер подвыборки fallback) |
| `--submodel` | `DrugGEN` | `DrugGEN` или `NoTarget` (имя чекпоинта `{submodel}-G.ckpt`) |
| `--inference-model` | `experiments/models/DrugGEN-akt1` | Каталог относительно `DrugGEN/` с весами |
| `--max-atom` | 60 | Должен совпадать с обучением и именем `chembl_test{max_atom}.pt` |

Пример, как в оригинальном `config.py`:

```bash
python scripts/run_pipeline.py \
  --submodel DrugGEN \
  --inference-model experiments/models/DrugGEN-akt1 \
  --sample-num 5000 \
  --max-atom 60
```
## Зависимости

- **Пайплайн скоринга** (корень репозитория): `pandas`, `numpy`, `matplotlib`, `rdkit`, `tabulate` (для красивых таблиц в отчёте). См. `requirements-pipeline.txt`.
- **DrugGEN inference** (полная генерация): `torch`, `torch-geometric` и окружение из `DrugGEN/environment.yml`; данные и веса — скрипт `DrugGEN/setup.sh` (Google Drive).

Если **нет** `DrugGEN/data/*.pt` и чекпоинтов, пайплайн автоматически использует **fallback**: подвыборка SMILES из `DrugGEN/results/generated_molecules/DrugGEN_generated_molecules_AKT1.csv` (в отчёте явно указано `fallback_csv`).

## Артефакты (`artifacts/`)

| Файл | Содержание |
|------|------------|
| `generated_raw.csv` | Сырой пул (`smiles`) |
| `generated_clean.csv` | После канонизации и дедупликации |
| `scored_molecules.csv` | Все дескрипторы и флаги, колонка `passes_filters` |
| `final_candidates.csv` | До 10 лучших (колонки по заданию) |
| `final_candidates.png` | Сетка структур (RDKit Draw) |
| `report.md` | Краткий отчёт на русском |

## Токсичность и алерты

Используется **прокси**, а не полноценный in vivo-прогноз: каталоги **PAINS** (RDKit), при наличии — **BRENK**, плюс эвристики реактивных подструктур (azide, acyl halide, epoxide и т.д.). Подробности и формула итогового скоринга — в `artifacts/report.md` и в `scripts/pipeline_core.py`.

## Структура проекта

```
AI-pharmacy-homework/
├── DrugGEN/                 # исходный DrugGEN + inference.py
├── data/
│   └── egfr_reference_ligands.smi
├── scripts/
│   ├── pipeline_core.py     # логика пайплайна
│   └── run_pipeline.py      # CLI
├── notebooks/
│   └── molecule_discovery_pipeline.ipynb
├── artifacts/               # результаты прогонов
├── config.py                # пример аргументов для inference
├── requirements-pipeline.txt
└── README.md
```
