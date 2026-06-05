# PokerBench GTO Fine-Tuning — Local (VSCode)

Port of the original Colab notebook for CIS 4270/5270 Trustworthy ML. Runs SFT
and DPO fine-tuning of `gpt-4.1-nano` on PokerBench via Azure OpenAI. RFT is
preserved but gated off (course-wide hold).

## One-time setup

```bash
# 1. Python env (3.10+ recommended)
python -m venv .venv
.venv/Scripts/activate     # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Azure auth — required for deploying and cleaning up endpoints
az login
```

## Running the notebook

Open `cis5270_pokerbench_local.ipynb` in VSCode (or `jupyter lab`). **Section 0
is the only cell you edit between runs** — everything else is flag-driven.

### Default flags → "deploy existing models and evaluate"

```
REBUILD_DATA = False     # use cached data
TRAIN_SFT    = False     # use saved model ID in state
TRAIN_DPO    = False
DEPLOY_SFT   = True      # deploy existing SFT
DEPLOY_DPO   = True      # deploy existing DPO
RUN_FULL_EVAL = True     # evaluate on N_TEST=500
DELETE_DEPLOYMENTS_AT_END = True
```

Running top-to-bottom with these flags: loads cached data → deploys existing
SFT + DPO → evaluates on 500 test rows → produces plots and CSV → cleans up
deployments. **This is the workflow for the draft report.**

### To retrain

Flip `TRAIN_SFT` / `TRAIN_DPO` to `True`. If the training JSONLs aren't on
disk, also flip `REBUILD_DATA=True` and `REUPLOAD_SFT`/`REUPLOAD_DPO=True`.

### To run the hyperparameter sweep

Flip `DO_SWEEP=True`. The default grid is 2×2 per algorithm (4 SFT + 4 DPO
jobs, all with `validation_file` set). Section 8.2 selects the lowest-val-loss
job per algorithm and overwrites the active model ID — so Section 9 still
deploys only one model per algo (what the course policy requires). Extend the
grids in Section 7.4 if you want 3×3.

## Important notes

- **Deployments auto-terminate after 4 hours** on this course's Azure setup.
  Deploy → eval → cleanup must all finish inside one 4h window. The notebook
  logs time-sensitive warnings in Section 9.
- **Deployment creation requires TA coordination** per course policy. If
  `begin_create_or_update` fails with a quota/permission error, ping the TAs
  before retrying.
- **State persists** across kernel restarts in `finetuned_state.json` —
  model IDs, file IDs, prompt versions, deployment names, baseline accuracy,
  results.
- **Caches** live in `./cache/` (~1.6 GB for PokerBench). Delete them to force
  re-download.
- **DPO prompt-version fix:** the existing DPO model was trained under
  `v1_long_system`, but current run defaults to `v2_no_system`. Evaluation
  auto-picks the right prompt per-model via `build_messages(..., prompt_version=...)`
  in Section 2.1.

## Files produced during a run

```
results_comparison.png      # bar chart — binary accuracy per model
detailed_metrics.csv        # Class A/B/C, illegal, partial credit
confusion_matrices.png      # action-category confusion per model
training_curves.png         # training loss from Azure result files
finetuned_state.json        # persisted state
cache/                      # pickled PokerBench frames + unified rows
sft_train.jsonl, sft_val.jsonl      # (if REBUILD_DATA=True)
dpo_train.jsonl, dpo_val.jsonl      # (if REBUILD_DATA=True)
rft_train.jsonl, rft_val.jsonl      # (if REBUILD_DATA and RUN_RFT)
```

## Troubleshooting

- **Baseline eval fails:** the base `gpt-4.1-nano` deployment may not exist on
  your account. Either deploy it manually first (no fine-tuning) or set
  `BASELINE_OVERRIDE = <float>` in Section 0 to skip with a cached number.
- **Safety filter rejections on training:** the sanitizer in Section 4.4
  replaces "all-in" → "shove" and strips role-prime wording. If a new run
  still fails, check the training file for other trigger phrases.
- **`az` not on PATH:** install the Azure CLI. Data prep + training don't need
  it; deploy + cleanup do.
