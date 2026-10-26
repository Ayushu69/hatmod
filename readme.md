# Hinglish Hate-Speech Detection (HINGHATE + MuRIL)

A reproducible pipeline for detecting hateful and hostile comments in **Hinglish** (Hindi-English code-mixed social media text). It audits and cleans the HINGHATE corpus, builds a leakage-free train/val/test split, trains a classical baseline (TF-IDF + Logistic Regression), fine-tunes **MuRIL** (`google/muril-base-cased`) on a 6 GB consumer GPU, and then checks how much the results can be trusted, including a hand audit of the dataset's labels against the dataset's own documented definition of hate.

> **Status: Stages 1-6 complete on HINGHATE (in-dataset). The external test (HASOC) is pending data access.**
> This README documents what was done, why, and what the numbers do and do not mean.

---

## Table of contents

1. [TL;DR](#tldr)
2. [Project status](#project-status)
3. [The dataset (HINGHATE)](#the-dataset-hinghate)
4. [Environment and hardware](#environment-and-hardware)
5. [Repository layout](#repository-layout)
6. [Pipeline, step by step](#pipeline-step-by-step)
7. [Results](#results)
8. [Label audit (100 rows, hand-tagged)](#label-audit-100-rows-hand-tagged)
9. [Verdict, and what is NOT established](#verdict-and-what-is-not-established)
10. [How to reproduce](#how-to-reproduce)
11. [Roadmap](#roadmap)
12. [Data, ethics and licensing](#data-ethics-and-licensing)
13. [Authors, license, citation](#authors-license-citation)

---

## TL;DR

| Question | Answer |
|---|---|
| Task | Binary classification: hate (1) vs non-hate (0) on Hinglish comments, using HINGHATE's labels |
| Data | HINGHATE: 248,129 rows, **136,632 unique** after deduplication, about 8% hate |
| Baseline | TF-IDF (char + word n-grams) + Logistic Regression: **hate-class F1 0.524** |
| Main model | MuRIL fine-tuned. Single model, 3 seeds: **hate F1 0.784 +/- 0.003**. 3-seed probability ensemble: **hate F1 0.802** (95% CI 0.784-0.821), macro-F1 0.893, PR-AUC 0.890 |
| Label audit | Of 100 random test rows labelled "hate", one annotator judged **17% strict group-targeted hate**, 22% personal abuse, 19% political criticism, 42% no visible hostility. The dataset's own guideline says the last three categories are not hate |
| What the model learned | It largely reproduces HINGHATE's labelling as shipped: it flagged about 69% of the rows judged "no visible hostility" and 95% of personal-abuse rows |
| Verdict | HINGHATE is enough to train and benchmark **within HINGHATE**. It is **not** enough to claim general hate-speech detection: there is no external test, and the shipped positive label appears broader than the dataset's documented definition |

**Do not read 0.80 as "solved hate speech detection".** It means "reproduces HINGHATE's labels well on held-out HINGHATE comments".

---

## Project status

| Stage | What | Status |
|---|---|---|
| 1 | Load and audit the raw data | Done |
| 2 | Clean, deduplicate, split, token-length audit | Done |
| 3 | TF-IDF + Logistic Regression baseline | Done |
| 4 | MuRIL fine-tuning, 3 seeds, threshold study, near-duplicate check | Done |
| 5 | Probability ensemble, bootstrap CIs, error analysis, slices | Done |
| 6 | Verdict and explicit list of what is not established | Done |
| - | Hand audit of 100 positive rows | Done (single annotator) |
| - | External test on HASOC | Pending access |

---

## The dataset (HINGHATE)

**HINGHATE** is a large Hinglish hate-speech corpus built from YouTube and Reddit comments, with a binary label. It is distributed through IEEE DataPort and needs a (subscriber) login: <https://ieee-dataport.org/documents/hinghate-one-largest-hinglish-hate-speech-datasets-youtube-and-reddit>

**The data is not included in this repository** (see [Data, ethics and licensing](#data-ethics-and-licensing)). Get it yourself and place it as described in [How to reproduce](#how-to-reproduce).

### File format

A tab-separated file `hinglish_corpus_final.tsv`, UTF-8, with columns:

| Column | Meaning |
|---|---|
| `text` | The comment |
| `label` | `0` = non-hate, `1` = hate |
| `source` | `youtube`, `reddit`, or blank |
| `script` | `roman` (Hindi in English letters), `deva` (Devanagari), `mixed`, or blank |

Read it with `sep="\t", quoting=3, dtype=str, keep_default_na=False`. The `quoting=3` setting stops stray quote characters in scraped text from breaking the parser.

> **Gotcha:** the download came as three TSV files (`hinglish_corpus_final.tsv`, `..._0.tsv`, `..._1.tsv`). They are **byte-identical** (same size, same MD5). `_0` and `_1` are *not* label splits. Use only one file; concatenating them would triple-count every row.

> **No context.** The file has no video ID, no parent comment and no thread information. Comments are labelled and modelled in isolation, so meaning that depends on context is invisible to both the model and a human re-reading the data.

### What the dataset documentation says

The DataPort page links a `Document.pdf` that describes the dataset. The points that matter for this project:

- **Collection:** records were collected through the YouTube and Reddit APIs, and every record is described as having a `source` and a `script` field.
- **Labels were assigned by human annotation** using a written guideline. **Hate requires both gates:** (1) the content targets people as a group or as members of a group, and (2) it attacks that group's worth, humanity, belonging, rights or safety.
- **What the guideline says is not hate** unless both gates are met: political criticism, disagreement, theological arguments, ordinary rudeness, sporting rivalry, personal insults (including an insult aimed at a named politician), and criticism of a policy.
- **Groups considered:** religion, caste, region, language, nationality, ethnicity, tribe, sex, gender identity, sexual orientation, disability.
- **Counts match this repository's audit:** 248,129 records, 136,680 unique texts, 18,667 hate and 229,462 non-hate. The document recommends duplicate-aware splitting, which this pipeline does.
- **Not documented:** the number of annotators, inter-annotator agreement, whether annotators saw context (parent comment or video), and what the 4,315 records with blank `source`/`script` are.

### What the audit found

| Property | Value |
|---|---|
| Raw rows | 248,129 |
| Raw label split | 229,462 non-hate / 18,667 hate (7.52% hate) |
| Exact duplicate texts | 111,449 (44.9% of rows) |
| Unique texts after normalized dedupe | 136,632 |
| Hate rows after dedupe | 10,877 (7.96%) |
| Conflicting labels on duplicate texts | 0 |
| Sources (raw) | YouTube 233,425 / Reddit 10,389 / blank 4,315 |
| Scripts (raw) | roman 185,082 / deva 51,374 / mixed 7,358 / blank 4,315 |

Other things found in the text: about 9,900 zero-width spaces (`U+200B`), about 17,000 `<USER>` placeholders (anonymized mentions), emojis, no URLs, and a few extremely long comments (max 5,250 words).

**Two structural problems worth knowing about:**

1. **Script is confounded with the label.** After dedupe, the hate rate is about **5.8% in Roman** text but about **14% in Devanagari and mixed** text. Results are always reported per script.
2. **Reddit is nearly empty of hate.** Only 41 unique Reddit rows are labelled hate (6 in the test set). Results on Reddit are not meaningful.

---

## Environment and hardware

| Item | Value |
|---|---|
| OS | Windows |
| Python | Official CPython 3.13 in a virtual environment (`.venv`). **Not** the MSYS2 Python, whose venv layout and PyTorch support differ |
| PyTorch | `torch 2.14.1+cu126` (CUDA 12.6 build) |
| Libraries | `transformers`, `datasets`, `scikit-learn`, `pandas`, `matplotlib`, `pyarrow`, `pyyaml` |
| GPU | NVIDIA GTX 1660 Ti, 6 GB VRAM |

### Why fp32 and not fp16

A 20-step benchmark of MuRIL training at batch size 8, max length 64:

| Precision | Speed | Peak GPU memory |
|---|---|---|
| fp32 | 0.192 s/step | 4.77 GB |
| fp16 | 0.535 s/step | 4.78 GB |

The GTX 1660 Ti has no tensor cores, so fp16 was about 2.8x **slower** and saved no memory. All training uses fp32. Real training peaks at about 5.0 GB, leaving about 1 GB of headroom, so do not raise the per-device batch above 8. If you hit an out-of-memory error, freeze MuRIL's embedding layer first (its vocabulary is large).

---

## Repository layout

```
.
├── README.md
├── PROGRESS.md                  # running handoff notes for contributors
├── LICENSE
├── requirements.txt
├── configs/                     # one YAML per stage (seeds, paths, hyperparameters)
├── src/
│   ├── data/                    # loading, auditing, cleaning, splitting
│   ├── models/                  # baseline and MuRIL fine-tuning
│   ├── train/
│   └── eval/
├── scripts/                     # small helper scripts (e.g. count_audit.py)
├── results/                     # metric reports (.md) and run log (runs.csv) are tracked
│   └── checkpoints/             # NOT tracked (large)
└── data/                        # NOT tracked (licensed data)
    ├── raw/hinghate/            # put the original TSV here, never edit it
    └── processed/               # parquet files written by Stage 2
```

Every stage is driven by a YAML config and logs to `results/runs.csv`, so any number in this README can be traced to a run.

---

## Pipeline, step by step

Each stage states **what** was done and **why**.

### Stage 1: Load and audit

- Load the TSV, print shape, label counts, source and script counts, nulls, empty texts.
- Write an audit report (duplicates, label ratio before and after dedupe, hate rate by script and source, word-length percentiles).
- Export random hate and non-hate rows to a local CSV so a human can read them for label quality.

**Why:** you cannot trust a dataset you have not looked at. This stage found the byte-identical files, the 45% duplicates, and the script confound before any model was trained.

### Stage 2: Clean, deduplicate, split

**Cleaning**

- Remove zero-width space (`U+200B`) and BOM (`U+FEFF`).
- Collapse repeated whitespace; collapse character runs of 3+ down to 2.
- **Keep** `<USER>`, emojis, casing, and both scripts. **Keep** `U+200C` and `U+200D` (zero-width non-joiner/joiner), because they affect how Devanagari conjuncts render.
- **No** stopword removal and **no** stemming. Those help classic bag-of-words models and hurt pretrained transformers.

**Deduplication**

- Texts are compared on a *normalized key* (lowercase, punctuation stripped, whitespace collapsed).
- Waterfall: 248,129 rows -> 136,680 unique raw texts -> 136,675 after cleaning -> **136,632** unique normalized keys.
- The pipeline fails loudly if two texts with the same key have different labels. There were none.
- Blank `source`/`script` values are mapped to `"unknown"` (no rows dropped).

**Why dedupe before splitting:** if copies of the same comment land in both train and test, the test score measures memorization. 45% of this corpus was duplicates, so skipping this step would have inflated every later number.

**Split**

- Stratified 80/10/10 on `(label, script)`, seed 42, so each split keeps the same hate rate and script mix.
- Verified: **zero** normalized-text overlap between train/val, train/test, and val/test.

| Split | Rows | Non-hate | Hate | Hate % |
|---|---|---|---|---|
| Train | 109,305 | 100,604 | 8,701 | 7.96% |
| Train, subsampled | 34,804 | 26,103 | 8,701 | 25.00% |
| Validation | 13,663 | 12,575 | 1,088 | 7.96% |
| Test | 13,664 | 12,576 | 1,088 | 7.96% |

**Why subsample:** the **training** set keeps all hate rows plus 3x as many non-hate rows. This speeds up training on a 6 GB GPU and lessens class imbalance. **Validation and test keep the natural 8% hate rate**, so reported metrics reflect realistic conditions.

**Token-length audit (MuRIL tokenizer, training set):** median 20 tokens, 95th percentile 67, and **5.5%** of texts exceed 64 tokens (hate comments slightly more: 6.9% vs 5.4%). The model therefore uses `max_len = 64`. Roman Hinglish splits into more tokens than words (for example `bakwas` becomes `ba ##k ##was`), so word counts understate token counts.

### Stage 3: Baseline (TF-IDF + Logistic Regression)

- Features: character n-grams (2-5) plus word n-grams (1-2), sublinear TF scaling, 230,898 features.
- Model: Logistic Regression, `class_weight="balanced"`, fit on the subsampled training set.
- Tuning on **validation only**: regularization `C` in {0.1, 0.5, 1, 2, 5, 10} and a decision-threshold sweep. Best: `C = 10`, threshold 0.73.
- Evaluated once on the test set.

**Why a baseline:** it tells you how much of MuRIL's score comes from the model and how much from the data being easy.

### Stage 4: MuRIL fine-tuning

| Setting | Value |
|---|---|
| Model | `google/muril-base-cased` + new 2-class head |
| Precision | fp32 |
| Max length | 64 tokens, dynamic padding |
| Batch | 8 per step x 4 gradient accumulation = effective 32 |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01, 10% linear warmup |
| Epochs | 3 |
| Loss | Class-weighted cross-entropy (weights about 0.67 non-hate / 2.0 hate) |
| Model selection | Best validation macro-F1 checkpoint |
| Threshold | Tuned on validation, then frozen for test |
| Seeds | 42, 43, 44 (only the training seed changes; the data split stays fixed) |
| Time | About 27 minutes per seed |

**Why MuRIL:** it was pretrained on Indic languages including transliterated (Roman-script) text, so it handles Hinglish spelling variation better than English-only models.

**Why several seeds:** training has randomness (head initialization, batch order, dropout). Three runs give a mean and standard deviation.

**Why tune the threshold on validation:** the model trained on 25% hate but is evaluated on 8% hate, so the default 0.5 cut-off is not necessarily right.

### Stage 5: Ensemble, confidence intervals, error analysis

Uses only the saved predictions (no retraining).

- **Ensemble:** the mean predicted probability of the three seeds, with the threshold re-tuned on validation (0.746).
- **Bootstrap 95% CIs:** 1,000 resamples over test rows, including a *paired* Roman-minus-Devanagari difference.
- **Near-duplicate checks:** TF-IDF cosine similarity of each test row to its closest training row, bucketed by similarity, plus label agreement with the nearest neighbour for very close pairs.
- **Political-term false-positive check:** false-positive rate on non-hate test rows with and without a list of political/communal terms (Roman and Devanagari spellings of: modi, bjp, congress, sarkar, rahul, yogi, hindu, muslim, jati, aarakshan).
- **Slices:** by script, source, length and emoji.
- **Error read:** the 50 most confident false positives and false negatives, read by hand.

### Stage 6: Verdict

Four pre-declared rules were applied to the results (script gap, seed stability, error character, otherwise-enough). See [Verdict](#verdict-and-what-is-not-established).

---

## Results

All numbers are on the **test set** (13,664 rows, 1,088 hate). "Hate-class F1" is the F1 for the hate class only. Accuracy is deliberately not reported: predicting "non-hate" for everything already scores about 92%.

### Baseline vs MuRIL

| Model | Macro-F1 | Hate precision | Hate recall | **Hate F1** | PR-AUC |
|---|---|---|---|---|---|
| TF-IDF + LR (threshold 0.73) | 0.7385 | 0.470 | 0.592 | **0.524** | 0.543 |
| MuRIL, mean of 3 single-seed runs | 0.8825 +/- 0.0016 | 0.788 +/- 0.012 | 0.779 +/- 0.018 | **0.784 +/- 0.003** | 0.874 +/- 0.003 |
| MuRIL, 3-seed probability ensemble (threshold 0.746) | 0.8929 | 0.822 | 0.784 | **0.802** | 0.890 |

PR-AUC does not depend on any threshold, so the gap between the baseline and MuRIL is real and not a tuning artifact. The ensemble's hate F1 has a bootstrap 95% CI of **[0.784, 0.821]**, a bootstrap standard error of about 0.01. The seed standard deviation (+/- 0.003) reflects training randomness only, so test-sampling noise is roughly three times larger. Report the CI, not the seed spread, as the uncertainty on a score.

### Single-seed runs

| Seed | Threshold | Macro-F1 | Hate F1 | PR-AUC |
|---|---|---|---|---|
| 42 | 0.89 | 0.8823 | 0.7832 | 0.8759 |
| 43 | 0.90 | 0.8842 | 0.7873 | 0.8711 |
| 44 | 0.84 | 0.8865 | 0.7917 | 0.8757 |

### Per script (ensemble)

| Script | Test hate rows | Hate F1 | 95% CI | PR-AUC |
|---|---|---|---|---|
| Roman | 583 | 0.789 | [0.761, 0.814] | 0.876 |
| Devanagari | 435 | 0.813 | [0.782, 0.840] | 0.903 |
| Mixed | 59 | 0.850 | not computed | 0.926 |
| Unknown | 11 | 0.818 | not computed | 0.909 |

**Paired difference (Roman minus Devanagari): -0.024, 95% CI [-0.064, +0.016].** The interval includes zero, so there is **no detectable difference at this sample size**. It does not prove the scripts are equal. Mixed and unknown have too few hate rows to interpret.

### Other slices (ensemble)

| Slice | Hate F1 | Note |
|---|---|---|
| Source: YouTube | 0.804 | 1,071 hate rows |
| Source: Reddit | 0.444 | **6 hate rows**; not interpretable |
| 1-10 words | 0.781 | Lowest precision; short abusive words can be banter |
| 11-25 words | 0.819 | |
| 26-50 words | 0.803 | |
| >50 words | 0.765 | 57 hate rows; may be truncation at 64 tokens (untested) |
| With emoji | 0.802 | |
| Without emoji | 0.803 | |

### Threshold study (single seed 42)

| Threshold | Precision | Recall | Hate F1 |
|---|---|---|---|
| 0.878 (tuned on val) | 0.785 | 0.784 | 0.784 |
| 0.50 (default) | 0.742 | 0.838 | 0.787 |

F1 barely changes with the threshold. The choice is a **product decision**: a higher threshold means fewer wrongly flagged comments, a lower one means fewer missed hateful comments.

### Near-duplicate and leakage checks

- Test rows bucketed by similarity to their closest training row: hate F1 does not collapse in the least similar bucket (single-seed range 0.75-0.81 across quartiles), so near-copies leaking across the split are not the main driver of the score.
- Only **158 test rows (1.2%)** have a training row with cosine similarity above 0.8. Their labels agree with the nearest neighbour **155/158 (98.1%)**; for hate-labelled rows it is **12/15 (80%, Wilson 95% CI 55-93%)**. With n=15 this says nothing reliable about the dataset's overall label error rate.
- These checks measure character overlap only. They cannot rule out paraphrases or comments from the same video thread.

### Political-term false positives

False-positive rate on **non-hate** test rows: **1.52% (20/1,315)** with political/communal terms vs **1.47% (165/11,261)** without (Wilson 95% CIs [0.99%, 2.34%] vs [1.26%, 1.70%]). No detectable keyword effect.

This does **not** mean the model ignores politics: reading the most confident false positives by hand shows mostly hostile political abuse (insults aimed at politicians or parties). The model responds to hostility, including political hostility, more than to the mere mention of a name.

### Failure modes (from reading the top-50 false positives and false negatives)

- **False positives** (non-hate flagged at high confidence): mostly aggressive partisan criticism, insults at political figures, and caste/reservation arguments where a policy or person, not a group, is attacked. The selection was by *confidence*, so this describes the extremes, not the average error.
- **False negatives** (hate missed): mostly obfuscated spellings, such as letters separated by spaces, and sarcasm without overt abusive vocabulary. These are **model weaknesses**, not label problems.

---

## Label audit (100 rows, hand-tagged)

**Why:** the model's F1 is only as meaningful as the labels it is measured against. A manual check was run on the **positive** labels.

**Method**

- 100 random **test** rows with `label = 1` (export: `results/audit_100.csv`, **not tracked**; it contains raw comments).
- One annotator, a **written guide**, **no parent context**, no access to the thread or video.
- This tagging guide is the annotator's own and is **not** the dataset's two-gate guideline, so the tags are not a direct test of conformity to it.
- Four tags, applied in order, stopping at the first match:

| Tag | Meaning |
|---|---|
| `group_hate` | Attacks, dehumanizes, threatens, or calls for exclusion or harm against people because of religion, caste, ethnicity, gender, region/language, or nationality |
| `personal_abuse` | Insults or demeans an individual (character, intelligence, appearance, family), vulgar or not, with no group identity involved |
| `political_criticism` | Criticism, mockery or accusations aimed at a government, party, policy, media, or public figure's conduct, without abuse beyond harsh criticism |
| `neutral` | No visible hostility (question, praise, quote, proverb, identity pride, or a plain mislabel) |

- Marks appended to a tag: `?` = contestable (another annotator might choose differently), `~` = the tag could flip with context, `!` = flagged as extreme hate.
- Model scores were visible next to the rows for part of the tagging, which may have anchored some tags.

**Results** (95% Wilson intervals; about +/-10 points at n=100)

| Tag | Count | 95% CI |
|---|---|---|
| `group_hate` | 17 | 11-26% |
| `personal_abuse` | 22 | 15-31% |
| `political_criticism` | 19 | 13-28% |
| `neutral` | 42 | 33-52% |

Marks: 25 rows `?`, 7 rows `~`, 4 rows `!`. Of the 42 `neutral` rows, 18 carry `?` and 6 carry `~`, and 21 carry neither. The share that is confidently "no visible hostility" is therefore 21%, and a fair range is about **21-42%** depending on how the marked rows are counted.

**How the model scored these rows** (ensemble, threshold 0.746):

| Tag | Flagged as hate (all rows) | Flagged (rows without `?` or `~`) | Median probability |
|---|---|---|---|
| `personal_abuse` | 21/22 (95%) | 18/18 | 0.991 |
| `group_hate` | 14/17 (82%) | 12/14 | 0.991 |
| `political_criticism` | 14/19 (74%) | 14/19 | 0.989 |
| `neutral` | 29/42 (69%) | 14/21 (67%) | 0.991 |

**Reading:** the model partly tracks hostility (personal abuse is flagged most, neutral least) but still flags about two thirds of the rows judged to show no visible hostility, so it follows the dataset's label as shipped more than the documented hate definition. Differences between adjacent tags (for example group hate vs neutral) are within the intervals and should not be over-interpreted.

**Limits:** single annotator (no agreement score), no parent context, n=100, and **only positive labels were audited**: nothing is known about hate hiding among the 0 labels.

---

## Verdict, and what is NOT established

**Rules declared before looking at the results:**

| Rule | Outcome |
|---|---|
| Roman hate F1 far below Devanagari -> need more Roman data | Not triggered: paired CI [-0.064, +0.016] includes zero |
| Seed std of hate F1 above about 2 points -> unstable | Not triggered: std 0.0034 (but see the wider bootstrap CI above) |
| Errors dominated by label problems -> need a cleaner benchmark | **Supported, mainly by the label audit**: only 17% of audited positives are strict group hate, and 62-83 of 100 fall in categories the dataset's own guideline excludes. The error read adds that false negatives are model weaknesses, so errors are mixed, not purely label-driven |
| Otherwise HINGHATE is enough | Not applicable |

**Verdict:** HINGHATE is enough to **train and benchmark within HINGHATE** (the model is stable and clearly beats a strong baseline). It is **not enough** to support claims about hate-speech detection in general: the shipped positive label appears broader than the documented definition, and nothing has been tested outside this dataset. A second benchmark with a different, stricter label scheme (for example HASOC) is needed before any wider claim.

**Label scope vs the dataset's own definition.** By a single annotator's reading, without context, 62-83 of 100 audited positives fall into categories the guideline excludes (political criticism 19, personal abuse 22, no visible hostility 42, of which 21 carry at least one mark: 18 with `?`, 6 with `~`). The shipped labels therefore appear broader than the documented definition. The audit cannot say why: annotators may have seen context that is not in the file, or the guideline may have been applied loosely.

**Suggested wording for the task:** "detecting hostile or polarizing Hinglish comments, as labelled in HINGHATE" (the dataset documents a narrower definition of hate than its labels appear to follow), not "hate speech detection".

**Not established**

- **External generalization.** No cross-dataset or cross-platform test has been run. About 94% of the data is YouTube.
- **Label provenance.** The dataset documentation (`Document.pdf` on the IEEE DataPort page) describes human annotation with a two-gate guideline: hate requires that the content (1) targets people as a group or as members of a group and (2) attacks that group's worth, humanity, belonging, rights or safety. It states that political criticism, disagreement, ordinary rudeness and insults aimed at an individual, including a named politician, are not hate unless both gates are met. The number of annotators, agreement, and whether annotators saw context are not documented, the 4,315 rows with blank source/script are not explained, and the dataset authors were not contacted.
- **Label quality at scale.** The audit is one annotator, 100 positive rows, no parent context, so it is indicative, not a measured error rate. It suggests the labels as shipped are broader than the documented guideline, unless annotators saw context that this file does not contain.
- **Hate hidden in label 0.** Never audited.
- **Reddit.** 6 test hate rows; any statement is unsupported.
- **The 15-row near-neighbour result.** Too small to support a claim.
- **Roman vs Devanagari.** Not shown to differ, and not shown to be equal.
- **Comments from the same thread.** The split is random, and no video or thread IDs exist, so thread-level leakage cannot be excluded.
- **Other models.** Only MuRIL (plus the TF-IDF baseline) was trained. Whether another model changes the picture is untested.
- **Production use.** The operating point (threshold) is a product decision, and nothing here supports automated enforcement against real users.

---

## How to reproduce

### 1. Environment (Windows, cmd or PowerShell)

```bash
py -3.13 -m venv .venv          # use the official Python, not MSYS2
.venv\Scripts\activate
python -m pip install --upgrade pip
# CUDA build of PyTorch (pick the command matching your GPU/driver from pytorch.org):
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets scikit-learn pandas matplotlib pyarrow pyyaml
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Get the data

Download HINGHATE (login required) and place **one** file at:

```
data/raw/hinghate/hinglish_corpus_final.tsv
```

### 3. Run the stages

Each stage is one command driven by its config:

```bash
python -m src.<stage module> --config configs/<stage config>.yaml
```

For example, Stage 2 and Stage 3:

```bash
python -m src.data.preprocess --config configs/stage2_preprocess.yaml
python -m src.models.baseline --config configs/stage3_baseline.yaml
```

The audit (Stage 1), MuRIL fine-tuning (Stage 4) and error analysis (Stage 5) follow the same pattern; check the `configs/` and `src/` folders for their module names. Outputs land in `data/processed/` and `results/`.

---

## Roadmap

- [x] Stages 1-6 on HINGHATE, including bootstrap CIs and a 100-row label audit
- [ ] **Re-tag with the dataset's own rule.** Re-tag the 100 audit rows with the two-gate guideline (does it target a group by identity? does it attack that group's worth, humanity, belonging, rights or safety?) to measure conformity to the documented definition directly.
- [ ] **External test set.** HASOC (Hindi/Hinglish tracks) requires registration with the organizers: <https://hasocfire.github.io>. Fix the label mapping *before* looking at results, and report a strict and a broad view.
- [ ] **Second annotator.** Have another person tag 30-50 of the same audit rows blind and report Cohen's kappa.
- [ ] **Context.** HASOC's conversational subtasks include parent comments, which can show how much of the apparent label noise is missing context.
- [ ] **Model comparison.** Same recipe with `xlm-roberta-base` and a Hinglish-specific model (check memory and tokenization on the 6 GB GPU first). Note: `ai4bharat/indic-bert` needed a gated-access login when tried.
- [ ] **Robustness to obfuscation.** False negatives are dominated by spaced-out or disguised spellings; test a normalization step that rejoins them.
- [ ] **Longer inputs.** Check whether the weaker >50-word slice is a truncation effect (`max_len` 96 needs a memory re-test).
- [ ] **Audit negatives.** Sample label-0 rows with high model scores to look for missed hate.

---

## Data, ethics and licensing

- **No dataset content is committed.** HINGHATE is provided for research and educational purposes; its IEEE DataPort page lists no explicit license and the files require a subscription. Do not push the raw data, processed parquet files, or anything containing comment text (including prediction files, error exports, label samples and `audit_100.csv`) to a public repository. These are excluded in `.gitignore`.
- **Offensive content.** The data contains abusive, hateful and distressing language. Take breaks when reading samples, and avoid reproducing harmful examples unnecessarily.
- **Not a production moderation system.** The model is trained on broad labels from mainly one platform and has no external validation. It should not be used to take automated action against real users.
- Follow each platform's terms and the dataset's terms when using or citing the data.

Recommended `.gitignore` entries:

```
data/
.venv/
*.pt
*.bin
results/checkpoints/
results/runs/
results/*.parquet
results/*sample*.csv
results/errors.csv
results/audit_100*.csv
__pycache__/
```

---

## Authors, license, citation

- **Authors / contributors:** Ayushmaan ([@Ayushu69](https://github.com/Ayushu69)): project lead, data audit, modelling, label audit
- **License (code):** MIT (see `LICENSE`). Applies to the code only. The HINGHATE dataset is not included and is subject to its own terms on IEEE DataPort.
- **Dataset:** HINGHATE (IEEE DataPort, doi:10.21227/7ye1-4489). The page lists no explicit license; it states the data is provided for research and educational purposes, and the files require an IEEE DataPort subscription. This repository does not redistribute any of it.
- **Citation:** Shubham Shukla, Sushama Nagpal, Sangeeta Sabharwal (2026). HINGHATE: One of the Largest Hinglish Hate Speech Datasets from YouTube and Reddit. IEEE Dataport. https://doi.org/10.21227/7ye1-4489
- **Model:** MuRIL: Multilingual Representations for Indian Languages, `google/muril-base-cased` (Khanuja et al., 2021).
- **Tooling note:** pipeline code was developed with AI coding assistants; results and conclusions were reviewed by the author.

For day-to-day handoff notes, decisions and the exact next task, see `PROGRESS.md`.