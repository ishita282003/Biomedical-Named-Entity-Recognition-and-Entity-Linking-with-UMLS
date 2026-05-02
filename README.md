# 🏥 Biomedical Named Entity Recognition and Entity Linking with UMLS

**Author:** Ishita Singh | B.Tech CSE (2021–2025), IGDTUW

---

## Overview

This project implements and evaluates a two-stage pipeline for processing clinical text from i2b2 datasets:

1. **Bio-NER** — Identifying and classifying medical entities (diseases, treatments, medications, ADEs) using LSTM-based deep learning models.
2. **Bio-EL** — Linking the extracted entities to standardized concepts in the **Unified Medical Language System (UMLS)** using BERT embeddings and cosine similarity.

---

## Datasets

Three de-identified clinical datasets from the [i2b2 NLP challenges](https://www.i2b2.org/NLP/) are used:

| Dataset | Focus | Total Entities |
|---------|-------|----------------|
| **i2b2 2010** | Discharge summaries; problems, treatments, tests & relations | 3,74,397 |
| **i2b2 2012** | Medical events, time expressions, temporal relations | 1,03,502 |
| **i2b2 2018** | Medications, dosage, frequency, route, ADEs | 9,43,987 |

---

## Part I — Named Entity Recognition (NER)

### Models

| Model | Description | Total Params |
|-------|-------------|-------------|
| **LSTM** | Single-directional; learns temporal dependencies | 2,069,680 |
| **BiLSTM** | Bidirectional; accesses both past and future context | 682,248 |
| **BiLSTM + Time Distributed Dense** | Per-token dense layer via `TimeDistributed` wrapper | 645,128 |
| **BiLSTM + TDD + CNN** | Hybrid; CNN captures local patterns before BiLSTM | 634,920 |

### NER Results

| Model | i2b2 2010 | i2b2 2012 | i2b2 2018 | Avg Accuracy |
|-------|-----------|-----------|-----------|-------------|
| **LSTM** | 93% | 83% | 97% | **91%** |
| BiLSTM + TDD + CNN | 93% | 81% | 96% | 90% |
| BiLSTM + TDD | 92% | 83% | 96% | 90.33% |
| BiLSTM | 92% | 81% | 95% | 89.33% |

> **LSTM** achieves the best average accuracy (91%) across all three datasets.

---

## Part II — Entity Linking with UMLS

### Pipeline

```
i2b2 Entities (NER output)
        │
        ▼
BERT Embeddings (bert-base-uncased / BioBERT / SciBERT)
        │
        ▼
Cosine Similarity vs. UMLS Concept Embeddings
        │
        ▼
Threshold @ 0.85 → Match / No Match
        │
        ▼
Results DataFrame [Entity | Match | Matched Entity | Score]
```

### Key Steps

**1. Extract Entities** from i2b2 clinical notes (e.g., `"diabetes"`, `"aspirin"`, `"CT scan"`).

**2. Retrieve UMLS Concepts** with Concept Unique Identifiers (CUIs):
- `"Diabetes Mellitus"` → CUI: C0011849
- `"Aspirin"` → CUI: C0004057
- `"CT Scan"` → CUI: C0033144

**3. Generate BERT Embeddings** for both i2b2 entities and UMLS concepts.

**4. Compute Cosine Similarity:**
```
Cosine Similarity = (A · B) / (‖A‖ ‖B‖)
```

**5. Apply Threshold** — Match if similarity ≥ 0.85.

**6. Save Results** — DataFrame with: `Entity`, `Match`, `Matched Entity`, `Score`.

### Entity Linking Results

**Match % by dataset (base BERT):**

| i2b2 2010 | i2b2 2012 | i2b2 2018 |
|-----------|-----------|-----------|
| 29.94% | 22.61% | 46.56% |

**BERT variant comparison on i2b2 2018:**

| BERT | BioBERT | SciBERT |
|------|---------|---------|
| 46.56% | **99.05%** | 47.35% |

> **BioBERT** achieves 99.05% match rate — domain-specific pre-training is critical for biomedical entity linking.

---

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Embeddings:** HuggingFace Transformers (`bert-base-uncased`, `BioBERT`, `SciBERT`)
- **Similarity:** `sklearn.metrics.pairwise.cosine_similarity`
- **Knowledge Base:** UMLS (Unified Medical Language System)
- **Data:** i2b2 2010, 2012, 2018 clinical NLP challenge datasets

---

## Key Findings

1. **LSTM** outperforms more complex BiLSTM variants for NER (avg 91% accuracy).
2. **BioBERT** far outperforms general BERT for entity linking (99.05% vs 46.56% on i2b2 2018).
3. Entity linking match rates are lower on i2b2 2010/2012 due to broader, less medication-specific entity categories.
4. A cosine similarity threshold of **0.85** provides a good precision/recall tradeoff.

---

## Evaluation Metrics

- **Accuracy** = (TP + TN) / (TP + FP + TN + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **% Matched Entities** = (Matched / Total Extracted) × 100

---

## Future Work

- Fine-tune NER models on domain-specific biomedical corpora
- Expand UMLS concept coverage to reduce unmatched entities
- Add multilingual support for diverse clinical settings
- Integrate the pipeline into EHR systems for real-time clinical decision support
- Explore joint NER + EL end-to-end models