# Model Card — Explainable TFT for CKD Progression Prediction

> **⚠️ IMPORTANT DISCLAIMER**
> This is a **research prototype** trained on **synthetically generated** longitudinal CKD data.
> It is **NOT intended for clinical diagnosis or treatment**.
> This system has **not been clinically validated**.
> Predictions must **never** be used to inform real medical decisions.

---

## 1. Model Summary

| Property | Value |
|---|---|
| **Model Type** | Temporal Fusion Transformer (TFT) |
| **Architecture** | Multi-task, multi-horizon PyTorch model |
| **Tasks** | CKD Stage Classification, eGFR Regression, Progression Risk |
| **Horizons** | 3 months, 6 months, 12 months |
| **Data** | Synthetically generated longitudinal CKD records |
| **Status** | Research prototype — Phase 2 of 4 |
| **Phase 3 Addition** | Explainability API (VSN weights + Temporal Attention) |

---

## 2. Architecture

The model is a custom Temporal Fusion Transformer based on [Lim et al. (2021)](https://doi.org/10.1016/j.ijforecast.2021.03.012), adapted for multi-task multi-horizon clinical sequence modeling.

### Components

| Component | Description |
|---|---|
| **Static Encoder** | Variable Selection Network (VSN) — learns importance of static patient attributes |
| **Temporal Encoder** | LSTM + VSN — processes each visit with learned feature selection |
| **Static Enrichment** | GRN that injects static context into temporal representations |
| **Temporal Attention** | Interpretable Multi-Head Attention (mean-head, as in original TFT paper) |
| **Output Heads** | Three task-specific heads (Stage: 6-class, eGFR: scalar, Progression: binary) |

### Key Configuration

```json
{
  "n_static_features": 4,
  "n_tv_features": 29,
  "n_temporal_features": 2,
  "hidden_size": 16,
  "num_attention_heads": 1,
  "n_classes": 6,
  "horizons": ["3m", "6m", "12m"]
}
```

---

## 3. Inputs

### Static Features (4)
Features taken from the patient's first visit that are treated as time-invariant:

| Feature | Type |
|---|---|
| `age` | Continuous (standardised) |
| `height_cm` | Continuous (standardised) |
| `family_history_ckd` | Binary (0/1) |
| `baseline_bmi` | Continuous (standardised) |

### Time-Varying Features (29)
Features measured at each visit:

`weight_kg`, `bmi`, `systolic_bp`, `diastolic_bp`, `serum_creatinine_mg_dl`,
`egfr_ml_min_1_73m2`, `bun_mg_dl`, `uacr_mg_g`, `fasting_glucose_mg_dl`,
`hba1c_pct`, `hemoglobin_g_dl`, `anemia`, `sodium_meq_l`, `potassium_meq_l`,
`calcium_mg_dl`, `phosphate_mg_dl`, `bicarbonate_meq_l`, `uric_acid_mg_dl`,
`acei_arb`, `sglt2_inhibitor`, `diuretic`, `antihypertensive`, `statin`,
`epo`, `iron_supplement`, `metformin`, `insulin`, `on_dialysis`, `kidney_transplant`

### Temporal Context Features (2)
`visit_number`, `days_from_baseline`

---

## 4. Outputs

| Task | Type | Output Shape | Loss |
|---|---|---|---|
| CKD Stage | 6-class classification | `[3, 6]` logits | Cross-Entropy |
| eGFR | Scalar regression | `[3]` values | Smooth L1 |
| Progression Risk | Binary classification | `[3]` logits | BCE with Logits |

Stage labels: `G1`, `G2`, `G3a`, `G3b`, `G4`, `G5`

---

## 5. Training Data

| Property | Value |
|---|---|
| **Data Type** | **Synthetically generated** — NOT real patient data |
| **Total patients** | 40,000 synthetic patients |
| **Train split** | 28,000 patients |
| **Validation split** | 2,000 patients |
| **Test split** | 10,000 patients |
| **Splits** | Patient-disjoint (no patient appears in >1 split) |
| **Sequence length** | 8 visits per patient |

---

## 6. Explainability

### Methods Used

Phase 3 exposes two TFT-native interpretability mechanisms. These are **model interpretability signals** — they indicate what the model paid attention to, not causal relationships between features and outcomes.

#### Variable Selection Networks (VSN)
- **What it is:** Softmax weights computed by the TFT's variable selection network at each forward pass.
- **What it tells you:** Which features the model assigned higher importance to when making this specific prediction.
- **What it does NOT tell you:** Whether those features causally influence CKD progression. Correlation and attention are not causation.
- **Static VSN:** Single weight vector per patient `[N_static]`
- **Temporal VSN:** Per-timestep weight vector `[SeqLen, N_tv + N_temporal]`

#### Interpretable Multi-Head Attention
- **What it is:** The mean-head temporal self-attention weights `[SeqLen, SeqLen]`.
- **What it tells you:** Which historical visits the model gave more weight to when making its prediction from the most recent visit.
- **What it does NOT tell you:** Whether events at those time steps caused the predicted outcome.

> ⚠️ **Attention is not causality. Feature importance is not causal inference.**

### SHAP
SHAP was **not used** as the primary explanation method. Reason: The TFT's native VSN + attention mechanism is more architecturally aligned with how the model actually makes decisions. SHAP would require many hundreds of forward passes on CPU and is impractical for this architecture in this environment.

---

## 7. API (Phase 3)

Start the backend:
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Model metadata + disclaimer |
| `GET` | `/patient/{id}` | Synthetic patient visit history |
| `POST` | `/predict` | Predictions (stage, eGFR, progression × 3 horizons) |
| `POST` | `/explain` | Predictions + VSN importance + attention |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "12345"}'
```

---

## 8. Limitations

1. **Synthetic data only** — trained and evaluated on computer-generated data, not real patient records.
2. **Not clinically validated** — performance on synthetic data cannot establish clinical effectiveness.
3. **Attention ≠ causality** — temporal attention weights indicate model focus, not causal importance.
4. **Feature importance ≠ causal inference** — VSN weights indicate what the model uses, not what drives disease.
5. **No uncertainty quantification** — eGFR outputs are point estimates, not confidence intervals.
6. **Limited training** — trained for 1 epoch with a small model (hidden_size=16) as a prototype.
7. **CPU only** — GPU support is available but not tested in the prototype environment.
8. **External validation required** — before any real-world deployment, the model would need validation on real clinical data by qualified medical professionals.

---

## 9. Ethical Considerations

- This system must **never** be used to inform real medical decisions.
- Predictions about CKD progression affect patient quality of life; any clinical tool would require rigorous regulatory review.
- Synthetic data training limits generalisability to real patient populations.

---

## 10. Run Tests

```bash
# All tests (Phase 1 + Phase 2 + Phase 3)
pytest tests/ -v --tb=short

# Phase 3 only
pytest tests/test_inference_engine.py tests/test_explainability.py tests/test_api_endpoints.py -v

# Generate Phase 3 artifacts
python -m scripts.generate_phase3_artifacts
```

---

*Research Prototype | Synthetic Data | Not for Clinical Use*
