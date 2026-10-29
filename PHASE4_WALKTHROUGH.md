# Phase 4 Walkthrough: CKD Prediction Dashboard

This document outlines the final deliverables for **Phase 4** of the Explainable Temporal Fusion Transformer for CKD Progression Prediction project.

The final piece of the system—a research-grade, multi-page **Streamlit Dashboard**—has been successfully built and integrated with the Phase 3 FastAPI backend. It demonstrates the multi-task, multi-horizon predictions and explainability features developed in earlier phases.

## 1. What was built

We have developed a comprehensive Streamlit application (`dashboard.py`) with 5 dedicated pages located in `app/pages/`:

1. **Overview**: Project description, system architecture diagram, and a live check of the backend model configuration.
2. **Patient Prediction**: A search interface (using synthetic patient IDs like `60001` from the test cohort) that visualises the predicted CKD stage, 3m/6m/12m eGFR trajectory, and progression risk.
3. **Explainability**: An interface to generate on-the-fly model explanations (VSN feature importance and temporal attention) for a specific prediction, alongside global interpretability charts derived from the Phase 3 evaluation cohort.
4. **Model Performance**: A unified view of the Phase 2 test set evaluation metrics for the TFT, compared against the baseline models.
5. **Model Information**: A display of the `MODEL_CARD.md`, providing full transparency on data, architecture, and limitations.

Additionally, a thin API client (`app/api_client.py`) safely routes all requests to the backend, and shared utilities (`app/utils.py`) ensure the persistent display of required synthetic-data and non-clinical disclaimers across every page.

## 2. Running the System

To use the dashboard, both the backend (Phase 3) and the frontend (Phase 4) must be running.

### Step A: Start the Backend API
In a new terminal:
```bash
# Ensure you are in the project root and virtual environment is active
uvicorn src.api.main:app --reload --port 8000
```
This loads the TFT checkpoint, the preprocessing pipeline, and exposes the `/predict` and `/explain` endpoints.

### Step B: Start the Streamlit Dashboard
In another terminal:
```bash
# Ensure you are in the project root and virtual environment is active
streamlit run dashboard.py
```
This will open the dashboard in your browser (typically at `http://localhost:8501`).

## 3. Demonstration Flow

1. Open the **Overview** page and confirm the **Backend Status** in the sidebar reads `🟢 Backend: Connected`.
2. Navigate to the **Patient Prediction** page.
3. Enter `60001` (a synthetic patient ID from the test set) and click **Look Up Patient**.
4. Review the observed patient history.
5. Click **Run Prediction + Explanation**.
6. Explore the multi-horizon predictions for CKD Stage, eGFR trajectory, and Progression risk.
7. Navigate to the **Explainability** page.
8. The explanation for patient `60001` will automatically load. Review the Static and Time-Varying Feature Importance, and the Temporal Attention weights indicating which historical visits the model focused on.
9. Switch to the **Global Model Interpretability** tab to see how the model behaves across the entire test cohort.

## 4. Testing & Verification

As part of Phase 4, we also added frontend API client tests:
- Checked error handling for backend outages (ConnectionError).
- Checked graceful handling of invalid patient IDs (HTTP 404).
- Confirmed correct response parsing.

The complete project test suite now contains **127 passing tests (1 skipped)**, ensuring robust functionality across data preprocessing, model inference, API routing, and frontend communication.
