import type {
  HealthResponse,
  ModelInfoResponse,
  PatientHistoryResponse,
  PredictionResponse,
  ExplanationResponse
} from '../types/api';

export const mockHealth: HealthResponse = {
  status: 'ok',
  model_loaded: true,
  data_disclaimer: 'Research Prototype. Synthetic data only. Not validated for clinical use.'
};

export const mockModelInfo: ModelInfoResponse = {
  model_type: 'TemporalFusionTransformer',
  checkpoint: 'models/checkpoints/best_model.pt',
  data_type: 'synthetic_longitudinal_ckd',
  disclaimer: 'Academic prototype. Strictly for research and educational purposes.',
  architecture: {
    hidden_size: 16,
    num_attention_heads: 1,
    n_static_features: 10,
    n_tv_features: 29,
    n_temporal_features: 3,
    n_classes: 6
  },
  tasks: ['stage_classification', 'continuous_egfr_forecasting', 'progression_risk_prediction'],
  horizons: ['3m', '6m', '12m'],
  stage_labels: ['G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'],
  explainability_methods: ['VariableSelectionNetwork', 'TemporalSelfAttention'],
  training: {
    optimizer: 'AdamW',
    learning_rate: 0.001,
    num_epochs: 10,
    random_seed: 42
  },
  limitations: [
    'Synthetic data only; not validated on real clinical cohorts',
    'Feature salience is non-causal statistical correlation',
    'Right-censored terminal encounters'
  ]
};

export function getMockPatientData(patientId: string): PatientHistoryResponse {
  const numId = parseInt(patientId, 10) || 60001;
  const visitCount = 8;
  const visits = [];

  for (let i = 1; i <= visitCount; i++) {
    visits.push({
      visit_index: i - 1,
      visit_number: i,
      days_from_baseline: (i - 1) * 90 + ((numId + i * 7) % 15),
      is_padding: false
    });
  }

  return {
    patient_id: patientId,
    split: 'test',
    n_visits: visitCount,
    sequence_length: 8,
    visits,
    disclaimer: 'Synthetic research cohort. Not for real patient management.'
  };
}

export function getMockPrediction(patientId: string): PredictionResponse {
  const numId = parseInt(patientId, 10) || 60001;
  const isHighRisk = numId % 2 === 1;

  if (isHighRisk) {
    return {
      patient_id: patientId,
      prediction: {
        stage: {
          '3m': {
            predicted_stage: 'G3b',
            probabilities: { G1: 0.01, G2: 0.04, G3a: 0.18, G3b: 0.58, G4: 0.16, G5: 0.03 }
          },
          '6m': {
            predicted_stage: 'G4',
            probabilities: { G1: 0.00, G2: 0.02, G3a: 0.11, G3b: 0.36, G4: 0.49, G5: 0.02 }
          },
          '12m': {
            predicted_stage: 'G4',
            probabilities: { G1: 0.00, G2: 0.01, G3a: 0.05, G3b: 0.22, G4: 0.56, G5: 0.16 }
          }
        },
        egfr: {
          '3m': 38.4,
          '6m': 29.1,
          '12m': 22.8
        },
        progression: {
          '3m': { probability: 0.38, predicted_class: 0 },
          '6m': { probability: 0.74, predicted_class: 1 },
          '12m': { probability: 0.89, predicted_class: 1 }
        }
      },
      metadata: {
        disclaimer: 'Academic prototype. Sub-20ms inference latency.',
        data_type: 'synthetic_tft_inference'
      }
    };
  }

  // Stable / Moderate profile
  return {
    patient_id: patientId,
    prediction: {
      stage: {
        '3m': {
          predicted_stage: 'G2',
          probabilities: { G1: 0.22, G2: 0.68, G3a: 0.08, G3b: 0.02, G4: 0.00, G5: 0.00 }
        },
        '6m': {
          predicted_stage: 'G2',
          probabilities: { G1: 0.18, G2: 0.65, G3a: 0.14, G3b: 0.03, G4: 0.00, G5: 0.00 }
        },
        '12m': {
          predicted_stage: 'G3a',
          probabilities: { G1: 0.10, G2: 0.48, G3a: 0.36, G3b: 0.06, G4: 0.00, G5: 0.00 }
        }
      },
      egfr: {
        '3m': 68.2,
        '6m': 64.5,
        '12m': 58.1
      },
      progression: {
        '3m': { probability: 0.12, predicted_class: 0 },
        '6m': { probability: 0.24, predicted_class: 0 },
        '12m': { probability: 0.35, predicted_class: 0 }
      }
    },
    metadata: {
      disclaimer: 'Academic prototype. Sub-20ms inference latency.',
      data_type: 'synthetic_tft_inference'
    }
  };
}

export function getMockExplanation(patientId: string): ExplanationResponse {
  const pred = getMockPrediction(patientId);
  const numId = parseInt(patientId, 10) || 60001;
  const isHighRisk = numId % 2 === 1;

  return {
    patient_id: patientId,
    prediction: pred.prediction,
    feature_importance: {
      static: isHighRisk ? {
        family_history_ckd: 0.499,
        baseline_bmi: 0.334,
        standing_height: 0.130,
        baseline_diabetes: 0.112,
        baseline_hypertension: 0.098,
        baseline_cvd: 0.064
      } : {
        baseline_bmi: 0.412,
        standing_height: 0.225,
        family_history_ckd: 0.184,
        baseline_hypertension: 0.142,
        baseline_diabetes: 0.085
      },
      time_varying: isHighRisk ? {
        serum_creatinine: 0.245,
        hemoglobin: 0.182,
        uacr: 0.158,
        serum_calcium: 0.092,
        diastolic_bp: 0.076,
        metformin: 0.068,
        serum_bicarbonate: 0.054,
        blood_urea_nitrogen: 0.048,
        systolic_bp: 0.042
      } : {
        serum_creatinine: 0.198,
        hemoglobin: 0.165,
        diastolic_bp: 0.115,
        serum_calcium: 0.085,
        systolic_bp: 0.078,
        uacr: 0.064
      },
      note: 'Extracted from TFT Variable Selection Networks (VSN) Softmax Gating.'
    },
    temporal_attention: [
      { visit_index: 0, visit_number: 1, days_from_baseline: 0, attention_weight: 0.148, is_padding: false, disclaimer: '' },
      { visit_index: 1, visit_number: 2, days_from_baseline: 92, attention_weight: 0.142, is_padding: false, disclaimer: '' },
      { visit_index: 2, visit_number: 3, days_from_baseline: 184, attention_weight: 0.125, is_padding: false, disclaimer: '' },
      { visit_index: 3, visit_number: 4, days_from_baseline: 275, attention_weight: 0.118, is_padding: false, disclaimer: '' },
      { visit_index: 4, visit_number: 5, days_from_baseline: 366, attention_weight: 0.121, is_padding: false, disclaimer: '' },
      { visit_index: 5, visit_number: 6, days_from_baseline: 458, attention_weight: 0.126, is_padding: false, disclaimer: '' },
      { visit_index: 6, visit_number: 7, days_from_baseline: 550, attention_weight: 0.152, is_padding: false, disclaimer: '' },
      { visit_index: 7, visit_number: 8, days_from_baseline: 642, attention_weight: 0.168, is_padding: false, disclaimer: '' }
    ],
    metadata: {
      patient_id: patientId,
      model_checkpoint: 'best_model.pt',
      explanation_method: 'VariableSelectionNetworks_and_TemporalAttention',
      explanation_components: ['static_vsn', 'time_varying_vsn', 'temporal_attention'],
      generated_at: new Date().toISOString(),
      disclaimer: 'In-model glass-box feature salience (non-causal statistical correlation).'
    }
  };
}
