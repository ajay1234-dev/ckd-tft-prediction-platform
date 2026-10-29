export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  data_disclaimer: string;
}

export interface ModelInfoResponse {
  model_type: string;
  checkpoint: string;
  data_type: string;
  disclaimer: string;
  architecture: {
    hidden_size: number;
    num_attention_heads: number;
    n_static_features: number;
    n_tv_features: number;
    n_temporal_features: number;
    n_classes: number;
  };
  tasks: string[];
  horizons: string[];
  stage_labels: string[];
  explainability_methods: string[];
  training: {
    optimizer: string;
    learning_rate: number;
    num_epochs: number;
    random_seed: number;
  };
  limitations: string[];
}

export interface PatientVisit {
  visit_index: number;
  visit_number: number | null;
  days_from_baseline: number | null;
  is_padding: boolean;
}

export interface PatientHistoryResponse {
  patient_id: string;
  split: string;
  n_visits: number;
  sequence_length: number;
  visits: PatientVisit[];
  disclaimer: string;
}

export interface HorizonStage {
  predicted_stage: string;
  probabilities: Record<string, number>;
}

export interface HorizonProgression {
  probability: number;
  predicted_class: number;
}

export interface PredictionMeta {
  disclaimer: string;
  data_type: string;
}

export interface AllPredictions {
  stage: Record<string, HorizonStage>;
  egfr: Record<string, number>;
  progression: Record<string, HorizonProgression>;
}

export interface PredictionResponse {
  patient_id: string;
  prediction: AllPredictions;
  metadata: PredictionMeta;
}

export interface AttentionStep {
  visit_index: number;
  visit_number: number;
  days_from_baseline: number | null;
  attention_weight: number;
  is_padding: boolean;
  disclaimer: string;
}

export interface FeatureImportance {
  static: Record<string, number>;
  time_varying: Record<string, number>;
  note: string;
}

export interface ExplanationMeta {
  patient_id: string;
  model_checkpoint: string;
  explanation_method: string;
  explanation_components: string[];
  generated_at: string;
  disclaimer: string;
}

export interface ExplanationResponse {
  patient_id: string;
  prediction: AllPredictions;
  feature_importance: FeatureImportance;
  temporal_attention: AttentionStep[];
  metadata: ExplanationMeta;
}
