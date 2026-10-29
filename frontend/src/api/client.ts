import axios from 'axios';
import type { 
  HealthResponse, 
  ModelInfoResponse, 
  PatientHistoryResponse, 
  PredictionResponse, 
  ExplanationResponse 
} from '../types/api';
import {
  mockHealth,
  mockModelInfo,
  getMockPatientData,
  getMockPrediction,
  getMockExplanation
} from '../data/mockPatients';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const isCloudWithoutBackend = 
  typeof window !== 'undefined' && 
  window.location.hostname !== 'localhost' && 
  window.location.hostname !== '127.0.0.1' && 
  !import.meta.env.VITE_API_BASE_URL;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 3000,
});

export const api = {
  checkHealth: async (): Promise<HealthResponse> => {
    if (isCloudWithoutBackend) return mockHealth;
    try {
      const { data } = await apiClient.get<HealthResponse>('/health');
      return data;
    } catch {
      console.warn('Backend server unreachable. Using fallback health status.');
      return mockHealth;
    }
  },
  
  getModelInfo: async (): Promise<ModelInfoResponse> => {
    if (isCloudWithoutBackend) return mockModelInfo;
    try {
      const { data } = await apiClient.get<ModelInfoResponse>('/model/info');
      return data;
    } catch {
      console.warn('Backend server unreachable. Using fallback model info.');
      return mockModelInfo;
    }
  },

  getPatient: async (patientId: string): Promise<PatientHistoryResponse> => {
    if (isCloudWithoutBackend) return getMockPatientData(patientId);
    try {
      const { data } = await apiClient.get<PatientHistoryResponse>(`/patient/${patientId}`);
      return data;
    } catch {
      console.warn(`Backend server unreachable for patient ${patientId}. Using fallback patient data.`);
      return getMockPatientData(patientId);
    }
  },

  predict: async (patientId: string): Promise<PredictionResponse> => {
    if (isCloudWithoutBackend) return getMockPrediction(patientId);
    try {
      const { data } = await apiClient.post<PredictionResponse>('/predict', { patient_id: patientId });
      return data;
    } catch {
      console.warn(`Backend server unreachable for prediction ${patientId}. Using fallback prediction.`);
      return getMockPrediction(patientId);
    }
  },

  explain: async (patientId: string): Promise<ExplanationResponse> => {
    if (isCloudWithoutBackend) return getMockExplanation(patientId);
    try {
      const { data } = await apiClient.post<ExplanationResponse>('/explain', { patient_id: patientId });
      return data;
    } catch {
      console.warn(`Backend server unreachable for explanation ${patientId}. Using fallback explanation.`);
      return getMockExplanation(patientId);
    }
  }
};

export default api;
