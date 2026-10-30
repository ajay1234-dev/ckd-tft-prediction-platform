import { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import {
  Search, Loader2, ArrowRight, TrendingDown, TrendingUp,
  Minus, BrainCircuit, ChevronDown, ChevronUp, Sparkles,
  Clock, AlertTriangle, CheckCircle2, FileText, Printer, Copy, Check, X, Microscope
} from 'lucide-react';
import api from '../api/client';
import { generateAIExplanation } from '../api/openrouter';
import type { PatientHistoryResponse, PredictionResponse, ExplanationResponse, AllPredictions } from '../types/api';

const STAGE_COLORS: Record<string, string> = {
  G1: '#059669', G2: '#10b981', G3a: '#d97706',
  G3b: '#ea580c', G4: '#dc2626', G5: '#7c3aed'
};
const HORIZONS = ['3m', '6m', '12m'];
const HORIZON_LABELS: Record<string, string> = { '3m': '3 Months', '6m': '6 Months', '12m': '12 Months' };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const plotLayout: any = {
  autosize: true,
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: 'Inter, sans-serif', color: '#475569', size: 11 },
  margin: { t: 16, b: 40, l: 48, r: 16 },
  xaxis: { gridcolor: '#f1f5f9', tickfont: { size: 11, color: '#64748b' }, zeroline: false },
  yaxis: { gridcolor: '#f1f5f9', tickfont: { size: 11, color: '#64748b' }, zeroline: false },
};

function StagePill({ stage }: { stage: string }) {
  return (
    <span className="stage-pill" style={{ background: STAGE_COLORS[stage] || '#64748b' }}>
      {stage}
    </span>
  );
}

function RiskGauge({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  const color = pct < 35 ? '#059669' : pct < 65 ? '#d97706' : '#dc2626';
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - pct / 100);

  return (
    <div style={{ textAlign: 'center', padding: '1.25rem 1rem' }}>
      <div style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#64748b', marginBottom: '0.75rem' }}>
        {label}
      </div>
      <svg width="100" height="100" viewBox="0 0 100 100" style={{ display: 'block', margin: '0 auto' }}>
        <circle cx="50" cy="50" r={radius} fill="none" stroke="#f1f5f9" strokeWidth="10" />
        <circle
          cx="50" cy="50" r={radius} fill="none"
          stroke={color} strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%', transition: 'stroke-dashoffset 0.8s ease' }}
        />
        <text x="50" y="46" textAnchor="middle" fill={color} fontSize="17" fontWeight="800" fontFamily="Inter">{pct}%</text>
        <text x="50" y="60" textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="600" fontFamily="Inter">
          {pct < 50 ? 'Low Risk' : pct < 70 ? 'Moderate' : 'High Risk'}
        </text>
      </svg>
    </div>
  );
}

interface KidneyFailureEstimate {
  status: 'imminent' | 'intermediate' | 'late' | 'beyond_12m' | 'low_risk';
  durationText: string;
  detail: string;
  badgeColor: string;
  badgeBg: string;
  borderColor: string;
  g5Prob3m: number;
  g5Prob6m: number;
  g5Prob12m: number;
}

function getKidneyFailureEstimate(pred: AllPredictions): KidneyFailureEstimate {
  const stage3m = pred.stage['3m'];
  const stage6m = pred.stage['6m'];
  const stage12m = pred.stage['12m'];

  const g5Prob3m = stage3m.probabilities['G5'] || 0;
  const g5Prob6m = stage6m.probabilities['G5'] || 0;
  const g5Prob12m = stage12m.probabilities['G5'] || 0;

  if (stage3m.predicted_stage === 'G5' || g5Prob3m >= 0.5) {
    return {
      status: 'imminent',
      durationText: 'Within 0 – 3 Months (Immediate Risk)',
      detail: `The model projects transition to Stage G5 (Kidney Failure) at the 3-month horizon with ${(g5Prob3m * 100).toFixed(1)}% probability. Urgent clinical intervention is indicated.`,
      badgeColor: '#dc2626',
      badgeBg: '#fef2f2',
      borderColor: '#fca5a5',
      g5Prob3m,
      g5Prob6m,
      g5Prob12m,
    };
  }
  if (stage6m.predicted_stage === 'G5' || g5Prob6m >= 0.5) {
    return {
      status: 'intermediate',
      durationText: 'Within 3 – 6 Months',
      detail: `The model projects transition to Stage G5 (Kidney Failure) between 3 and 6 months (${(g5Prob6m * 100).toFixed(1)}% G5 probability at 6m). Rapid renal decline detected.`,
      badgeColor: '#ea580c',
      badgeBg: '#fff7ed',
      borderColor: '#fdba74',
      g5Prob3m,
      g5Prob6m,
      g5Prob12m,
    };
  }
  if (stage12m.predicted_stage === 'G5' || g5Prob12m >= 0.5) {
    return {
      status: 'late',
      durationText: 'Within 6 – 12 Months',
      detail: `The model projects transition to Stage G5 (Kidney Failure) between 6 and 12 months (${(g5Prob12m * 100).toFixed(1)}% G5 probability at 12m). Progression risk is significant.`,
      badgeColor: '#d97706',
      badgeBg: '#fffbeb',
      borderColor: '#fcd34d',
      g5Prob3m,
      g5Prob6m,
      g5Prob12m,
    };
  }
  if (g5Prob12m >= 0.2 || pred.progression['12m'].probability >= 0.6) {
    return {
      status: 'beyond_12m',
      durationText: '> 12 Months (Elevated Risk, Outside 1-Year Horizon)',
      detail: `Stage G5 is not reached within 12 months, but disease progression risk is elevated (${(pred.progression['12m'].probability * 100).toFixed(1)}% at 12m). Potential progression beyond 12 months if untreated.`,
      badgeColor: '#4f46e5',
      badgeBg: '#eef2ff',
      borderColor: '#c7d2fe',
      g5Prob3m,
      g5Prob6m,
      g5Prob12m,
    };
  }
  return {
    status: 'low_risk',
    durationText: 'No Kidney Failure Projected within 12 Months',
    detail: `Patient demonstrates stable renal trajectory across all horizons. Stage G5 probability is negligible (< ${(Math.max(g5Prob3m, g5Prob6m, g5Prob12m) * 100).toFixed(1)}%) and progression probability is low.`,
    badgeColor: '#059669',
    badgeBg: '#ecfdf5',
    borderColor: '#a7f3d0',
    g5Prob3m,
    g5Prob6m,
    g5Prob12m,
  };
}

function KidneyFailureDurationCard({ prediction }: { prediction: AllPredictions }) {
  const estimate = getKidneyFailureEstimate(prediction);

  return (
    <div className="white-card" style={{
      borderLeft: `5px solid ${estimate.badgeColor}`,
      background: 'linear-gradient(to right, #ffffff, #fafafa)',
      marginBottom: '1rem'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: estimate.badgeBg, border: `1px solid ${estimate.borderColor}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            {estimate.status === 'imminent' ? (
              <AlertTriangle size={20} color={estimate.badgeColor} />
            ) : estimate.status === 'low_risk' ? (
              <CheckCircle2 size={20} color={estimate.badgeColor} />
            ) : (
              <Clock size={20} color={estimate.badgeColor} />
            )}
          </div>
          <div>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#64748b' }}>
              Prognostic Time-to-Failure Analysis
            </div>
            <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a' }}>
              Estimated Duration to Kidney Failure (Stage G5)
            </div>
          </div>
        </div>

        <div style={{
          background: estimate.badgeBg,
          border: `1.5px solid ${estimate.borderColor}`,
          color: estimate.badgeColor,
          fontWeight: 800,
          fontSize: '0.95rem',
          padding: '0.45rem 1rem',
          borderRadius: 9999,
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          boxShadow: '0 1px 3px rgba(0,0,0,0.04)'
        }}>
          <Clock size={16} />
          {estimate.durationText}
        </div>
      </div>

      <p style={{ color: '#475569', fontSize: '0.88rem', lineHeight: '1.5', margin: '0.5rem 0 1rem 0' }}>
        {estimate.detail}
      </p>

      {/* Horizon G5 Risk Probability Breakdown */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '0.75rem',
        background: '#f8fafc',
        padding: '0.75rem',
        borderRadius: 10,
        border: '1px solid #e2e8f0'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>3-Month G5 Risk</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 800, color: estimate.g5Prob3m >= 0.5 ? '#dc2626' : '#0f172a' }}>
            {(estimate.g5Prob3m * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ textAlign: 'center', borderLeft: '1px solid #e2e8f0', borderRight: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>6-Month G5 Risk</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 800, color: estimate.g5Prob6m >= 0.5 ? '#dc2626' : '#0f172a' }}>
            {(estimate.g5Prob6m * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>12-Month G5 Risk</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 800, color: estimate.g5Prob12m >= 0.5 ? '#dc2626' : '#0f172a' }}>
            {(estimate.g5Prob12m * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}

function AIExplanationPanel({
  prediction,
  explanation,
  patientId,
}: {
  prediction: PredictionResponse | ExplanationResponse;
  explanation: ExplanationResponse | null;
  patientId: string;
}) {
  const envKey = import.meta.env.VITE_OPENROUTER_API_KEY || localStorage.getItem('openrouter_key') || '';
  const [loading, setLoading] = useState(false);
  const [aiText, setAiText] = useState('');
  const [aiModel, setAiModel] = useState('');
  const [copied, setCopied] = useState(false);
  const textRef = useRef<HTMLDivElement>(null);

  const getTopFeatures = (features?: Record<string, number>, n = 3) => {
    if (!features) return [];
    return Object.entries(features).sort((a, b) => b[1] - a[1]).slice(0, n).map(f => f[0].replace(/_/g, ' '));
  };

  const runExplanation = async () => {
    setLoading(true);
    setAiText('');

    const pred = prediction.prediction;
    const expData = explanation?.feature_importance;

    try {
      const result = await generateAIExplanation(
        envKey,
        {
          patientId,
          stage3m: pred.stage['3m'].predicted_stage,
          stage6m: pred.stage['6m'].predicted_stage,
          stage12m: pred.stage['12m'].predicted_stage,
          egfr3m: pred.egfr['3m'],
          egfr6m: pred.egfr['6m'],
          egfr12m: pred.egfr['12m'],
          prog3m: pred.progression['3m'].probability,
          prog6m: pred.progression['6m'].probability,
          prog12m: pred.progression['12m'].probability,
          topStaticFeatures: expData ? getTopFeatures(expData.static) : undefined,
          topTvFeatures: expData ? getTopFeatures(expData.time_varying) : undefined,
        }
      );
      setAiText(result.text);
      setAiModel(result.model);
    } catch {
      // Handled via fallback
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!aiText) return;
    navigator.clipboard.writeText(aiText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="white-card" style={{ marginTop: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
          <div style={{
            width: 32, height: 32, borderRadius: 10,
            background: 'linear-gradient(135deg, #4f46e5, #0284c7)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 6px rgba(79, 70, 229, 0.25)',
          }}>
            <BrainCircuit size={16} color="white" />
          </div>
          <div>
            <span style={{ fontWeight: 700, fontSize: '0.98rem', color: '#0f172a' }}>Clinical Decision Support Synthesis</span>
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Automated Longitudinal Multi-Horizon Interpretation</div>
          </div>
        </div>

        {aiText && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              onClick={handleCopy}
              className="btn-ghost"
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
            >
              {copied ? <Check size={13} color="#059669" /> : <Copy size={13} />}
              {copied ? 'Copied!' : 'Copy Summary'}
            </button>
            <button
              onClick={() => window.print()}
              className="btn-ghost"
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
            >
              <Printer size={13} /> Print
            </button>
          </div>
        )}
      </div>

      {!aiText && !loading && (
        <div style={{
          background: '#f8fafc', borderRadius: 12, border: '1px dashed #cbd5e1',
          padding: '1.75rem 1.25rem', textAlign: 'center',
        }}>
          <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem' }}>
            Generate a structured clinical interpretation synthesizing the patient's predicted trajectory, underlying causes, and downstream risks.
          </p>
          <button className="btn-primary" onClick={runExplanation} disabled={loading}>
            <Sparkles size={14} /> Generate Clinical Interpretation
          </button>
        </div>
      )}

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1.75rem 0', justifyContent: 'center' }}>
          <Loader2 size={22} color="#4f46e5" className="animate-spin" />
          <span style={{ color: '#475569', fontSize: '0.9rem', fontWeight: 600 }}>
            Synthesizing clinical decision support analysis...
          </span>
        </div>
      )}

      {aiText && (
        <div ref={textRef} style={{
          background: '#ffffff',
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          padding: '1rem 1.25rem',
          maxHeight: '460px',
          overflowY: 'auto',
          boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.02)'
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {aiText.split('\n\n').filter(sec => sec.trim()).map((section, secIdx) => {
              const lines = section.trim().split('\n');
              const headerLine = lines[0] || '';
              const isH3 = headerLine.startsWith('###') || headerLine.startsWith('##') || /^\*\*[^*]+\*\*:?$/.test(headerLine);
              const headerText = isH3
                ? headerLine.replace(/^#{2,3}\s*/, '').replace(/^\*\*(.*?)\*\*:?$/, '$1').trim()
                : null;
              const bodyLines = isH3 ? lines.slice(1) : lines;

              let bg = '#f8fafc';
              let border = '#e2e8f0';
              let badgeColor = '#475569';
              if (headerText?.includes('Problem') || headerText?.includes('1.')) {
                bg = '#fef2f2'; border = '#fecaca'; badgeColor = '#b91c1c';
              } else if (headerText?.includes('Causes') || headerText?.includes('2.')) {
                bg = '#eef2ff'; border = '#c7d2fe'; badgeColor = '#4338ca';
              } else if (headerText?.includes('Downstream') || headerText?.includes('3.')) {
                bg = '#fff7ed'; border = '#fed7aa'; badgeColor = '#c2410c';
              } else if (headerText?.includes('Action') || headerText?.includes('4.')) {
                bg = '#f0fdf4'; border = '#bbf7d0'; badgeColor = '#15803d';
              }

              return (
                <div key={secIdx} style={{
                  background: bg,
                  border: `1px solid ${border}`,
                  borderRadius: 10,
                  padding: '1rem 1.15rem',
                }}>
                  {headerText && (
                    <div style={{
                      fontWeight: 800,
                      fontSize: '0.88rem',
                      color: badgeColor,
                      letterSpacing: '0.01em',
                      marginBottom: '0.65rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}>
                      {headerText}
                    </div>
                  )}

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                    {bodyLines.filter(l => l.trim()).map((line, lineIdx) => {
                      const trimmed = line.trim();
                      const isBullet = trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*');
                      const content = isBullet ? trimmed.replace(/^[•\-\*]\s*/, '') : trimmed;
                      const parts = content.split(/(\*\*.*?\*\*)/g);

                      return (
                        <div key={lineIdx} style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '0.45rem',
                          fontSize: '0.85rem',
                          color: '#334155',
                          lineHeight: 1.6,
                        }}>
                          {isBullet && (
                            <span style={{ color: badgeColor, fontWeight: 'bold', fontSize: '0.9rem', lineHeight: '1.2' }}>
                              •
                            </span>
                          )}
                          <div style={{ flex: 1 }}>
                            {parts.map((p, pIdx) => {
                              if (p.startsWith('**') && p.endsWith('**')) {
                                return <strong key={pIdx} style={{ color: '#0f172a', fontWeight: 700 }}>{p.slice(2, -2)}</strong>;
                              }
                              return p;
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
            <span style={{ fontSize: '0.72rem', color: '#64748b' }}>Generated via: <span style={{ color: '#0f172a', fontWeight: 600 }}>{aiModel || 'Clinical AI Decision Support Engine'}</span></span>
            <button className="btn-ghost" style={{ fontSize: '0.75rem', padding: '0.3rem 0.75rem' }} onClick={runExplanation}>
              <Sparkles size={11} /> Regenerate
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function ClinicalReportModal({
  patientId,
  patientData,
  prediction,
  explanation,
  onClose,
}: {
  patientId: string;
  patientData: PatientHistoryResponse;
  prediction: PredictionResponse | null;
  explanation: ExplanationResponse | null;
  onClose: () => void;
}) {
  const pred = prediction?.prediction;
  const validVisits = patientData.visits.filter(v => !v.is_padding);
  const totalDays = validVisits[validVisits.length - 1]?.days_from_baseline || 0;

  const egfr3m = pred ? pred.egfr['3m'] : null;
  const egfr12m = pred ? pred.egfr['12m'] : null;
  const annualDrop = (egfr3m !== null && egfr12m !== null) ? (egfr3m - egfr12m) : null;
  const isRapid = annualDrop !== null && annualDrop > 5;

  const topStatic = explanation?.feature_importance?.static 
    ? Object.entries(explanation.feature_importance.static).sort((a,b) => b[1]-a[1]).slice(0, 3) 
    : [];
  const topTv = explanation?.feature_importance?.time_varying 
    ? Object.entries(explanation.feature_importance.time_varying).sort((a,b) => b[1]-a[1]).slice(0, 3) 
    : [];

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1.5rem'
    }}>
      <div style={{
        background: '#ffffff', borderRadius: 16, width: '100%', maxWidth: '920px',
        maxHeight: '92vh', overflowY: 'auto', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        display: 'flex', flexDirection: 'column'
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '1.25rem 1.75rem', borderBottom: '1px solid #e2e8f0',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'linear-gradient(to right, #f8fafc, #ffffff)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: 38, height: 38, borderRadius: 10, background: '#4f46e5',
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white'
            }}>
              <FileText size={20} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                Longitudinal Nephrology Consultation Report
              </h2>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                Multi-Horizon Disease Trajectory & Risk Stratification Audit
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              onClick={() => window.print()}
              className="btn-primary"
              style={{ fontSize: '0.8rem', padding: '0.45rem 1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            >
              <Printer size={14} /> Print / Save PDF
            </button>
            <button
              onClick={onClose}
              className="btn-ghost"
              style={{ padding: '0.45rem', borderRadius: 8 }}
            >
              <X size={18} color="#64748b" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Patient Overview Card */}
          <div style={{
            background: '#f8fafc', borderRadius: 12, border: '1px solid #e2e8f0',
            padding: '1.25rem', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem'
          }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Patient Record ID</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#0f172a' }}>#{patientId}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Cohort Split</div>
              <div style={{ fontSize: '1.0rem', fontWeight: 700, color: '#4f46e5' }}>{patientData.split.toUpperCase()}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Observed Timeline</div>
              <div style={{ fontSize: '1.0rem', fontWeight: 700, color: '#0f172a' }}>{totalDays} Days ({validVisits.length} visits)</div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase' }}>Progression Velocity</div>
              <div style={{
                fontSize: '0.95rem', fontWeight: 800,
                color: isRapid ? '#dc2626' : '#059669'
              }}>
                {isRapid ? 'Rapid Decliner' : 'Standard Trajectory'}
              </div>
            </div>
          </div>

          {/* Longitudinal Encounter Record History Table */}
          <div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.65rem' }}>
              Longitudinal Outpatient Encounter History
            </div>
            <div style={{ borderRadius: 10, border: '1px solid #e2e8f0', overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                <thead style={{ background: '#f1f5f9', borderBottom: '1px solid #cbd5e1' }}>
                  <tr>
                    <th style={{ padding: '0.6rem 0.85rem', textAlign: 'left', color: '#475569', fontWeight: 700 }}>Visit #</th>
                    <th style={{ padding: '0.6rem 0.85rem', textAlign: 'left', color: '#475569', fontWeight: 700 }}>Elapsed Days</th>
                    <th style={{ padding: '0.6rem 0.85rem', textAlign: 'left', color: '#475569', fontWeight: 700 }}>Visit Interval (Delta t)</th>
                    <th style={{ padding: '0.6rem 0.85rem', textAlign: 'left', color: '#475569', fontWeight: 700 }}>Encounter Status</th>
                  </tr>
                </thead>
                <tbody>
                  {patientData.visits.map((v, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: v.is_padding ? '#fafafa' : '#ffffff' }}>
                      <td style={{ padding: '0.6rem 0.85rem', fontWeight: 600, color: v.is_padding ? '#94a3b8' : '#0f172a' }}>
                        {v.is_padding ? `Padded step ${v.visit_index + 1}` : `Visit #${v.visit_number}`}
                      </td>
                      <td style={{ padding: '0.6rem 0.85rem', color: '#475569' }}>
                        {v.is_padding ? '—' : `${v.days_from_baseline} days`}
                      </td>
                      <td style={{ padding: '0.6rem 0.85rem', color: '#64748b' }}>
                        {v.is_padding ? '—' : i === 0 ? 'Baseline presentation' : `+${(v.days_from_baseline || 0) - (patientData.visits[i - 1]?.days_from_baseline || 0)} days`}
                      </td>
                      <td style={{ padding: '0.6rem 0.85rem' }}>
                        <span style={{
                          fontSize: '0.68rem', fontWeight: 700, padding: '0.15rem 0.5rem', borderRadius: 9999,
                          background: v.is_padding ? '#f1f5f9' : '#ecfdf5',
                          color: v.is_padding ? '#64748b' : '#047857',
                          border: `1px solid ${v.is_padding ? '#e2e8f0' : '#a7f3d0'}`
                        }}>
                          {v.is_padding ? 'Zero-Padded Step' : 'Valid Encounter'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Multi-Horizon Forecast Matrix */}
          {pred && (
            <div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.65rem' }}>
                TFT Multi-Horizon Prognosis Summary
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.85rem' }}>
                {HORIZONS.map(h => {
                  const stage = pred.stage[h];
                  const egfr = pred.egfr[h];
                  const prog = pred.progression[h];
                  const stgCol = STAGE_COLORS[stage.predicted_stage] || '#64748b';

                  return (
                    <div key={h} style={{
                      background: '#ffffff', border: `1px solid ${stgCol}50`,
                      borderRadius: 12, padding: '1.1rem', textAlign: 'center',
                      boxShadow: '0 2px 6px rgba(0,0,0,0.02)'
                    }}>
                      <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', fontWeight: 700, marginBottom: '0.5rem' }}>
                        {HORIZON_LABELS[h]} Outlook
                      </div>
                      <div style={{ display: 'inline-block', marginBottom: '0.5rem' }}>
                        <StagePill stage={stage.predicted_stage} />
                      </div>
                      <div style={{ fontSize: '0.82rem', color: '#334155', marginBottom: '0.35rem' }}>
                        Stage Probability: <strong style={{ color: stgCol }}>{(stage.probabilities[stage.predicted_stage] * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{ fontSize: '0.82rem', color: '#334155', marginBottom: '0.35rem' }}>
                        eGFR Forecast: <strong>{egfr.toFixed(1)} mL/min</strong>
                      </div>
                      <div style={{ fontSize: '0.82rem', color: '#334155' }}>
                        Progression Risk: <strong style={{ color: prog.probability > 0.6 ? '#dc2626' : '#059669' }}>
                          {(prog.probability * 100).toFixed(1)}%
                        </strong>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Feature Importance Drivers */}
          {(topStatic.length > 0 || topTv.length > 0) && (
            <div style={{
              background: '#f8fafc', borderRadius: 12, border: '1px solid #e2e8f0', padding: '1.25rem'
            }}>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.65rem' }}>
                Key Pathophysiological Drivers (Variable Selection Networks)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '0.82rem' }}>
                <div>
                  <div style={{ fontWeight: 700, color: '#4f46e5', marginBottom: '0.35rem' }}>Top Baseline Demographics:</div>
                  <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#334155' }}>
                    {topStatic.map(([name, val]) => (
                      <li key={name}><strong>{name.replace(/_/g, ' ')}:</strong> {(val * 100).toFixed(1)}% importance</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div style={{ fontWeight: 700, color: '#0284c7', marginBottom: '0.35rem' }}>Top Longitudinal Lab Markers:</div>
                  <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#334155' }}>
                    {topTv.map(([name, val]) => (
                      <li key={name}><strong>{name.replace(/_/g, ' ')}:</strong> {(val * 100).toFixed(1)}% importance</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Clinical Decision Protocol */}
          <div style={{
            background: '#eff6ff', borderRadius: 12, border: '1px solid #bfdbfe',
            padding: '1.25rem'
          }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 800, color: '#1e40af', marginBottom: '0.5rem' }}>
              Actionable Nephrology Care Guidance
            </div>
            <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.82rem', color: '#1e3a8a', lineHeight: 1.7 }}>
              <li><strong>Pharmacotherapy:</strong> Review ACEi/ARB or SGLT2 inhibitor dosing; monitor serum creatinine for acute drops &gt;30%.</li>
              <li><strong>Follow-up Cadence:</strong> Schedule nephrology consultation at 3 months if progression risk exceeds 50%.</li>
              <li><strong>Emergency Prevention:</strong> If Stage G4/G5 trajectory is confirmed, initiate early vascular access planning (AV fistula) to prevent unplanned central venous catheter dialysis crash-landings.</li>
            </ul>
          </div>
        </div>

        {/* Modal Footer */}
        <div style={{
          padding: '1rem 1.75rem', borderTop: '1px solid #e2e8f0',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: '#f8fafc'
        }}>
          <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
            Generated via Explainable Temporal Fusion Transformer Prototype. Not for unguided clinical use.
          </span>
          <button onClick={onClose} className="btn-secondary" style={{ fontSize: '0.8rem', padding: '0.4rem 1.25rem' }}>
            Close Report
          </button>
        </div>
      </div>
    </div>
  );
}

export default function PatientPrediction() {
  const [patientId, setPatientId] = useState('');
  const [loadingPatient, setLoadingPatient] = useState(false);
  const [patientError, setPatientError] = useState<string | null>(null);
  const [patientData, setPatientData] = useState<PatientHistoryResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | ExplanationResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!patientId.trim()) return;
    setLoadingPatient(true);
    setPatientError(null);
    setPatientData(null);
    setPrediction(null);
    setExplanation(null);
    try {
      setPatientData(await api.getPatient(patientId));
    } catch (err: any) {
      setPatientError(err.response?.status === 404 ? `Patient '${patientId}' not found.` : 'Failed to load patient.');
    } finally {
      setLoadingPatient(false);
    }
  };

  const handlePredict = async () => {
    if (!patientId) return;
    setPredicting(true);
    setPredictionError(null);
    try {
      const [pred, exp] = await Promise.all([
        api.predict(patientId),
        api.explain(patientId),
      ]);
      setPrediction(pred);
      setExplanation(exp as ExplanationResponse);
    } catch (err: any) {
      setPredictionError(err.message || 'Prediction failed.');
    } finally {
      setPredicting(false);
    }
  };

  const egfrValues = prediction
    ? HORIZONS.map(h => prediction.prediction.egfr[h])
    : [];

  const egfrTrend = egfrValues.length === 3
    ? egfrValues[2] > egfrValues[0] ? 'up' : egfrValues[2] < egfrValues[0] ? 'down' : 'flat'
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* ── Page Hero ─────────────────────────────────────── */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(79,70,229,0.06) 0%, rgba(2,132,199,0.04) 100%)',
        border: '1px solid rgba(165,180,252,0.3)',
        borderRadius: 20,
        padding: '1.75rem 2rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1.5rem',
        flexWrap: 'wrap',
        animation: 'fadeInUp 0.4s ease both',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
            <div style={{
              background: 'linear-gradient(135deg, #4f46e5, #0284c7)',
              borderRadius: 10,
              padding: '0.4rem',
              display: 'flex',
            }}>
              <Microscope size={18} color="#ffffff" />
            </div>
            <span style={{
              fontSize: '0.7rem', fontWeight: 800, letterSpacing: '0.1em',
              textTransform: 'uppercase', color: '#4338ca',
              background: '#eef2ff', border: '1px solid #c7d2fe',
              padding: '0.2rem 0.6rem', borderRadius: 9999,
            }}>Prediction Engine</span>
          </div>
          <h1 className="page-title" style={{ marginBottom: '0.4rem' }}>
            Patient Progression Forecasting
          </h1>
          <p style={{ color: '#64748b', fontSize: '0.88rem', margin: 0, maxWidth: 520 }}>
            Multi-horizon TFT predictions of CKD stage, eGFR trajectory, and kidney failure risk.
            Enter a patient ID from the test cohort (60001–70000) to begin.
          </p>
        </div>
        {/* Mini metric summary */}
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {[
            { label: '3 Horizons', desc: '3m · 6m · 12m', bg: '#eef2ff', border: '#c7d2fe', color: '#4338ca' },
            { label: 'eGFR Trend', desc: 'mL/min/1.73m²', bg: '#f0f9ff', border: '#bae6fd', color: '#0369a1' },
            { label: 'Stage G1→G5', desc: 'KDIGO scale', bg: '#ecfdf5', border: '#a7f3d0', color: '#047857' },
          ].map(item => (
            <div key={item.label} style={{
              background: item.bg, border: `1px solid ${item.border}`,
              borderRadius: 14, padding: '0.75rem 1rem', minWidth: 120,
            }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.2rem' }}>{item.label}</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Search Input Card ──────────────────────────── */}
      <div className="white-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <Search size={16} color="#4f46e5" />
          <div className="section-title" style={{ marginBottom: 0 }}>Patient Lookup — Enter Test Cohort ID</div>
        </div>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1, maxWidth: '460px' }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '0.875rem', top: '50%', transform: 'translateY(-50%)' }} />
            <input
              className="input-field"
              type="text"
              value={patientId}
              onChange={e => setPatientId(e.target.value)}
              placeholder="Enter patient ID (e.g. 60001, 60002)"
              style={{ paddingLeft: '2.5rem' }}
            />
          </div>
          <button className="btn-primary" type="submit" disabled={loadingPatient || !patientId.trim()}>
            {loadingPatient && <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />}
            Load Patient
          </button>
        </form>
        {patientError && (
          <div style={{
            marginTop: '0.75rem', padding: '0.6rem 0.875rem',
            background: '#fef2f2', border: '1px solid #fecaca',
            borderRadius: 10, fontSize: '0.85rem', color: '#dc2626',
            display: 'flex', alignItems: 'center', gap: '0.4rem',
          }}>⚠ {patientError}</div>
        )}
      </div>

      {patientData && (
        <>
          {/* Patient Record Summary */}
          <div className="white-card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div className="section-title" style={{ marginBottom: 0 }}>
                Patient Demographic & Sequence Summary
              </div>
              <span style={{
                fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase',
                color: '#4f46e5', background: '#eef2ff', border: '1px solid #c7d2fe', padding: '0.2rem 0.65rem', borderRadius: 9999,
              }}>
                {patientData.split} Cohort
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.875rem', marginBottom: '1.25rem' }}>
              {[
                { label: 'Patient ID', value: patientData.patient_id },
                { label: 'Recorded Visits', value: patientData.n_visits },
                { label: 'Sequence Window', value: `${patientData.sequence_length} steps` },
                { label: 'Cohort Split', value: patientData.split.toUpperCase() },
              ].map(s => (
                <div key={s.label} className="stat-card">
                  <div className="stat-value">{s.value}</div>
                  <div className="stat-label">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Visit timeline dropdown */}
            <button
              className="btn-ghost"
              style={{ fontSize: '0.8rem', width: '100%', justifyContent: 'center' }}
              onClick={() => setHistoryOpen(!historyOpen)}
            >
              {historyOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {historyOpen ? 'Hide' : 'Show'} Longitudinal Clinical History ({patientData.n_visits} visits)
            </button>

            {historyOpen && (
              <div style={{ marginTop: '1rem', overflowX: 'auto', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
                  <thead style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <tr>
                      {['Visit Number', 'Days from Baseline', 'Visit Status'].map(h => (
                        <th key={h} style={{ padding: '0.625rem 1rem', textAlign: 'left', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', fontSize: '0.72rem', textTransform: 'uppercase' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {patientData.visits.filter(v => !v.is_padding).map((v, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#ffffff' : '#fafafa' }}>
                        <td style={{ padding: '0.625rem 1rem', color: '#0f172a', fontWeight: 600 }}>Visit #{v.visit_number}</td>
                        <td style={{ padding: '0.625rem 1rem', color: '#475569' }}>{v.days_from_baseline} days</td>
                        <td style={{ padding: '0.625rem 1rem' }}>
                          <span style={{ fontSize: '0.7rem', color: '#059669', background: '#ecfdf5', border: '1px solid #a7f3d0', padding: '0.15rem 0.55rem', borderRadius: 9999, fontWeight: 600 }}>
                            Recorded
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Action Trigger */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <button
              className="btn-primary"
              style={{ padding: '0.8rem 2.5rem', fontSize: '0.98rem' }}
              onClick={handlePredict}
              disabled={predicting}
            >
              {predicting
                ? <><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Forecasting Outcomes…</>
                : <><Sparkles size={18} /> Run Full Prediction & Analysis</>
              }
            </button>

            {prediction && (
              <button
                type="button"
                className="btn-secondary"
                style={{
                  padding: '0.8rem 1.75rem',
                  fontSize: '0.95rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  background: '#ffffff',
                  border: '1.5px solid #cbd5e1',
                  color: '#1e293b',
                  fontWeight: 700,
                  boxShadow: '0 1px 3px rgba(0,0,0,0.04)'
                }}
                onClick={() => setShowReportModal(true)}
              >
                <FileText size={18} color="#4f46e5" />
                Comprehensive Consultation Report
              </button>
            )}
          </div>

          {predictionError && (
            <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 12, padding: '1rem', color: '#b91c1c', fontSize: '0.875rem' }}>
              {predictionError}
            </div>
          )}
        </>
      )}

      {prediction && (
        <>
          {/* Kidney Failure Prognosis & Duration Card */}
          <KidneyFailureDurationCard prediction={prediction.prediction} />

          {/* Stage Progression Timeline */}
          <div className="white-card">
            <div className="section-title">CKD Stage Multi-Horizon Forecast</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
              {HORIZONS.map((h) => {
                const d = prediction.prediction.stage[h];
                const col = STAGE_COLORS[d.predicted_stage];
                const prob = d.probabilities[d.predicted_stage];
                return (
                  <div key={h} style={{
                    background: '#ffffff',
                    border: `1px solid ${col}40`,
                    borderRadius: 14,
                    padding: '1.25rem',
                    textAlign: 'center',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)',
                  }}>
                    <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: '0.75rem' }}>
                      {HORIZON_LABELS[h]}
                    </div>
                    <StagePill stage={d.predicted_stage} />
                    <div style={{ marginTop: '0.75rem', fontSize: '1.6rem', fontWeight: 800, color: col }}>
                      {(prob * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 500 }}>model confidence</div>
                    <div style={{
                      marginTop: '0.75rem', height: 5, background: '#f1f5f9', borderRadius: 9999, overflow: 'hidden',
                    }}>
                      <div style={{ height: '100%', width: `${prob * 100}%`, background: col, borderRadius: 9999, transition: 'width 0.8s ease' }} />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Stage Transition Flow Row */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1.25rem',
              padding: '1rem', background: '#f8fafc', borderRadius: 12, border: '1px solid #e2e8f0',
            }}>
              {HORIZONS.map((h, i) => (
                <div key={h} style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, marginBottom: '0.25rem' }}>{h}</div>
                    <StagePill stage={prediction.prediction.stage[h].predicted_stage} />
                  </div>
                  {i < 2 && <ArrowRight size={18} color="#94a3b8" />}
                </div>
              ))}
            </div>
          </div>

          {/* eGFR Trajectory + Progression Risk Gauges */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            {/* eGFR Trajectory */}
            <div className="white-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <div className="section-title" style={{ marginBottom: 0 }}>Predicted eGFR Trajectory</div>
                {egfrTrend && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '0.3rem',
                    fontSize: '0.78rem', fontWeight: 700,
                    color: egfrTrend === 'up' ? '#059669' : egfrTrend === 'down' ? '#dc2626' : '#64748b',
                  }}>
                    {egfrTrend === 'up' && <TrendingUp size={14} />}
                    {egfrTrend === 'down' && <TrendingDown size={14} />}
                    {egfrTrend === 'flat' && <Minus size={14} />}
                    {egfrTrend === 'up' ? 'Improving' : egfrTrend === 'down' ? 'Declining' : 'Stable'}
                  </div>
                )}
              </div>
              <div style={{ height: 230 }}>
                <Plot
                  data={[{
                    x: ['3 Months', '6 Months', '12 Months'],
                    y: egfrValues,
                    type: 'scatter', mode: 'lines+markers',
                    name: 'eGFR Forecast',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(79, 70, 229, 0.08)',
                    line: { color: '#4f46e5', width: 3, shape: 'spline' },
                    marker: { size: 10, color: '#4f46e5', line: { color: '#ffffff', width: 2.5 } },
                    hovertemplate: '<b>%{x}</b><br>Forecast eGFR: <b>%{y:.1f}</b> mL/min/1.73m²<extra></extra>',
                  }]}
                  layout={{
                    ...plotLayout,
                    height: 220,
                    margin: { t: 14, b: 36, l: 48, r: 14 },
                    yaxis: { 
                      ...plotLayout.yaxis, 
                      title: { text: 'eGFR (mL/min/1.73m²)', font: { size: 10.5, color: '#64748b' } },
                      zeroline: false
                    },
                    shapes: [
                      {
                        type: 'line',
                        xref: 'paper',
                        x0: 0,
                        x1: 1,
                        yref: 'y',
                        y0: 15,
                        y1: 15,
                        line: { color: '#ef4444', width: 1.5, dash: 'dash' }
                      }
                    ],
                    annotations: [
                      {
                        xref: 'paper',
                        x: 0.98,
                        yref: 'y',
                        y: 15,
                        text: 'Kidney Failure (G5: <15)',
                        showarrow: false,
                        xanchor: 'right',
                        yanchor: 'bottom',
                        font: { size: 9, color: '#dc2626', family: 'Inter, sans-serif' }
                      }
                    ]
                  }}
                  useResizeHandler style={{ width: '100%', height: '100%' }}
                  config={{ displayModeBar: false }}
                />
              </div>
            </div>

            {/* Progression Risk */}
            <div className="white-card">
              <div className="section-title">Progression Risk Assessment</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)' }}>
                {HORIZONS.map(h => (
                  <RiskGauge
                    key={h}
                    value={prediction.prediction.progression[h].probability}
                    label={HORIZON_LABELS[h]}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Full Stage Probability Distribution */}
          <div className="white-card">
            <div className="section-title">Multi-Horizon Stage Probability Distribution</div>
            <div style={{ height: 240 }}>
              <Plot
                data={HORIZONS.map((h, i) => ({
                  x: Object.keys(prediction.prediction.stage[h].probabilities),
                  y: Object.values(prediction.prediction.stage[h].probabilities).map(v => +(v * 100).toFixed(2)),
                  type: 'bar' as const,
                  name: HORIZON_LABELS[h],
                  marker: {
                    color: i === 0 ? '#4f46e5' : i === 1 ? '#0284c7' : '#059669',
                    line: { color: 'transparent' }
                  },
                  hovertemplate: '<b>%{fullData.name} - Stage %{x}</b><br>Stage Confidence: <b>%{y:.1f}%</b><extra></extra>',
                }))}
                layout={{
                  ...plotLayout,
                  barmode: 'group',
                  height: 230,
                  margin: { t: 8, b: 36, l: 48, r: 12 },
                  legend: { orientation: 'h', x: 0, y: -0.3, font: { color: '#475569', size: 11 } },
                  xaxis: { ...plotLayout.xaxis, title: { text: 'KDIGO CKD Stage', font: { size: 10.5, color: '#64748b' } } },
                  yaxis: { ...plotLayout.yaxis, title: { text: 'Stage Probability (%)', font: { size: 10.5, color: '#64748b' } } },
                }}
                useResizeHandler style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </div>
          </div>

          {/* AI Clinical Interpretation Panel */}
          <AIExplanationPanel
            prediction={prediction}
            explanation={explanation}
            patientId={patientId}
          />
        </>
      )}

      {showReportModal && patientData && (
        <ClinicalReportModal
          patientId={patientId}
          patientData={patientData}
          prediction={prediction as PredictionResponse | null}
          explanation={explanation}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
}
