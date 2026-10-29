import { useState } from 'react';
import Plot from 'react-plotly.js';
import { Search, Loader2, BrainCircuit, BarChart2, Clock } from 'lucide-react';
import api from '../api/client';
import type { ExplanationResponse } from '../types/api';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const plotLayout: any = {
  autosize: true,
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: 'Inter, sans-serif', color: '#475569', size: 11 },
  margin: { t: 8, b: 40, l: 160, r: 8 },
  xaxis: { gridcolor: '#f1f5f9', tickfont: { size: 11, color: '#64748b' }, zeroline: false },
  yaxis: { gridcolor: '#f1f5f9', tickfont: { size: 11, color: '#64748b' }, zeroline: false },
};

function FeatureBar({ name, value, max, color }: { name: string; value: number; max: number; color: string }) {
  const pct = (value / max) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.625rem' }}>
      <div style={{ width: 150, fontSize: '0.8rem', fontWeight: 500, color: '#334155', textAlign: 'right', flexShrink: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {name.replace(/_/g, ' ')}
      </div>
      <div style={{ flex: 1, height: 7, background: '#f1f5f9', borderRadius: 9999, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 9999,
          background: color, transition: 'width 0.7s ease',
        }} />
      </div>
      <div style={{ width: 44, fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textAlign: 'right', flexShrink: 0 }}>
        {(value * 100).toFixed(1)}%
      </div>
    </div>
  );
}

export default function Explainability() {
  const [patientId, setPatientId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);

  const handleExplain = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!patientId.trim()) return;
    setLoading(true);
    setError(null);
    setExplanation(null);
    try {
      setExplanation(await api.explain(patientId) as ExplanationResponse);
    } catch (err: any) {
      setError(err.response?.status === 404 ? `Patient '${patientId}' not found.` : 'Failed to load explainability data.');
    } finally {
      setLoading(false);
    }
  };

  const topN = (features: Record<string, number>, n = 10) =>
    Object.entries(features).sort((a, b) => b[1] - a[1]).slice(0, n);

  const staticFeats = explanation ? topN(explanation.feature_importance.static) : [];
  const tvFeats = explanation ? topN(explanation.feature_importance.time_varying) : [];
  const maxStatic = staticFeats[0]?.[1] || 1;
  const maxTv = tvFeats[0]?.[1] || 1;
  const validAttention = explanation?.temporal_attention.filter(a => !a.is_padding) || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.02em', marginBottom: '0.25rem' }}>
          Clinical Interpretability & Progression Drivers
        </h1>
        <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
          Key clinical biomarker importance and historical visit attention analysis.
        </p>
      </div>

      {/* Search Bar */}
      <div className="white-card">
        <div className="section-title">Patient Lookup</div>
        <form onSubmit={handleExplain} style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1, maxWidth: '420px' }}>
            <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '0.875rem', top: '50%', transform: 'translateY(-50%)' }} />
            <input
              className="input-field"
              type="text"
              value={patientId}
              onChange={e => setPatientId(e.target.value)}
              placeholder="Enter patient ID (e.g. 60001)"
              style={{ paddingLeft: '2.5rem' }}
            />
          </div>
          <button className="btn-primary" type="submit" disabled={loading || !patientId.trim()}>
            {loading ? <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <BrainCircuit size={15} />}
            Analyse Drivers
          </button>
        </form>
        {error && <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: '#dc2626' }}>{error}</div>}
      </div>

      {explanation && (
        <>
          {/* Key Insights Summary Cards */}
          <div className="white-card">
            <div className="section-title">Core Interpretability Insights</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <div style={{ background: '#eef2ff', borderRadius: 12, padding: '1.15rem', border: '1px solid #c7d2fe' }}>
                <div style={{ fontSize: '0.72rem', color: '#4338ca', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700, marginBottom: '0.4rem' }}>
                  Dominant Baseline Driver
                </div>
                <div style={{ fontWeight: 800, fontSize: '1.05rem', color: '#1e1b4b' }}>
                  {staticFeats[0]?.[0]?.replace(/_/g, ' ') || '—'}
                </div>
                <div style={{ fontSize: '0.78rem', color: '#4f46e5', fontWeight: 600, marginTop: '0.25rem' }}>
                  {staticFeats[0] ? (staticFeats[0][1] * 100).toFixed(1) + '% importance' : ''}
                </div>
              </div>

              <div style={{ background: '#f0f9ff', borderRadius: 12, padding: '1.15rem', border: '1px solid #bae6fd' }}>
                <div style={{ fontSize: '0.72rem', color: '#0369a1', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700, marginBottom: '0.4rem' }}>
                  Top Longitudinal Marker
                </div>
                <div style={{ fontWeight: 800, fontSize: '1.05rem', color: '#082f49' }}>
                  {tvFeats[0]?.[0]?.replace(/_/g, ' ') || '—'}
                </div>
                <div style={{ fontSize: '0.78rem', color: '#0284c7', fontWeight: 600, marginTop: '0.25rem' }}>
                  {tvFeats[0] ? (tvFeats[0][1] * 100).toFixed(1) + '% importance' : ''}
                </div>
              </div>

              <div style={{ background: '#ecfdf5', borderRadius: 12, padding: '1.15rem', border: '1px solid #a7f3d0' }}>
                <div style={{ fontSize: '0.72rem', color: '#047857', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700, marginBottom: '0.4rem' }}>
                  Most Attended Visit
                </div>
                <div style={{ fontWeight: 800, fontSize: '1.05rem', color: '#064e3b' }}>
                  {validAttention.length
                    ? `Visit #${validAttention.sort((a, b) => b.attention_weight - a.attention_weight)[0].visit_number}`
                    : '—'
                  }
                </div>
                <div style={{ fontSize: '0.78rem', color: '#059669', fontWeight: 600, marginTop: '0.25rem' }}>
                  {validAttention.length
                    ? (validAttention.sort((a, b) => b.attention_weight - a.attention_weight)[0].attention_weight * 100).toFixed(1) + '% attention focus'
                    : ''
                  }
                </div>
              </div>
            </div>
          </div>

          {/* Feature Importance Side-by-Side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            {/* Static Drivers */}
            <div className="white-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
                <BarChart2 size={16} color="#4f46e5" />
                <span className="section-title" style={{ marginBottom: 0 }}>Baseline Demographic & Risk Factors</span>
              </div>
              <div style={{ maxHeight: '360px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                {staticFeats.map(([name, value]) => (
                  <FeatureBar key={name} name={name} value={value} max={maxStatic} color="#4f46e5" />
                ))}
              </div>
            </div>

            {/* Time-Varying Drivers */}
            <div className="white-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
                <BarChart2 size={16} color="#0284c7" />
                <span className="section-title" style={{ marginBottom: 0 }}>Longitudinal Biomarker & Lab Drivers</span>
              </div>
              <div style={{ maxHeight: '360px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                {tvFeats.map(([name, value]) => (
                  <FeatureBar key={name} name={name} value={value} max={maxTv} color="#0284c7" />
                ))}
              </div>
            </div>
          </div>

          {/* Temporal Attention Chart */}
          <div className="white-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
              <Clock size={16} color="#059669" />
              <div className="section-title" style={{ marginBottom: 0 }}>Historical Visit Attention & Focus</div>
            </div>
            <p style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '1rem' }}>
              Historical visit focus: which past visits the clinical model weighted most heavily to predict progression.
            </p>
            <div style={{ height: 230 }}>
              <Plot
                data={[{
                  x: validAttention.map(a => `Visit #${a.visit_number} (${a.days_from_baseline}d)`),
                  y: validAttention.map(a => +(a.attention_weight * 100).toFixed(3)),
                  type: 'bar',
                  marker: {
                    color: validAttention.map(a => {
                      const v = a.attention_weight;
                      return v > 0.2 ? '#059669' : v > 0.1 ? '#d97706' : '#4f46e5';
                    }),
                    line: { color: 'transparent' },
                  },
                  hovertemplate: '<b>%{x}</b><br>Attention Focus: %{y:.2f}%<extra></extra>',
                }]}
                layout={{
                  ...plotLayout,
                  height: 220,
                  margin: { t: 8, b: 48, l: 48, r: 8 },
                  xaxis: { ...plotLayout.xaxis, tickangle: -20, tickfont: { size: 10, color: '#475569' } },
                  yaxis: {
                    ...plotLayout.yaxis,
                    title: { text: 'Attention (%)', font: { size: 10, color: '#64748b' } },
                  },
                }}
                useResizeHandler style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
