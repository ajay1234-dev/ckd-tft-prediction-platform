import { useState } from 'react';
import Plot from 'react-plotly.js';
import { Search, Loader2, BrainCircuit, BarChart2, Clock, Lightbulb, Activity, TrendingUp } from 'lucide-react';
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* ── Page Hero ───────────────────────────────────── */}
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
              <BrainCircuit size={18} color="#ffffff" />
            </div>
            <span className="badge badge-indigo">XAI Module</span>
          </div>
          <h1 className="page-title" style={{ marginBottom: '0.4rem' }}>
            Clinical Interpretability
          </h1>
          <p style={{ color: '#64748b', fontSize: '0.88rem', margin: 0, maxWidth: 500 }}>
            Explore which biomarkers and historical visits drive the TFT model's progression predictions — full SHAP-equivalent feature attribution.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {[
            { icon: <BarChart2 size={16} color="#4f46e5" />, label: 'Static Drivers', desc: 'Baseline demographics', bg: '#eef2ff', border: '#c7d2fe' },
            { icon: <Activity size={16} color="#0284c7" />, label: 'Lab Trends', desc: 'Time-varying markers', bg: '#f0f9ff', border: '#bae6fd' },
            { icon: <Clock size={16} color="#059669" />, label: 'Visit Attention', desc: 'Temporal focus weights', bg: '#ecfdf5', border: '#a7f3d0' },
          ].map(item => (
            <div key={item.label} style={{
              background: item.bg,
              border: `1px solid ${item.border}`,
              borderRadius: 14,
              padding: '0.75rem 1rem',
              minWidth: 130,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
                {item.icon}
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a' }}>{item.label}</span>
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Search Bar ──────────────────────────────────── */}
      <div className="white-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <Lightbulb size={16} color="#d97706" />
          <div className="section-title" style={{ marginBottom: 0 }}>Patient Lookup — Enter ID to Analyse Drivers</div>
        </div>
        <form onSubmit={handleExplain} style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{ position: 'relative', flex: 1, maxWidth: '460px' }}>
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
        {error && (
          <div style={{
            marginTop: '0.75rem',
            padding: '0.6rem 0.875rem',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 10,
            fontSize: '0.85rem',
            color: '#dc2626',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
          }}>⚠ {error}</div>
        )}
      </div>

      {explanation && (
        <>
          {/* Key Insights Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
            {/* Dominant Baseline Driver */}
            <div className="white-card" style={{ padding: '1.5rem', borderTop: '3px solid #4f46e5' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <div style={{ background: '#eef2ff', borderRadius: 8, padding: '0.35rem', display: 'flex' }}>
                  <BarChart2 size={15} color="#4f46e5" />
                </div>
                <div style={{ fontSize: '0.68rem', color: '#4338ca', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 800 }}>
                  Dominant Baseline Driver
                </div>
              </div>
              <div style={{ fontWeight: 900, fontSize: '1.15rem', color: '#1e1b4b', marginBottom: '0.3rem', lineHeight: 1.2 }}>
                {staticFeats[0]?.[0]?.replace(/_/g, ' ') || '—'}
              </div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.2rem 0.65rem',
                borderRadius: 9999,
                background: '#eef2ff',
                border: '1px solid #c7d2fe',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#4f46e5',
              }}>
                {staticFeats[0] ? (staticFeats[0][1] * 100).toFixed(1) + '% importance' : ''}
              </div>
            </div>

            {/* Top Longitudinal Marker */}
            <div className="white-card" style={{ padding: '1.5rem', borderTop: '3px solid #0284c7' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <div style={{ background: '#f0f9ff', borderRadius: 8, padding: '0.35rem', display: 'flex' }}>
                  <TrendingUp size={15} color="#0284c7" />
                </div>
                <div style={{ fontSize: '0.68rem', color: '#0369a1', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 800 }}>
                  Top Longitudinal Marker
                </div>
              </div>
              <div style={{ fontWeight: 900, fontSize: '1.15rem', color: '#082f49', marginBottom: '0.3rem', lineHeight: 1.2 }}>
                {tvFeats[0]?.[0]?.replace(/_/g, ' ') || '—'}
              </div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.2rem 0.65rem',
                borderRadius: 9999,
                background: '#f0f9ff',
                border: '1px solid #bae6fd',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#0284c7',
              }}>
                {tvFeats[0] ? (tvFeats[0][1] * 100).toFixed(1) + '% importance' : ''}
              </div>
            </div>

            {/* Most Attended Visit */}
            <div className="white-card" style={{ padding: '1.5rem', borderTop: '3px solid #059669' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <div style={{ background: '#ecfdf5', borderRadius: 8, padding: '0.35rem', display: 'flex' }}>
                  <Clock size={15} color="#059669" />
                </div>
                <div style={{ fontSize: '0.68rem', color: '#047857', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 800 }}>
                  Most Attended Visit
                </div>
              </div>
              <div style={{ fontWeight: 900, fontSize: '1.15rem', color: '#064e3b', marginBottom: '0.3rem', lineHeight: 1.2 }}>
                {validAttention.length
                  ? `Visit #${validAttention.sort((a, b) => b.attention_weight - a.attention_weight)[0].visit_number}`
                  : '—'
                }
              </div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.2rem 0.65rem',
                borderRadius: 9999,
                background: '#ecfdf5',
                border: '1px solid #a7f3d0',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#059669',
              }}>
                {validAttention.length
                  ? (validAttention.sort((a, b) => b.attention_weight - a.attention_weight)[0].attention_weight * 100).toFixed(1) + '% attention focus'
                  : ''
                }
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
                      return v > 0.2 ? '#059669' : v > 0.1 ? '#0284c7' : '#4f46e5';
                    }),
                    line: { color: 'transparent' },
                  },
                  hovertemplate: '<b>%{x}</b><br>Temporal Attention Focus: <b>%{y:.2f}%</b><extra></extra>',
                }]}
                layout={{
                  ...plotLayout,
                  height: 220,
                  margin: { t: 12, b: 48, l: 52, r: 12 },
                  xaxis: { ...plotLayout.xaxis, tickangle: -15, tickfont: { size: 10, color: '#475569' } },
                  yaxis: {
                    ...plotLayout.yaxis,
                    title: { text: 'Attention Weight (%)', font: { size: 10.5, color: '#64748b' } },
                  },
                  shapes: validAttention.length > 0 ? [
                    {
                      type: 'line',
                      xref: 'paper',
                      x0: 0,
                      x1: 1,
                      yref: 'y',
                      y0: +(100 / validAttention.length).toFixed(1),
                      y1: +(100 / validAttention.length).toFixed(1),
                      line: { color: '#94a3b8', width: 1.5, dash: 'dot' }
                    }
                  ] : [],
                  annotations: validAttention.length > 0 ? [
                    {
                      xref: 'paper',
                      x: 0.98,
                      yref: 'y',
                      y: +(100 / validAttention.length).toFixed(1),
                      text: `Uniform Baseline (${(100 / validAttention.length).toFixed(1)}%)`,
                      showarrow: false,
                      xanchor: 'right',
                      yanchor: 'bottom',
                      font: { size: 9, color: '#64748b', family: 'Inter, sans-serif' }
                    }
                  ] : []
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
