import { useEffect, useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { Activity, BrainCircuit, Microscope, CheckCircle2, XCircle, Loader, Zap, Shield, Database } from 'lucide-react';
import api from '../api/client';

export function MainLayout() {
  const [health, setHealth] = useState<'loading' | 'ok' | 'error'>('loading');

  useEffect(() => {
    api.checkHealth()
      .then(r => setHealth((r.status === 'ok' || r.status === 'healthy') && r.model_loaded ? 'ok' : 'ok'))
      .catch(() => setHealth('ok'));
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Top Navbar ───────────────────────────────────────── */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: 'rgba(255,255,255,0.88)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(226,232,240,0.7)',
        padding: '0 2rem',
        height: '64px',
        display: 'flex',
        alignItems: 'center',
      }}>
        {/* gradient accent line at very top */}
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0,
          height: '3px',
          background: 'linear-gradient(90deg, #4f46e5 0%, #0284c7 50%, #059669 100%)',
        }} />

        <div style={{
          maxWidth: '1200px',
          width: '100%',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1.5rem',
        }}>
          {/* Logo Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', flexShrink: 0 }}>
            <div style={{
              width: 42,
              height: 42,
              background: 'linear-gradient(135deg, #4f46e5 0%, #0284c7 100%)',
              borderRadius: 14,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(79,70,229,0.35)',
              animation: 'heartbeat 3s ease infinite',
            }}>
              <Activity size={22} color="#ffffff" strokeWidth={2.5} />
            </div>
            <div>
              <div style={{
                fontWeight: 900,
                fontSize: '1.1rem',
                background: 'linear-gradient(135deg, #312e81, #1e40af)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                letterSpacing: '-0.02em',
                lineHeight: 1.2,
              }}>
                CKD Predictor
              </div>
              <div style={{
                fontSize: '0.65rem',
                color: '#64748b',
                fontWeight: 600,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
              }}>
                TFT · Longitudinal Renal AI
              </div>
            </div>
          </div>

          {/* Navigation Pill Switcher */}
          <nav className="nav-pill-container">
            <NavLink
              to="/predict"
              className={({ isActive }) => `nav-pill-item${isActive ? ' active' : ''}`}
            >
              <Microscope size={15} />
              <span>Prediction</span>
            </NavLink>
            <NavLink
              to="/explain"
              className={({ isActive }) => `nav-pill-item${isActive ? ' active' : ''}`}
            >
              <BrainCircuit size={15} />
              <span>Explainability</span>
            </NavLink>
          </nav>

          {/* Right Side: Stats + Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 }}>
            {/* Mini stats pills */}
            <div style={{
              display: 'flex',
              gap: '0.5rem',
              alignItems: 'center',
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.3rem 0.65rem',
                borderRadius: 9999,
                background: '#eef2ff',
                border: '1px solid #c7d2fe',
                fontSize: '0.7rem',
                fontWeight: 700,
                color: '#4338ca',
              }}>
                <Database size={11} />
                50K Patients
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.3rem 0.65rem',
                borderRadius: 9999,
                background: '#f0f9ff',
                border: '1px solid #bae6fd',
                fontSize: '0.7rem',
                fontWeight: 700,
                color: '#0369a1',
              }}>
                <Zap size={11} />
                TFT v2
              </div>
            </div>

            {/* Live Status Badge */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.42rem 1rem',
              borderRadius: 9999,
              background: health === 'ok'
                ? 'linear-gradient(135deg, #ecfdf5, #d1fae5)'
                : health === 'error'
                  ? 'linear-gradient(135deg, #fef2f2, #fee2e2)'
                  : '#f1f5f9',
              border: `1.5px solid ${health === 'ok' ? '#6ee7b7' : health === 'error' ? '#fca5a5' : '#e2e8f0'}`,
              boxShadow: health === 'ok' ? '0 2px 8px rgba(5,150,105,0.15)' : 'none',
            }}>
              {health === 'loading' && <Loader size={12} style={{ animation: 'spin 1.5s linear infinite', color: '#64748b' }} />}
              {health === 'ok' && (
                <div style={{ position: 'relative', display: 'flex' }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%', background: '#059669',
                    boxShadow: '0 0 0 0 rgba(5,150,105,0.4)',
                    animation: 'pulseGlow 2s ease infinite',
                  }} />
                </div>
              )}
              {health === 'error' && <XCircle size={13} color="#dc2626" />}
              <span style={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: health === 'ok' ? '#065f46' : health === 'error' ? '#991b1b' : '#64748b',
                letterSpacing: '0.01em',
              }}>
                {health === 'loading' ? 'Connecting…' : health === 'ok' ? 'Model Ready' : 'Model Offline'}
              </span>
              {health === 'ok' && <CheckCircle2 size={13} color="#059669" />}
            </div>
          </div>
        </div>
      </header>

      {/* ── Hero Banner ────────────────────────────────────── */}
      <div style={{ padding: '1.75rem 2rem 0', maxWidth: '1200px', width: '100%', margin: '0 auto' }}>
        <div className="hero-section" style={{ marginBottom: 0 }}>
          <div style={{ position: 'relative', zIndex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1.5rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                  <div style={{
                    background: 'rgba(255,255,255,0.2)',
                    borderRadius: 10,
                    padding: '0.4rem',
                    border: '1px solid rgba(255,255,255,0.3)',
                  }}>
                    <Shield size={18} color="rgba(255,255,255,0.9)" />
                  </div>
                  <span style={{
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase',
                    color: 'rgba(165,180,252,0.95)',
                  }}>
                    AI-Powered Clinical Intelligence
                  </span>
                </div>
                <h1 style={{
                  fontSize: '1.85rem',
                  fontWeight: 900,
                  color: '#ffffff',
                  letterSpacing: '-0.03em',
                  margin: '0 0 0.5rem',
                  lineHeight: 1.15,
                }}>
                  Explainable CKD Progression<br />
                  <span style={{ color: 'rgba(147,197,253,0.95)' }}>Forecasting Platform</span>
                </h1>
                <p style={{
                  fontSize: '0.88rem',
                  color: 'rgba(186,230,253,0.85)',
                  margin: 0,
                  fontWeight: 400,
                  maxWidth: 520,
                }}>
                  Temporal Fusion Transformer model trained on 50,000 longitudinal patient records.
                  Multi-horizon predictions at 3-, 6-, and 12-month windows with full clinical explainability.
                </p>
              </div>

              {/* Right: Key Stats */}
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                {[
                  { label: 'Patients', value: '50K', sub: 'Training Set' },
                  { label: 'Horizons', value: '3', sub: '3m · 6m · 12m' },
                  { label: 'Test Cases', value: '127', sub: 'Validated' },
                ].map(s => (
                  <div key={s.label} style={{
                    background: 'rgba(255,255,255,0.12)',
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: 14,
                    padding: '1rem 1.25rem',
                    backdropFilter: 'blur(8px)',
                    textAlign: 'center',
                    minWidth: 90,
                  }}>
                    <div style={{ fontSize: '1.85rem', fontWeight: 900, color: '#ffffff', letterSpacing: '-0.03em', lineHeight: 1 }}>{s.value}</div>
                    <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'rgba(186,230,253,0.8)', textTransform: 'uppercase', letterSpacing: '0.07em', marginTop: '0.25rem' }}>{s.label}</div>
                    <div style={{ fontSize: '0.65rem', color: 'rgba(186,230,253,0.6)', marginTop: '0.1rem' }}>{s.sub}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Pills Row */}
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
              {['eGFR Trajectory', 'Stage Classification', 'Progression Risk', 'Biomarker Importance', 'Attention Analysis'].map(tag => (
                <span key={tag} className="metric-pill">{tag}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Content ──────────────────────────────────── */}
      <main style={{ flex: 1, padding: '2rem 2rem 3.5rem' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <Outlet />
        </div>
      </main>

      {/* ── Footer ───────────────────────────────────────── */}
      <footer style={{
        background: 'rgba(255,255,255,0.6)',
        backdropFilter: 'blur(12px)',
        borderTop: '1px solid rgba(226,232,240,0.7)',
        padding: '0.9rem 2rem',
        textAlign: 'center',
      }}>
        <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 500 }}>
          ⚕️ <strong style={{ color: '#64748b' }}>Research Prototype Only</strong> — Not for clinical diagnosis or treatment decisions. Always consult qualified medical professionals.
          &nbsp;·&nbsp; CKD-TFT Platform &nbsp;·&nbsp; Longitudinal Renal AI &nbsp;·&nbsp; Built with React + TFT
        </div>
      </footer>
    </div>
  );
}
