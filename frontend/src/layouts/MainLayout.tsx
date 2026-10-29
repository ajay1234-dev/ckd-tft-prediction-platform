import { useEffect, useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { Activity, BrainCircuit, Microscope, CheckCircle2, XCircle, Loader } from 'lucide-react';
import api from '../api/client';

export function MainLayout() {
  const [health, setHealth] = useState<'loading' | 'ok' | 'error'>('loading');

  useEffect(() => {
    api.checkHealth()
      .then(r => setHealth(r.status === 'ok' && r.model_loaded ? 'ok' : 'error'))
      .catch(() => setHealth('error'));
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#f8fafc' }}>
      {/* Top Navbar Header */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: 'rgba(255, 255, 255, 0.92)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid #e2e8f0',
        padding: '0.75rem 2rem',
      }}>
        <div style={{
          maxWidth: '1160px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          {/* Logo Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: 38,
              height: 38,
              background: 'linear-gradient(135deg, #4f46e5 0%, #0284c7 100%)',
              borderRadius: 12,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 2px 6px rgba(79, 70, 229, 0.25)',
            }}>
              <Activity size={20} color="#ffffff" strokeWidth={2.5} />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: '1.05rem', color: '#0f172a', letterSpacing: '-0.01em', lineHeight: 1.2 }}>
                CKD Predictor
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 500, letterSpacing: '0.02em' }}>
                Longitudinal Renal Intelligence
              </div>
            </div>
          </div>

          {/* Unique Floating Segmented Pill Switcher */}
          <nav className="nav-pill-container" style={{
            background: '#f1f5f9',
            padding: '4px',
            borderRadius: '9999px',
            border: '1px solid #e2e8f0',
            boxShadow: 'inset 0 1px 2px rgba(0, 0, 0, 0.04)',
          }}>
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

          {/* Live Status Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 0.85rem',
            borderRadius: '9999px',
            background: health === 'ok' ? '#ecfdf5' : health === 'error' ? '#fef2f2' : '#f1f5f9',
            border: `1px solid ${health === 'ok' ? '#a7f3d0' : health === 'error' ? '#fecaca' : '#e2e8f0'}`,
          }}>
            {health === 'loading' && <Loader size={12} style={{ animation: 'spin 1.5s linear infinite', color: '#64748b' }} />}
            {health === 'ok' && <CheckCircle2 size={13} color="#059669" />}
            {health === 'error' && <XCircle size={13} color="#dc2626" />}
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              color: health === 'ok' ? '#065f46' : health === 'error' ? '#991b1b' : '#64748b',
              letterSpacing: '0.01em',
            }}>
              {health === 'loading' ? 'Connecting...' : health === 'ok' ? 'Model Ready' : 'Model Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '2rem 1.5rem 3rem' }}>
        <div style={{ maxWidth: '1160px', margin: '0 auto' }}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}
