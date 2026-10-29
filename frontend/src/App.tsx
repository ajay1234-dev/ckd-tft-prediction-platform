import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import PatientPrediction from './pages/PatientPrediction';
import Explainability from './pages/Explainability';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/" element={<Navigate to="/predict" replace />} />
          <Route path="/predict" element={<PatientPrediction />} />
          <Route path="/explain" element={<Explainability />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
