import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.tsx';

async function unregisterStaleMsw(): Promise<void> {
  if (!('serviceWorker' in navigator)) return;
  const registrations = await navigator.serviceWorker.getRegistrations();
  const staleRegistrations = registrations.filter((registration) =>
    registration.active?.scriptURL.includes('mockServiceWorker.js'),
  );
  if (staleRegistrations.length === 0) {
    sessionStorage.removeItem('twin-msw-unregistered');
    return;
  }
  await Promise.all(
    staleRegistrations.map((registration) => registration.unregister()),
  );
  if (
    navigator.serviceWorker.controller &&
    sessionStorage.getItem('twin-msw-unregistered') !== '1'
  ) {
    sessionStorage.setItem('twin-msw-unregistered', '1');
    window.location.reload();
  }
}

async function bootstrap(): Promise<void> {
  // MSW activation policy:
  //   - DEV         : on by default (set VITE_USE_MSW=false to disable)
  //   - PROD demo   : opt-in via VITE_FORCE_MSW=true (static demo deploy
  //                   has no FastAPI backend; MSW serves all fixtures
  //                   client-side via public/mockServiceWorker.js)
  //   - PROD real   : off (default), the app expects a real backend
  const dev = import.meta.env.DEV;
  const forced = import.meta.env.VITE_FORCE_MSW === 'true';
  const disabled = import.meta.env.VITE_USE_MSW === 'false';
  if ((dev && !disabled) || forced) {
    const { startMsw } = await import('./mocks/browser');
    await startMsw();
  } else {
    await unregisterStaleMsw();
  }
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void bootstrap();
