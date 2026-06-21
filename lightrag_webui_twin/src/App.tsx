import { QueryClientProvider } from '@tanstack/react-query';
import { AppShell } from './app/AppShell';
import { queryClient } from './app/queryClient';

// eslint-disable-next-line react-refresh/only-export-components -- compatibility export used by App.test.
export { shouldUseFixtureFallback } from './app/fixtureFallback';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  );
}

export default App;
