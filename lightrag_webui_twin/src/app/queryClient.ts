import { QueryClient } from '@tanstack/react-query';

declare global {
  interface Window {
    __TWIN_E2E_QUERY_CLIENT?: QueryClient;
  }
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

if (typeof window !== 'undefined' && import.meta.env.DEV) {
  window.__TWIN_E2E_QUERY_CLIENT = queryClient;
}
