import { describe, expect, it } from 'vitest';
import { shouldUseFixtureFallback } from './App';

describe('shouldUseFixtureFallback', () => {
  it('allows fixture fallbacks in dev when MSW is active', () => {
    expect(shouldUseFixtureFallback({ dev: true })).toBe(true);
  });

  it('allows fixture fallbacks for the explicit standalone MSW demo build', () => {
    expect(
      shouldUseFixtureFallback({
        dev: false,
        forceMsw: 'true',
        useMsw: 'false',
      }),
    ).toBe(true);
  });

  it('disables fixture fallbacks in real-backend mode', () => {
    expect(shouldUseFixtureFallback({ dev: false })).toBe(false);
    expect(shouldUseFixtureFallback({ dev: true, useMsw: 'false' })).toBe(false);
  });
});
