export function shouldUseFixtureFallback(env: {
  dev: boolean;
  forceMsw?: string;
  useMsw?: string;
}): boolean {
  if (env.forceMsw === 'true') return true;
  return env.dev && env.useMsw !== 'false';
}
