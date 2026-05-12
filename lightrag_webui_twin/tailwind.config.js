/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        // Twin RAG tokens exposed as Tailwind utility classes.
        // Reference the CSS variables so theme switching (data-theme="dark")
        // works without rebuilding the bundle.
        'twin-accent': 'var(--twin-accent)',
        'twin-accent-hover': 'var(--twin-accent-hover)',
        'twin-accent-soft-bg': 'var(--twin-accent-soft-bg)',
        'twin-accent-soft-text': 'var(--twin-accent-soft-text)',
        'twin-accent-soft-border': 'var(--twin-accent-soft-border)',

        'twin-green-500': 'var(--twin-green-500)',
        'twin-green-700': 'var(--twin-green-700)',
        'twin-green-50': 'var(--twin-green-50)',

        'twin-amber-vivid': 'var(--twin-amber-vivid)',
        'twin-amber-700': 'var(--twin-amber-700)',
        'twin-amber-50': 'var(--twin-amber-50)',

        'twin-red-vivid': 'var(--twin-red-vivid)',
        'twin-red-50': 'var(--twin-red-50)',
        'twin-red-border': 'var(--twin-red-border)',

        // Neutrals — read from CSS vars (light/dark aware)
        'text-primary': 'var(--color-text-primary)',
        'text-secondary': 'var(--color-text-secondary)',
        'text-tertiary': 'var(--color-text-tertiary)',
        'border-secondary': 'var(--color-border-secondary)',
        'border-tertiary': 'var(--color-border-tertiary)',
        'bg-primary': 'var(--color-background-primary)',
        'bg-secondary': 'var(--color-background-secondary)',
        'bg-tertiary': 'var(--color-background-tertiary)',
      },
      fontFamily: {
        mono: 'var(--font-mono)',
        sans: 'var(--font-sans)',
      },
      borderRadius: {
        sm: 'var(--border-radius-sm)',
        md: 'var(--border-radius-md)',
        lg: 'var(--border-radius-lg)',
      },
    },
  },
  plugins: [],
};
