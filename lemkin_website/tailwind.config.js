/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        'serif': ['Source Serif Pro', 'Georgia', 'serif'],
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'display': ['"SF Pro Display"', 'Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Achromatic institutional design system
        // Using CSS variables for complete theme control
        'bg-default': 'var(--color-bg-default)',
        'bg-surface': 'var(--color-bg-surface)',
        'bg-elevated': 'var(--color-bg-elevated)',

        'fg-primary': 'var(--color-fg-primary)',
        'fg-muted': 'var(--color-fg-muted)',
        'fg-subtle': 'var(--color-fg-subtle)',
        'fg-inverse': 'var(--color-fg-inverse)',

        'accent-brand': 'var(--color-accent-brand)',
        'accent-cta': 'var(--color-accent-cta)',

        'border-default': 'var(--color-border-default)',
        'border-focus': 'var(--color-border-focus)',

        // Status colors (semantic only, always with icons)
        'status-success': 'var(--color-status-success)',
        'status-warning': 'var(--color-status-warning)',
        'status-danger': 'var(--color-status-danger)',
        'status-info': 'var(--color-status-info)',
      },
      backgroundImage: {
        // Minimal dot grid pattern only
        'dot-grid': 'radial-gradient(circle at 1px 1px, rgb(148 163 184 / 0.03) 1px, transparent 0)',
      },
      fontSize: {
        // Restrained typography scale - no extreme sizes
        'heading-xl': ['2rem', { lineHeight: '1.25', letterSpacing: '-0.01em', fontWeight: '600' }],
        'heading-lg': ['1.5rem', { lineHeight: '1.3', letterSpacing: '-0.01em', fontWeight: '600' }],
        'heading-md': ['1.25rem', { lineHeight: '1.4', fontWeight: '600' }],
        'heading-sm': ['1.125rem', { lineHeight: '1.4', fontWeight: '600' }],
        'body-lg': ['1.125rem', { lineHeight: '1.6' }],
        'body-md': ['1rem', { lineHeight: '1.6' }],
        'body-sm': ['0.875rem', { lineHeight: '1.5' }],
        'caption': ['0.75rem', { lineHeight: '1.4' }],
      },
      boxShadow: {
        // No shadows - borders only for institutional restraint
        'none': 'none',
      },
      backdropBlur: {
        'xs': '2px',
      },
      animation: {
        // Minimal functional animations only
        'spin': 'spin 1s linear infinite',
      },
      keyframes: {
        spin: {
          from: { transform: 'rotate(0deg)' },
          to: { transform: 'rotate(360deg)' },
        },
      },
      borderRadius: {
        // Restrained corner radius - max 6px
        'none': '0',
        'sm': '2px',
        'md': '4px',
        'lg': '6px',
        'full': '9999px',
      },
    },
  },
  plugins: [],
}