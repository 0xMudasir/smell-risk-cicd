// Read CSS custom properties so ECharts matches the design tokens in both themes.
export function tok(name: string): string {
  if (typeof document === 'undefined') return '';
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function palette() {
  return {
    ink: tok('--ink') || '#15203A',
    muted: tok('--ink-muted') || '#5A6680',
    rule: tok('--rule') || '#E2E6F0',
    accent: tok('--accent') || '#4338CA',
    ok: tok('--ok') || '#0E9F6E',
    warn: tok('--warn') || '#D97706',
    crit: tok('--crit') || '#E11D48',
    surface: tok('--surface') || '#FFFFFF',
  };
}

// vibrant categorical palette for multi-series / per-category charts
export function categorical(): string[] {
  const fallback = ['#4338CA', '#0EA5E9', '#0D9488', '#D97706', '#DB2777', '#16A34A', '#7C3AED', '#E11D48'];
  const got = [1, 2, 3, 4, 5, 6, 7, 8].map(i => tok(`--c${i}`)).filter(Boolean);
  return got.length === 8 ? got : fallback;
}

// per-bar color list cycling the categorical palette
export function barColors(n: number): string[] {
  const c = categorical();
  return Array.from({ length: n }, (_, i) => c[i % c.length]);
}

// Common axis/grid styling for a clean, instrument-like chart.
export function axisBase() {
  const p = palette();
  return {
    axisLine: { lineStyle: { color: p.rule } },
    axisTick: { lineStyle: { color: p.rule } },
    axisLabel: { color: p.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: p.rule, type: 'dashed' as const } },
    nameTextStyle: { color: p.muted },
  };
}

export function baseGrid() {
  return { left: 56, right: 20, top: 28, bottom: 48, containLabel: true };
}
