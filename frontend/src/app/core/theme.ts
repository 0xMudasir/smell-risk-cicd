// Read CSS custom properties so ECharts matches the design tokens in both themes.
export function tok(name: string): string {
  if (typeof document === 'undefined') return '';
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function palette() {
  return {
    ink: tok('--ink') || '#18202A',
    muted: tok('--ink-muted') || '#5B6672',
    rule: tok('--rule') || '#D8DEDB',
    accent: tok('--accent') || '#1D5C7A',
    ok: tok('--ok') || '#2F6B4F',
    warn: tok('--warn') || '#B7791F',
    crit: tok('--crit') || '#A8322D',
    surface: tok('--surface') || '#FFFFFF',
  };
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
