import { Component, Input } from '@angular/core';

// Metric identifiers are code identifiers -> rendered in mono (CLAUDE.md 7.3).
@Component({
  selector: 'app-metric-name',
  standalone: true,
  template: `<code class="mn">{{ name }}</code>`,
  styles: [`.mn { font-size: 13px; white-space: nowrap; }`],
})
export class MetricNameComponent {
  @Input({ required: true }) name!: string;
}
