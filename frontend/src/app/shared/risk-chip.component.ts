import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-risk-chip',
  standalone: true,
  template: `<span class="chip" [class]="tier" [attr.aria-label]="'Risk: ' + label">{{ label }}</span>`,
  styles: [`
    .chip { font-weight: 600; text-transform: capitalize; }
    .low { background: var(--ok-tint); color: var(--ok); border-color: var(--ok); }
    .elevated { background: var(--warn-tint); color: var(--warn); border-color: var(--warn); }
    .critical { background: var(--crit-tint); color: var(--crit); border-color: var(--crit); }
  `],
})
export class RiskChipComponent {
  @Input({ required: true }) tier!: 'low' | 'elevated' | 'critical';
  get label(): string { return this.tier; }
}
