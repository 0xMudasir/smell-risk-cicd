import { Component, computed, input, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';

// Signature component (CLAUDE.md 7.4): a horizontal scale per metric showing
// where a value sits against the clean median, warning and critical thresholds.
@Component({
  selector: 'app-threshold-ruler',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="ruler">
      <div class="head">
        <code class="metric">{{ metric() }}</code>
        <span class="spacer"></span>
        <span class="val" [class]="level()">{{ markerLabel() }} {{ value() | number:'1.0-2' }}
          <span class="dot" [class]="level()"></span>{{ level() }}</span>
        <button class="log" (click)="log.set(!log())" [attr.aria-pressed]="log()"
                title="Toggle log scale">{{ log() ? 'log' : 'linear' }}</button>
      </div>

      <div class="track" role="img" [attr.aria-label]="ariaLabel()">
        <div class="seg neutral" [style.left.%]="0" [style.width.%]="posWarn()"></div>
        <div class="seg warn" [style.left.%]="posWarn()" [style.width.%]="posCrit() - posWarn()"></div>
        <div class="seg crit" [style.left.%]="posCrit()" [style.width.%]="100 - posCrit()"></div>

        <div class="tick clean" [style.left.%]="posClean()"></div>
        <div class="tick" [style.left.%]="posWarn()"></div>
        <div class="tick" [style.left.%]="posCrit()"></div>

        <div class="marker" [style.left.%]="posValue()" [class.clamped]="clamped()">
          <span class="arrow">{{ clamped() ? '▸' : '▼' }}</span>
        </div>
      </div>

      <div class="labels">
        <span class="lbl" [style.left.%]="posClean()">clean {{ cleanMedian() | number:'1.0-1' }}</span>
        <span class="lbl" [style.left.%]="posWarn()">warn {{ warning() | number:'1.0-1' }}</span>
        <span class="lbl" [style.left.%]="posCrit()">crit {{ critical() | number:'1.0-1' }}</span>
      </div>
    </div>
  `,
  styles: [`
    .ruler { padding: 6px 0 22px; }
    .head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
    .metric { font-size: 13px; }
    .spacer { flex: 1; }
    .val { font-variant-numeric: tabular-nums; text-transform: capitalize; }
    .val.warning { color: var(--warn); } .val.critical { color: var(--crit); } .val.clean { color: var(--ok); }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 4px 0 8px; }
    .dot.clean { background: var(--ok); } .dot.warning { background: var(--warn); } .dot.critical { background: var(--crit); }
    .log { font: inherit; font-size: 11px; padding: 1px 6px; border: 1px solid var(--rule);
           border-radius: var(--radius-sm); background: var(--surface); color: var(--ink-muted); cursor: pointer; }
    .track { position: relative; height: 12px; border-radius: 6px; overflow: visible; background: var(--rule); }
    .seg { position: absolute; top: 0; height: 12px; }
    .seg.neutral { background: var(--rule); border-radius: 6px 0 0 6px; }
    .seg.warn { background: var(--warn-tint); }
    .seg.crit { background: var(--crit-tint); border-radius: 0 6px 6px 0; }
    .tick { position: absolute; top: -3px; width: 2px; height: 18px; background: var(--ink-muted); transform: translateX(-1px); }
    .tick.clean { background: transparent; border-left: 2px dashed var(--ok); }
    .marker { position: absolute; top: -12px; transform: translateX(-50%); color: var(--ink); }
    .marker .arrow { font-size: 12px; }
    .marker.clamped { color: var(--crit); }
    .labels { position: relative; height: 0; }
    .lbl { position: absolute; top: 2px; transform: translateX(-50%); font-size: 11px; color: var(--ink-muted); white-space: nowrap; }
  `],
})
export class ThresholdRulerComponent {
  metric = input.required<string>();
  value = input.required<number>();
  min = input<number>(0);
  cleanMedian = input<number>(0);
  warning = input.required<number>();
  critical = input.required<number>();
  max = input.required<number>();
  markerLabel = input<string>('value');
  log = signal(false);

  private scale(v: number): number {
    const lo = this.min(), hi = Math.max(this.max(), this.critical() * 1.1, 1);
    let x: number;
    if (this.log()) {
      const a = Math.log1p(Math.max(lo, 0)), b = Math.log1p(hi);
      x = (Math.log1p(Math.max(v, 0)) - a) / (b - a || 1);
    } else {
      x = (v - lo) / (hi - lo || 1);
    }
    return Math.max(0, Math.min(1, x)) * 100;
  }

  posClean = computed(() => this.scale(this.cleanMedian()));
  posWarn = computed(() => this.scale(this.warning()));
  posCrit = computed(() => this.scale(this.critical()));
  posValue = computed(() => this.scale(this.value()));
  clamped = computed(() => this.value() > Math.max(this.max(), this.critical() * 1.1, 1));

  level = computed<'clean' | 'warning' | 'critical'>(() => {
    const v = this.value();
    if (v > this.critical()) return 'critical';
    if (v > this.warning()) return 'warning';
    return 'clean';
  });

  ariaLabel = computed(() =>
    `${this.metric()} is ${this.value()}, ${this.level()} ` +
    `(warning threshold ${this.warning()}, critical threshold ${this.critical()}).`);
}
