import { Component, OnInit, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { forkJoin } from 'rxjs';
import { ResultsService } from '../../core/results.service';
import { ThresholdRulerComponent } from '../../shared/threshold-ruler.component';
import { MetricNameComponent } from '../../shared/metric-name.component';
import { Rq2Stats, Rq2Tests, Rq2Thresholds, ThresholdEntry } from '../../core/models';

@Component({
  selector: 'app-thresholds',
  standalone: true,
  imports: [DecimalPipe, ThresholdRulerComponent, MetricNameComponent],
  template: `
    <h1>Thresholds (RQ2)</h1>
    @if (ready()) {
      <p class="t-muted">Per-metric separation and the paper's threshold rule (warning = smelly median,
        critical = smelly Q3). We also validate each threshold as a classifier — which the paper never did.</p>

      <div class="panel">
        <h3>Class separation and effect size</h3>
        <div class="scroll-x">
        <table>
          <thead><tr><th>Metric</th><th class="num">clean median</th><th class="num">smelly median</th>
            <th class="num">ratio</th><th class="num">p (Holm)</th><th>effect</th></tr></thead>
          <tbody>
            @for (m of metrics(); track m) {
              <tr>
                <td><app-metric-name [name]="m"/></td>
                <td class="num">{{ stats()!.stats[m].clean.median | number:'1.0-2' }}</td>
                <td class="num">{{ stats()!.stats[m].smelly.median | number:'1.0-2' }}</td>
                <td class="num">{{ ratio(m) }}</td>
                <td class="num">{{ pval(m) }}</td>
                <td><span class="chip" [class]="band(m)">{{ band(m) }} ({{ tests()!.tests[m].full.rank_biserial | number:'1.2-2' }})</span></td>
              </tr>
            }
          </tbody>
        </table>
        </div>
      </div>

      <div class="panel">
        <h3>Threshold rulers (marker = smelly median)</h3>
        @for (m of metrics(); track m) {
          <app-threshold-ruler
            [metric]="m" [value]="thr()!.thresholds[m].warning"
            [min]="thr()!.thresholds[m].ruler_quantiles.min"
            [cleanMedian]="thr()!.thresholds[m].clean_median"
            [warning]="thr()!.thresholds[m].warning"
            [critical]="thr()!.thresholds[m].critical"
            [max]="thr()!.thresholds[m].ruler_quantiles.p95_smelly"
            markerLabel="smelly median" />
        }
      </div>

      <div class="panel">
        <h3>Threshold as a rule (grouped-CV test folds)</h3>
        <div class="scroll-x">
        <table>
          <thead><tr><th>Metric</th><th>type</th><th class="num">threshold</th>
            <th class="num">precision</th><th class="num">recall</th><th class="num">F1</th></tr></thead>
          <tbody>
            @for (m of metrics(); track m) {
              @for (k of kinds; track k) {
                <tr><td>@if (k === 'warning') { <app-metric-name [name]="m"/> }</td>
                  <td>{{ k }}</td>
                  <td class="num">{{ cv(m, k).threshold_mean | number:'1.0-2' }}</td>
                  <td class="num">{{ cv(m, k).precision_mean | number:'1.3-3' }}</td>
                  <td class="num">{{ cv(m, k).recall_mean | number:'1.3-3' }}</td>
                  <td class="num">{{ cv(m, k).f1_mean | number:'1.3-3' }}</td></tr>
              }
            }
          </tbody>
        </table>
        </div>
        <p class="t-muted t-12">Single-metric thresholds are weak detectors (best F1 ~0.37); Youden trades
          precision for recall. This is why RQ3 uses multivariate models.</p>
      </div>
    } @else {
      <div class="panel"><div class="skeleton" style="height:260px"></div></div>
    }
  `,
  styles: [`
    .chip.small { background: var(--ok-tint); color: var(--ok); }
    .chip.medium { background: var(--warn-tint); color: var(--warn); }
    .chip.large { background: var(--crit-tint); color: var(--crit); }
    .chip.negligible { color: var(--ink-muted); }
  `],
})
export class ThresholdsComponent implements OnInit {
  private results = inject(ResultsService);
  stats = signal<Rq2Stats | null>(null);
  tests = signal<Rq2Tests | null>(null);
  thr = signal<Rq2Thresholds | null>(null);
  metrics = signal<string[]>([]);
  ready = signal(false);
  kinds = ['warning', 'critical', 'youden'];

  ngOnInit(): void {
    forkJoin({
      s: this.results.get<Rq2Stats>('rq2_stats.json'),
      t: this.results.get<Rq2Tests>('rq2_tests.json'),
      h: this.results.get<Rq2Thresholds>('rq2_thresholds.json'),
    }).subscribe(({ s, t, h }) => {
      this.stats.set(s); this.tests.set(t); this.thr.set(h);
      this.metrics.set(h.consensus_metrics);
      this.ready.set(true);
    });
  }

  ratio(m: string): string {
    const r = this.stats()!.stats[m].median_ratio_smelly_over_clean;
    return typeof r === 'number' ? r.toFixed(2) + '×' : String(r);
  }
  pval(m: string): string {
    const p = this.tests()!.tests[m].full.p_holm;
    return p < 1e-4 ? '<1e-4' : p.toFixed(4);
  }
  band(m: string): string { return this.tests()!.tests[m].full.effect_magnitude; }
  cv(m: string, k: string): { threshold_mean: number; precision_mean: number; recall_mean: number; f1_mean: number } {
    return (this.thr()!.thresholds[m] as ThresholdEntry).as_classifier_cv[k];
  }
}
