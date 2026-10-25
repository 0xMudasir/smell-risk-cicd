import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { ResultsService } from '../../core/results.service';
import { RiskChipComponent } from '../../shared/risk-chip.component';
import { ThresholdRulerComponent } from '../../shared/threshold-ruler.component';
import { MetricNameComponent } from '../../shared/metric-name.component';
import { environment } from '../../../environments/environment';
import { FeatureSchema, PredictResponse, Rq2Thresholds, DatasetAudit } from '../../core/models';

@Component({
  selector: 'app-predict',
  standalone: true,
  imports: [DecimalPipe, PercentPipe, FormsModule, RiskChipComponent, ThresholdRulerComponent, MetricNameComponent],
  template: `
    <h1>Predict</h1>

    <div class="tabs">
      <button class="btn" [class.btn-primary]="mode() === 'form'" (click)="mode.set('form')">Enter metrics</button>
      <button class="btn" [class.btn-primary]="mode() === 'csv'" (click)="mode.set('csv')">Upload CSV</button>
    </div>

    @if (offline()) {
      <div class="panel offline">
        <p>The prediction service isn't reachable at <code>{{ apiBase }}</code>.
          Start it with <code>python backend/app.py</code>, then select Retry.</p>
        <button class="btn" (click)="init()">Retry</button>
      </div>
    }

    @if (schema() && mode() === 'form') {
      <div class="panel">
        <div class="actions">
          <button class="btn" (click)="loadExample('clean')">Load typical clean example</button>
          <button class="btn" (click)="loadExample('smelly')">Load typical smelly example</button>
          <span class="spacer"></span>
          <button class="btn btn-primary" (click)="run()" [disabled]="offline() || busy()">Run prediction</button>
        </div>
        <div class="form-grid">
          @for (f of schema()!.features; track f.name) {
            <label class="field">
              <app-metric-name [name]="f.name"/>
              <input type="number" min="0" step="any" [(ngModel)]="values[f.name]"
                     [placeholder]="f.median" />
              <span class="help t-12 t-muted">{{ f.description }}</span>
            </label>
          }
        </div>
      </div>

      @if (result()) {
        <div class="panel result" aria-live="polite">
          <div class="rhead">
            <div><div class="t-muted t-12">Smell probability</div>
              <div class="prob">{{ result()!.probability | percent:'1.0-0' }}</div></div>
            <app-risk-chip [tier]="result()!.risk_tier" />
          </div>

          <h3>Why</h3>
          <div class="contribs">
            @for (c of result()!.top_contributors; track c.metric) {
              <div class="contrib">
                <app-metric-name [name]="c.metric"/>
                <span class="cv">{{ c.value | number:'1.0-2' }}</span>
                <div class="bar"><div class="fill" [class.neg]="c.impact < 0"
                  [style.width.%]="barWidth(c.impact)"></div></div>
                <span class="imp num">{{ c.impact | number:'1.2-2' }}</span>
              </div>
            }
          </div>

          @if (thr()) {
            <h3>Thresholds for this entity</h3>
            @for (m of consensus(); track m) {
              <app-threshold-ruler [metric]="m" [value]="num(values[m])"
                [min]="thr()!.thresholds[m].ruler_quantiles.min"
                [cleanMedian]="thr()!.thresholds[m].clean_median"
                [warning]="thr()!.thresholds[m].warning"
                [critical]="thr()!.thresholds[m].critical"
                [max]="thr()!.thresholds[m].ruler_quantiles.p95_smelly" />
            }
          }
        </div>
      }
      @if (error()) { <div class="panel err">{{ error() }}</div> }
    }

    @if (mode() === 'csv') {
      <div class="panel">
        <p class="t-muted">Upload a CSV with one column per feature (max 5 MB, 5000 rows).</p>
        <input type="file" accept=".csv" (change)="onFile($event)" [disabled]="offline()" />
        @if (batch().length) {
          <button class="btn" (click)="downloadCsv()" style="margin-left:12px">Download results CSV</button>
          <div class="scroll" style="margin-top:12px">
            <table>
              <thead><tr><th class="num">row</th><th class="num">probability</th><th>risk</th></tr></thead>
              <tbody>@for (r of batch(); track r.row) {
                <tr><td class="num">{{ r.row }}</td><td class="num">{{ r.probability | percent:'1.0-0' }}</td>
                  <td><app-risk-chip [tier]="r.risk_tier"/></td></tr> }</tbody>
            </table>
          </div>
        }
        @if (error()) { <div class="err" style="margin-top:12px">{{ error() }}</div> }
      </div>
    }
  `,
  styles: [`
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .actions { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
    .actions .spacer { flex: 1; }
    .form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    .help { min-height: 1.5em; }
    .offline { border-color: var(--warn); }
    .err { color: var(--crit); }
    .rhead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .prob { font-size: 32px; font-weight: 600; }
    .contrib { display: grid; grid-template-columns: 160px 70px 1fr 56px; align-items: center; gap: 8px; margin: 4px 0; }
    .contrib .cv { font-variant-numeric: tabular-nums; color: var(--ink-muted); font-size: 12px; }
    .bar { background: var(--paper); border-radius: 3px; height: 10px; }
    .fill { background: var(--crit); height: 10px; border-radius: 3px; }
    .fill.neg { background: var(--ok); }
    .scroll { max-height: 420px; overflow: auto; }
  `],
})
export class PredictComponent implements OnInit {
  private api = inject(ApiService);
  private results = inject(ResultsService);
  apiBase = environment.apiBaseUrl;

  mode = signal<'form' | 'csv'>('form');
  schema = signal<FeatureSchema | null>(null);
  thr = signal<Rq2Thresholds | null>(null);
  audit = signal<DatasetAudit | null>(null);
  offline = signal(false);
  busy = signal(false);
  result = signal<PredictResponse | null>(null);
  error = signal('');
  batch = signal<PredictResponse[]>([]);
  values: Record<string, number | null> = {};
  consensus = computed(() => this.thr()?.consensus_metrics ?? []);

  ngOnInit(): void { this.init(); }

  init(): void {
    this.offline.set(false); this.error.set('');
    // schema from API when reachable; otherwise fall back to the static schema
    this.api.schema().subscribe({
      next: s => this.setSchema(s),
      error: () => {
        this.offline.set(true);
        this.results.get<FeatureSchema>('feature_schema.json').subscribe(s => this.setSchema(s));
      },
    });
    this.results.get<Rq2Thresholds>('rq2_thresholds.json').subscribe(t => this.thr.set(t));
    this.results.get<DatasetAudit>('dataset_audit.json').subscribe(a => this.audit.set(a));
  }

  private setSchema(s: FeatureSchema): void {
    this.schema.set(s);
    for (const f of s.features) if (!(f.name in this.values)) this.values[f.name] = null;
  }

  loadExample(kind: 'clean' | 'smelly'): void {
    const sep = this.audit()?.separability;
    for (const f of this.schema()!.features) {
      const s = sep?.[f.name];
      this.values[f.name] = s ? (kind === 'clean' ? s.clean_median : s.smelly_median) : f.median;
    }
  }

  num(v: number | null | undefined): number { return typeof v === 'number' ? v : 0; }
  barWidth(impact: number): number { return Math.min(100, Math.abs(impact) * 20); }

  run(): void {
    this.error.set(''); this.busy.set(true);
    const metrics: Record<string, number> = {};
    for (const f of this.schema()!.features) metrics[f.name] = this.num(this.values[f.name]);
    this.api.predict(metrics).subscribe({
      next: r => { this.result.set(r); this.busy.set(false); },
      error: e => {
        this.busy.set(false);
        this.error.set(e?.error?.error === 'validation_error'
          ? 'Enter a number of 0 or more for every field.'
          : 'Prediction failed. Is the API running at ' + this.apiBase + '?');
        if (e.status === 0) this.offline.set(true);
      },
    });
  }

  onFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.error.set('');
    this.api.predictBatch(file).subscribe({
      next: r => this.batch.set(r.results),
      error: e => { this.error.set(e?.error?.error ? JSON.stringify(e.error) : 'Batch failed.'); if (e.status === 0) this.offline.set(true); },
    });
  }

  downloadCsv(): void {
    const rows = [['row', 'probability', 'risk_tier'],
      ...this.batch().map(r => [r.row, r.probability, r.risk_tier])];
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'predictions.csv'; a.click();
    URL.revokeObjectURL(a.href);
  }
}
