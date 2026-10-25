import { Component, OnInit, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import type { EChartsOption } from 'echarts';
import { ResultsService } from '../../core/results.service';
import { ChartPanelComponent } from '../../shared/chart-panel.component';
import { Rq1, Rq1Corr } from '../../core/models';
import { palette, axisBase, barColors } from '../../core/theme';

const METHODS = [
  ['linear_regression_coef', 'LinReg |coef|'],
  ['logistic_regression_coef', 'LogReg |coef|'],
  ['xgboost_gain', 'XGB gain'],
  ['anova_f', 'ANOVA F'],
  ['boruta', 'Boruta'],
] as const;

@Component({
  selector: 'app-features',
  standalone: true,
  imports: [DecimalPipe, ChartPanelComponent],
  template: `
    <h1>Features (RQ1)</h1>
    @if (rq1()) {
      <p class="t-muted">Five importance methods over {{ rq1()!.features.length }} metrics. Metric names cannot
        overlap the paper's Understand metrics, so we compare families: size + cyclomatic dominate, as in the paper.</p>

      <div class="panel consensus">
        <div><strong>Strict consensus (5/5):</strong>
          @for (m of rq1()!.consensus.strict_intersection_5of5; track m) { <span class="chip ok">{{ m }}</span> }</div>
        <div style="margin-top:8px"><strong>At least 4 of 5:</strong>
          @for (m of rq1()!.consensus.at_least_4of5; track m) { <span class="chip">{{ m }}</span> }</div>
      </div>

      <app-chart-panel title="Rank matrix (top-10 per method; 1 = most important)" [options]="rankMatrix()" [height]="420">
        <table table>
          <thead><tr><th>Metric</th>@for (m of methodLabels; track m){<th class="num">{{ m }}</th>}</tr></thead>
          <tbody>
            @for (f of rq1()!.features; track f) {
              <tr><td class="mono">{{ f }}</td>
                @for (mk of methodKeys; track mk) { <td class="num">{{ rankOf(mk, f) || '' }}</td> }</tr>
            }
          </tbody>
        </table>
      </app-chart-panel>

      <div class="grid two">
        <app-chart-panel title="Permutation importance (grouped-CV test folds)" [options]="permBar()" [height]="360">
          <table table>
            <thead><tr><th>Metric</th><th class="num">mean ROC-AUC drop</th></tr></thead>
            <tbody>@for (r of rq1()!.fixes.permutation_importance.ranked; track r.feature) {
              <tr><td class="mono">{{ r.feature }}</td><td class="num">{{ r.mean | number:'1.4-4' }}</td></tr> }</tbody>
          </table>
        </app-chart-panel>

        @if (corr()) {
          <app-chart-panel title="Metric–metric correlation (|r|)" [options]="corrHeat()" [height]="360">
            <table table><tbody><tr><td class="t-muted">{{ corr()!.features.length }}×{{ corr()!.features.length }} Pearson matrix; see chart.</td></tr></tbody></table>
          </app-chart-panel>
        }
      </div>
      <p class="t-muted t-12">Ours, extra: permutation + SHAP. Coefficient ranks are unstable
        ({{ rq1()!.fixes.correlated_pairs_gt_0_9.length }} pairs with |r| &gt; 0.9) — prefer tree/permutation/SHAP.</p>
    } @else {
      <div class="panel"><div class="skeleton" style="height:300px"></div></div>
    }
  `,
  styles: [`
    .consensus { margin: 12px 0; }
    .consensus .chip { margin: 2px 4px 2px 0; font-family: var(--font-mono); font-size: 11px; }
    .consensus .chip.ok { background: var(--ok-tint); color: var(--ok); border-color: var(--ok); }
    .two { grid-template-columns: 1fr 1fr; }
    @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
    .mono { font-family: var(--font-mono); font-size: 12px; }
  `],
})
export class FeaturesComponent implements OnInit {
  private results = inject(ResultsService);
  rq1 = signal<Rq1 | null>(null);
  corr = signal<Rq1Corr | null>(null);
  methodKeys = METHODS.map(m => m[0]);
  methodLabels = METHODS.map(m => m[1]);
  private rankMaps: Record<string, Record<string, number>> = {};

  ngOnInit(): void {
    this.results.get<Rq1>('rq1_importance.json').subscribe(r => {
      this.rq1.set(r);
      for (const [key] of METHODS) {
        this.rankMaps[key] = {};
        r.methods[key].ranked.forEach(x => (this.rankMaps[key][x.feature] = x.rank));
      }
    });
    this.results.get<Rq1Corr>('rq1_correlation.json').subscribe(c => this.corr.set(c));
  }

  rankOf(method: string, feature: string): number | null {
    const r = this.rankMaps[method]?.[feature];
    return r && r <= 10 ? r : null;
  }

  rankMatrix(): EChartsOption {
    const p = palette();
    const feats = this.rq1()!.features;
    const data: [number, number, number | null][] = [];
    METHODS.forEach(([key], j) => feats.forEach((f, i) => {
      const r = this.rankOf(key, f);
      data.push([j, i, r]);
    }));
    return {
      grid: { left: 150, right: 20, top: 10, bottom: 70 },
      tooltip: {},
      xAxis: { type: 'category', data: this.methodLabels, ...axisBase(), axisLabel: { color: p.muted, rotate: 20, fontSize: 10 } },
      yAxis: { type: 'category', data: feats, ...axisBase(), axisLabel: { color: p.muted, fontSize: 10 } },
      visualMap: { min: 1, max: 10, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
        inRange: { color: ['#4338CA', '#7C3AED', '#DB2777', '#F9A8D4'] }, textStyle: { color: p.muted } },
      series: [{ type: 'heatmap', data: data.filter(d => d[2] !== null) as any,
        itemStyle: { borderColor: p.surface, borderWidth: 2 },
        label: { show: true, color: '#fff', fontSize: 10, fontWeight: 'bold' } }],
    };
  }

  permBar(): EChartsOption {
    const p = palette();
    const r = [...this.rq1()!.fixes.permutation_importance.ranked].reverse();
    return {
      grid: { left: 150, right: 20, top: 10, bottom: 30, containLabel: true },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', ...axisBase() },
      yAxis: { type: 'category', data: r.map(x => x.feature), ...axisBase(), axisLabel: { color: p.muted, fontSize: 10 } },
      series: [{ type: 'bar',
        data: r.map((x, i) => ({ value: x.mean, itemStyle: { color: barColors(r.length)[i], borderRadius: [0, 4, 4, 0] } })) }],
    };
  }

  corrHeat(): EChartsOption {
    const p = palette();
    const c = this.corr()!;
    const data: [number, number, number][] = [];
    c.matrix.forEach((row, i) => row.forEach((v, j) => data.push([j, i, Math.abs(v)])));
    return {
      grid: { left: 150, right: 20, top: 10, bottom: 90 },
      tooltip: {},
      xAxis: { type: 'category', data: c.features, ...axisBase(), axisLabel: { color: p.muted, rotate: 70, fontSize: 9 } },
      yAxis: { type: 'category', data: c.features, ...axisBase(), axisLabel: { color: p.muted, fontSize: 9 } },
      visualMap: { min: 0, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
        inRange: { color: ['#EEF2FF', '#38BDF8', '#6D28D9', '#DB2777'] }, textStyle: { color: p.muted } },
      series: [{ type: 'heatmap', data }],
    };
  }
}
