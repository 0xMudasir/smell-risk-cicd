import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import type { EChartsOption } from 'echarts';
import { ResultsService } from '../../core/results.service';
import { ChartPanelComponent } from '../../shared/chart-panel.component';
import { Rq3Results, Rq3Nemenyi, Rq3Roc, Rq3Confusion, Rq3Improvements } from '../../core/models';
import { palette, axisBase, baseGrid, categorical, barColors } from '../../core/theme';

@Component({
  selector: 'app-models',
  standalone: true,
  imports: [DecimalPipe, FormsModule, ChartPanelComponent],
  template: `
    <h1>Models (RQ3)</h1>
    @if (ready()) {
      <p class="t-muted">11 classifiers, grouped CV, mean ± std over 5 folds. Paper AUC (security bugs) shown
        for context. XGBoost wins (Nemenyi rank 1), tree ensembles dominate — the paper's pattern.</p>

      <div class="panel">
        <h3>Leaderboard</h3>
        <div class="scroll">
          <table>
            <thead><tr><th>Model</th><th class="num">AUC</th><th class="num muted">paper AUC</th>
              <th class="num">PR-AUC</th><th class="num">F1</th><th class="num">MCC</th>
              <th class="num">recall</th><th class="num">rank</th></tr></thead>
            <tbody>
              @for (m of ranked(); track m) {
                <tr [class.best]="m === best()">
                  <td>{{ m }}</td>
                  <td class="num">{{ res()!.models[m].ours_mean.roc_auc | number:'1.3-3' }}
                    <span class="std">±{{ res()!.models[m].ours_std.roc_auc | number:'1.2-2' }}</span></td>
                  <td class="num muted">{{ res()!.models[m].paper?.auc ?? '—' }}</td>
                  <td class="num">{{ res()!.models[m].ours_mean.pr_auc | number:'1.3-3' }}</td>
                  <td class="num">{{ res()!.models[m].ours_mean.f1 | number:'1.3-3' }}</td>
                  <td class="num">{{ res()!.models[m].ours_mean.mcc | number:'1.3-3' }}</td>
                  <td class="num">{{ res()!.models[m].ours_mean.recall | number:'1.3-3' }}</td>
                  <td class="num">{{ nem()!.mean_ranks[m] | number:'1.1-1' }}</td>
                </tr>
              }
            </tbody>
          </table>
        </div>
        <p class="t-muted t-12">Friedman p = {{ nem()!.friedman_p | number:'1.0-6' }} (models differ significantly).</p>
      </div>

      <div class="grid two">
        <app-chart-panel title="ROC curves (top 3, out-of-fold)" [options]="rocChart()" [height]="340">
          <table table><thead><tr><th>Model</th><th class="num">AUC (OOF)</th></tr></thead>
            <tbody>@for (m of roc()!.top3; track m) {
              <tr><td>{{ m }}</td><td class="num">{{ roc()!.curves[m].auc_oof | number:'1.3-3' }}</td></tr> }</tbody></table>
        </app-chart-panel>

        <app-chart-panel title="Nemenyi mean ranks (ROC-AUC; lower = better)" [options]="nemChart()" [height]="340">
          <table table><thead><tr><th>Model</th><th class="num">mean rank</th></tr></thead>
            <tbody>@for (r of rankRows(); track r.m) {
              <tr><td>{{ r.m }}</td><td class="num">{{ r.rank | number:'1.2-2' }}</td></tr> }</tbody></table>
        </app-chart-panel>
      </div>

      <div class="panel">
        <h3>Confusion matrix (out-of-fold)
          <select [(ngModel)]="selected" class="sel">
            @for (m of ranked(); track m) { <option [value]="m">{{ m }}</option> }
          </select></h3>
        <table class="cm">
          <thead><tr><th></th><th class="num">pred clean</th><th class="num">pred smelly</th></tr></thead>
          <tbody>
            <tr><td>actual clean</td><td class="num">{{ cm().tn | number }}</td><td class="num">{{ cm().fp | number }}</td></tr>
            <tr><td>actual smelly</td><td class="num">{{ cm().fn | number }}</td><td class="num">{{ cm().tp | number }}</td></tr>
          </tbody>
        </table>
      </div>

      <div class="panel">
        <h3>Improvement track (Ours, tuned)</h3>
        <table>
          <thead><tr><th>Model (tuned)</th><th class="num">AUC</th><th class="num">PR-AUC</th><th class="num">F1</th></tr></thead>
          <tbody>@for (m of tunedKeys(); track m) {
            <tr><td>{{ m }}</td>
              <td class="num">{{ imp()!.tuned_nested_cv[m].mean.roc_auc | number:'1.3-3' }}</td>
              <td class="num">{{ imp()!.tuned_nested_cv[m].mean.pr_auc | number:'1.3-3' }}</td>
              <td class="num">{{ imp()!.tuned_nested_cv[m].mean.f1 | number:'1.3-3' }}</td></tr> }</tbody>
        </table>
        <p class="t-muted t-12">Size ablation: XGBoost AUC {{ imp()!.size_ablation_xgboost.full_features.roc_auc | number:'1.3-3' }}
          → {{ imp()!.size_ablation_xgboost.ablated.roc_auc | number:'1.3-3' }} when the size-dominant metrics are dropped.</p>
      </div>
    } @else {
      <div class="panel"><div class="skeleton" style="height:300px"></div></div>
    }
  `,
  styles: [`
    .scroll { overflow-x: auto; }
    .std { color: var(--ink-muted); font-size: 11px; }
    .muted { color: var(--ink-muted); }
    tr.best td { background: var(--ok-tint); }
    .two { grid-template-columns: 1fr 1fr; } @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
    .sel { font: inherit; margin-left: 8px; padding: 2px 6px; border: 1px solid var(--rule); border-radius: var(--radius-sm); background: var(--surface); color: var(--ink); }
    .cm td, .cm th { text-align: right; } .cm td:first-child, .cm th:first-child { text-align: left; }
  `],
})
export class ModelsComponent implements OnInit {
  private results = inject(ResultsService);
  res = signal<Rq3Results | null>(null);
  nem = signal<Rq3Nemenyi | null>(null);
  roc = signal<Rq3Roc | null>(null);
  conf = signal<Rq3Confusion | null>(null);
  imp = signal<Rq3Improvements | null>(null);
  ready = signal(false);
  selected = 'XGBoost';

  ranked = computed(() => {
    const r = this.res(); if (!r) return [];
    return Object.keys(r.models).sort((a, b) => r.models[b].ours_mean.roc_auc - r.models[a].ours_mean.roc_auc);
  });
  best = computed(() => this.ranked()[0]);
  rankRows = computed(() => {
    const n = this.nem(); if (!n) return [];
    return Object.entries(n.mean_ranks).map(([m, rank]) => ({ m, rank })).sort((a, b) => a.rank - b.rank);
  });
  tunedKeys = computed(() => this.imp() ? Object.keys(this.imp()!.tuned_nested_cv) : []);

  ngOnInit(): void {
    forkJoin({
      res: this.results.get<Rq3Results>('rq3_results.json'),
      nem: this.results.get<Rq3Nemenyi>('rq3_nemenyi.json'),
      roc: this.results.get<Rq3Roc>('rq3_roc.json'),
      conf: this.results.get<Rq3Confusion>('rq3_confusion.json'),
      imp: this.results.get<Rq3Improvements>('rq3_improvements.json'),
    }).subscribe(d => {
      this.res.set(d.res); this.nem.set(d.nem); this.roc.set(d.roc);
      this.conf.set(d.conf); this.imp.set(d.imp); this.ready.set(true);
    });
  }

  cm() { return this.conf()?.[this.selected] ?? { tn: 0, fp: 0, fn: 0, tp: 0 }; }

  rocChart(): EChartsOption {
    const p = palette();
    const r = this.roc()!;
    const colors = categorical();
    return {
      grid: baseGrid(),
      tooltip: { trigger: 'axis' },
      legend: { data: r.top3, bottom: 0, textStyle: { color: p.muted } },
      xAxis: { type: 'value', name: 'FPR', min: 0, max: 1, ...axisBase() },
      yAxis: { type: 'value', name: 'TPR', min: 0, max: 1, ...axisBase() },
      series: [
        ...r.top3.map((m, i) => ({
          name: m, type: 'line' as const, showSymbol: false, smooth: true,
          data: r.curves[m].fpr.map((f, k) => [f, r.curves[m].tpr[k]]),
          lineStyle: { color: colors[i % colors.length], width: 3 }, itemStyle: { color: colors[i % colors.length] },
        })),
        { name: 'chance', type: 'line' as const, showSymbol: false, data: [[0, 0], [1, 1]],
          lineStyle: { color: p.rule, type: 'dashed' } },
      ],
    };
  }

  nemChart(): EChartsOption {
    const p = palette();
    const rows = [...this.rankRows()].reverse();
    return {
      grid: { left: 150, right: 20, top: 10, bottom: 30, containLabel: true },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'mean rank', ...axisBase() },
      yAxis: { type: 'category', data: rows.map(r => r.m), ...axisBase(), axisLabel: { color: p.muted, fontSize: 10 } },
      series: [{ type: 'bar',
        data: rows.map((r, i) => ({ value: r.rank, itemStyle: { color: barColors(rows.length)[i], borderRadius: [0, 4, 4, 0] } })) }],
    };
  }
}
