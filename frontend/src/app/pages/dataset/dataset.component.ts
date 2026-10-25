import { Component, OnInit, inject, signal } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import type { EChartsOption } from 'echarts';
import { ResultsService } from '../../core/results.service';
import { ChartPanelComponent } from '../../shared/chart-panel.component';
import { MetricNameComponent } from '../../shared/metric-name.component';
import { DatasetAudit, ProjectRow } from '../../core/models';
import { palette, axisBase, baseGrid, barColors } from '../../core/theme';

@Component({
  selector: 'app-dataset',
  standalone: true,
  imports: [DecimalPipe, PercentPipe, ChartPanelComponent, MetricNameComponent],
  template: `
    <h1>Dataset</h1>
    @if (audit()) {
      <p class="t-muted">SmellyCode++: {{ audit()!.shape.rows | number }} entities,
        {{ audit()!.target_has_smell.rate | percent:'1.1-1' }} smelly
        ({{ audit()!.target_has_smell.positives | number }} of {{ audit()!.shape.rows | number }}).</p>

      <div class="grid two">
        <app-chart-panel title="Per-smell positive rate" [options]="smellBars()" [height]="260">
          <table table>
            <thead><tr><th>Smell</th><th class="num">positives</th><th class="num">rate</th></tr></thead>
            <tbody>
              @for (s of smellRows(); track s.name) {
                <tr><td><app-metric-name [name]="s.name"/></td>
                  <td class="num">{{ s.positives | number }}</td>
                  <td class="num">{{ s.rate | percent:'1.2-2' }}</td></tr>
              }
            </tbody>
          </table>
        </app-chart-panel>

        <app-chart-panel title="Smell co-occurrence (rows positive for both)" [options]="cooc()" [height]="260">
          <table table>
            <thead><tr><th></th>@for (c of smells(); track c){<th class="num">{{ c }}</th>}</tr></thead>
            <tbody>
              @for (a of smells(); track a) {
                <tr><td>{{ a }}</td>
                  @for (b of smells(); track b) { <td class="num">{{ audit()!.cooccurrence[a][b] | number }}</td> }
                </tr>
              }
            </tbody>
          </table>
        </app-chart-panel>
      </div>

      @if (audit()!.projects) {
        <div class="panel">
          <h3>Projects (top 25 by file count) — {{ audit()!.projects!.n_projects }} total</h3>
          <div class="scroll">
            <table>
              <thead><tr><th>Project</th><th class="num">files</th><th class="num">smelly</th><th class="num">rate</th></tr></thead>
              <tbody>
                @for (p of projects(); track p.Project) {
                  <tr><td class="mono">{{ p.Project }}</td>
                    <td class="num">{{ p.files | number }}</td>
                    <td class="num">{{ p.smelly | number }}</td>
                    <td class="num">{{ p.smelly_rate | percent:'1.1-1' }}</td></tr>
                }
              </tbody>
            </table>
          </div>
        </div>
      }
    } @else {
      <div class="panel"><div class="skeleton" style="height:200px"></div></div>
    }
  `,
  styles: [`
    .two { grid-template-columns: 1fr 1fr; }
    @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
    .scroll { max-height: 420px; overflow: auto; }
    .mono { font-family: var(--font-mono); font-size: 12px; }
  `],
})
export class DatasetComponent implements OnInit {
  private results = inject(ResultsService);
  audit = signal<DatasetAudit | null>(null);
  smells = signal<string[]>([]);
  smellRows = signal<{ name: string; positives: number; rate: number }[]>([]);
  projects = signal<ProjectRow[]>([]);

  ngOnInit(): void {
    this.results.get<DatasetAudit>('dataset_audit.json').subscribe(a => {
      this.audit.set(a);
      const names = Object.keys(a.per_smell);
      this.smells.set(names);
      this.smellRows.set(names.map(n => ({ name: n, positives: a.per_smell[n].positives, rate: a.per_smell[n].rate })));
      this.projects.set(a.projects?.top ?? []);
    });
  }

  smellBars(): EChartsOption {
    const p = palette();
    const rows = this.smellRows();
    return {
      grid: baseGrid(),
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: rows.map(r => r.name), ...axisBase(), axisLabel: { color: p.muted, fontSize: 10, interval: 0, rotate: 20 } },
      yAxis: { type: 'value', name: 'rate', ...axisBase(), axisLabel: { formatter: (v: number) => (v * 100).toFixed(1) + '%', color: p.muted } },
      series: [{ type: 'bar', barWidth: '55%',
        data: rows.map((r, i) => ({ value: r.rate, itemStyle: { color: barColors(rows.length)[i], borderRadius: [6, 6, 0, 0] } })) }],
    };
  }

  cooc(): EChartsOption {
    const p = palette();
    const s = this.smells();
    const a = this.audit()!;
    const data: [number, number, number][] = [];
    s.forEach((ra, i) => s.forEach((rb, j) => data.push([j, i, a.cooccurrence[ra][rb]])));
    const max = Math.max(...data.map(d => d[2]), 1);
    return {
      grid: { left: 110, right: 20, top: 10, bottom: 70 },
      tooltip: { position: 'top' },
      xAxis: { type: 'category', data: s, ...axisBase(), axisLabel: { color: p.muted, fontSize: 10, rotate: 25 } },
      yAxis: { type: 'category', data: s, ...axisBase(), axisLabel: { color: p.muted, fontSize: 10 } },
      visualMap: { min: 0, max, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
        inRange: { color: ['#EEF2FF', '#818CF8', '#4338CA'] }, textStyle: { color: p.muted } },
      series: [{ type: 'heatmap', data, itemStyle: { borderColor: p.surface, borderWidth: 2 },
        label: { show: true, color: p.ink, fontSize: 10 } }],
    };
  }
}
