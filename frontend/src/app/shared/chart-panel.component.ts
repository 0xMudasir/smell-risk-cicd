import { Component, Input, signal } from '@angular/core';
import { NgxEchartsDirective } from 'ngx-echarts';
import type { EChartsOption } from 'echarts';

// Chart + "View as table" toggle (CLAUDE.md 7.6 / 7.8: every chart has a table
// fallback). Pages project the table via [table].
@Component({
  selector: 'app-chart-panel',
  standalone: true,
  imports: [NgxEchartsDirective],
  template: `
    <div class="panel chart-panel">
      <div class="cp-head">
        <h3>{{ title }}</h3>
        <button class="btn" (click)="mode.set(mode() === 'chart' ? 'table' : 'chart')">
          {{ mode() === 'chart' ? 'View as table' : 'View as chart' }}
        </button>
      </div>
      @if (mode() === 'chart') {
        <div echarts [options]="options" [autoResize]="true" class="chart" [style.height.px]="height"></div>
      } @else {
        <div class="table-wrap"><ng-content select="[table]"></ng-content></div>
      }
    </div>
  `,
  styles: [`
    .chart-panel { padding: 12px 16px 16px; }
    .cp-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
    .cp-head h3 { margin: 0; }
    .chart { width: 100%; }
    .table-wrap { overflow-x: auto; }
  `],
})
export class ChartPanelComponent {
  @Input() title = '';
  @Input({ required: true }) options!: EChartsOption;
  @Input() height = 320;
  mode = signal<'chart' | 'table'>('chart');
}
