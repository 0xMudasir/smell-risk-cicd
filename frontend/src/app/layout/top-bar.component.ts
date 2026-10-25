import { Component, OnInit, inject, signal } from '@angular/core';
import { ResultsService } from '../core/results.service';
import { HealthService } from '../core/health.service';
import { Manifest } from '../core/models';
import { buildInfo } from '../../environments/build-info';

@Component({
  selector: 'app-top-bar',
  standalone: true,
  template: `
    <header class="top">
      <div class="title">Smell Risk Dashboard</div>
      <div class="meta">
        <span class="model">Model {{ model() }}</span>
        <span class="pill" [class.on]="online() === true" [class.off]="online() === false">
          <span class="dot"></span>{{ online() === null ? 'API checking' : online() ? 'API online' : 'API offline' }}
        </span>
        <span class="build" title="Build commit and Actions run number">
          Build {{ build.sha }}{{ build.runNumber ? ' #' + build.runNumber : '' }}
        </span>
      </div>
    </header>
  `,
  styles: [`
    .top { display: flex; align-items: center; gap: 16px; padding: 10px 20px;
           border-bottom: 1px solid var(--rule); background: var(--surface); }
    .title { font-weight: 600; font-size: 16px; }
    .meta { margin-left: auto; display: flex; align-items: center; gap: 14px; font-size: 13px; color: var(--ink-muted); flex-wrap: wrap; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px;
            border: 1px solid var(--rule); border-radius: var(--radius-sm); }
    .pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ink-muted); }
    .pill.on { color: var(--ok); border-color: var(--ok); } .pill.on .dot { background: var(--ok); }
    .pill.off { color: var(--crit); border-color: var(--crit); } .pill.off .dot { background: var(--crit); }
    .build { font-family: var(--font-mono); font-size: 12px; }
    @media (max-width: 700px) { .model { display: none; } }
  `],
})
export class TopBarComponent implements OnInit {
  private results = inject(ResultsService);
  private health = inject(HealthService);
  readonly build = buildInfo;
  readonly model = signal('...');
  readonly online = this.health.online;

  ngOnInit(): void {
    this.health.start();
    this.results.get<Manifest>('manifest.json').subscribe({
      next: (m) => this.model.set(m.model ? `${m.model.name} v${m.model.version}` : 'n/a'),
      error: () => this.model.set('n/a'),
    });
  }
}
