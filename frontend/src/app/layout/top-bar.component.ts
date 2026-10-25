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
    .top { display: flex; align-items: center; gap: 16px; padding: 12px 20px;
           background: var(--grad-brand); color: #fff; box-shadow: var(--shadow-md); }
    .title { font-weight: 700; font-size: 17px; letter-spacing: .2px; color: #fff; }
    .meta { margin-left: auto; display: flex; align-items: center; gap: 12px; font-size: 13px;
            color: rgba(255,255,255,0.88); flex-wrap: wrap; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
            border: 1px solid rgba(255,255,255,0.35); border-radius: 999px;
            background: rgba(255,255,255,0.12); color: #fff; }
    .pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #fff; }
    .pill.on { background: rgba(255,255,255,0.18); } .pill.on .dot { background: #4ADE80; box-shadow: 0 0 8px #4ADE80; }
    .pill.off .dot { background: #FB7185; box-shadow: 0 0 8px #FB7185; }
    .build { font-family: var(--font-mono); font-size: 12px; background: rgba(255,255,255,0.12);
             padding: 3px 8px; border-radius: 999px; }
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
