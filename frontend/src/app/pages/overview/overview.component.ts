import { Component, OnInit, inject, signal } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { ResultsService } from '../../core/results.service';
import { DatasetAudit, ModelCard, Rq1, Rq3Results, Rq3Improvements } from '../../core/models';

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [DecimalPipe, PercentPipe],
  template: `
    <h1>Replicating Diouf et al. (2026) on a code smell dataset</h1>
    <p class="t-muted intro">
      We apply the paper's method — rank metrics by importance (RQ1), test how the
      smelly class differs and derive thresholds (RQ2), and compare 11 classifiers
      with grouped cross-validation (RQ3) — to <strong>SmellyCode++</strong>, a
      code-smell dataset. The target is code smells, not security bugs, so we
      compare <em>patterns</em>, not exact numbers.
    </p>

    @if (loaded()) {
      <div class="row cards">
        <div class="panel card">
          <div class="k">Dataset</div>
          <div class="v">{{ rows() | number }} rows</div>
          <div class="s t-muted">{{ smelly() | percent:'1.1-1' }} smelly · {{ projects() }} projects</div>
        </div>
        <div class="panel card">
          <div class="k">Best model</div>
          <div class="v">{{ bestName() }}</div>
          <div class="s t-muted">ROC-AUC {{ bestAuc() | number:'1.3-3' }} (grouped CV)</div>
        </div>
        <div class="panel card">
          <div class="k">Consensus metrics</div>
          <div class="v">{{ coreCount() }}</div>
          <div class="s t-muted mono">{{ coreList() }}</div>
        </div>
      </div>

      <div class="panel">
        <h3>Paper vs ours (best model)</h3>
        <div class="scroll-x">
        <table>
          <thead><tr><th>Source</th><th class="num">Acc</th><th class="num">Prec</th>
            <th class="num">Rec</th><th class="num">F1</th><th class="num">MCC</th><th class="num">AUC</th></tr></thead>
          <tbody>
            <tr><td>Paper (security bugs)</td>
              <td class="num">{{ paper()?.accuracy | number:'1.2-2' }}</td>
              <td class="num">{{ paper()?.precision | number:'1.2-2' }}</td>
              <td class="num">{{ paper()?.recall | number:'1.2-2' }}</td>
              <td class="num">{{ paper()?.f1 | number:'1.2-2' }}</td>
              <td class="num">{{ paper()?.mcc | number:'1.2-2' }}</td>
              <td class="num">{{ paper()?.auc | number:'1.2-2' }}</td></tr>
            <tr><td>Ours, replication</td>
              <td class="num">{{ rep()?.accuracy | number:'1.3-3' }}</td>
              <td class="num">{{ rep()?.precision | number:'1.3-3' }}</td>
              <td class="num">{{ rep()?.recall | number:'1.3-3' }}</td>
              <td class="num">{{ rep()?.f1 | number:'1.3-3' }}</td>
              <td class="num">{{ rep()?.mcc | number:'1.3-3' }}</td>
              <td class="num">{{ rep()?.roc_auc | number:'1.3-3' }}</td></tr>
            <tr><td>Ours, tuned</td>
              <td class="num">—</td><td class="num">—</td>
              <td class="num">{{ tuned()?.recall | number:'1.3-3' }}</td>
              <td class="num">{{ tuned()?.f1 | number:'1.3-3' }}</td>
              <td class="num">{{ tuned()?.mcc | number:'1.3-3' }}</td>
              <td class="num">{{ tuned()?.roc_auc | number:'1.3-3' }}</td></tr>
          </tbody>
        </table>
        </div>
        <p class="t-muted t-12">Different target (code smells), so compare patterns, not exact numbers.
          Tuned = nested RandomizedSearchCV; accuracy/precision omitted (threshold-dependent).</p>
      </div>
    } @else {
      <div class="panel"><div class="skeleton" style="height:120px"></div></div>
    }
  `,
  styles: [`
    .intro { max-width: 70ch; }
    .cards { margin: 16px 0; }
    .card { flex: 1; min-width: 220px; }
    .card .k { font-size: 12px; color: var(--ink-muted); text-transform: none; }
    .card .v { font-size: 24px; font-weight: 600; margin: 4px 0; }
    .card .s { font-size: 12px; }
    .mono { font-family: var(--font-mono); font-size: 11px; }
  `],
})
export class OverviewComponent implements OnInit {
  private results = inject(ResultsService);
  loaded = signal(false);
  rows = signal(0); smelly = signal(0); projects = signal(0);
  bestName = signal(''); bestAuc = signal(0);
  coreList = signal(''); coreCount = signal(0);
  paper = signal<any>(null); rep = signal<any>(null); tuned = signal<any>(null);

  ngOnInit(): void {
    this.results.get<ModelCard>('model_card.json').subscribe(card => {
      this.bestName.set(card.model.name);
      this.bestAuc.set(card.cv_metrics_mean.roc_auc);
      this.rows.set(card.dataset.rows);
      this.smelly.set(card.dataset.smelly_rate);
      this.projects.set(card.dataset.n_projects);
      this.rep.set(card.cv_metrics_mean);
      this.paper.set(card.paper_reference_same_model);
      this.loaded.set(true);
    });
    this.results.get<Rq1>('rq1_importance.json').subscribe(r => {
      const c = r.consensus.at_least_4of5;
      this.coreCount.set(c.length);
      this.coreList.set(c.join(', '));
    });
    this.results.get<Rq3Improvements>('rq3_improvements.json').subscribe(i => {
      this.tuned.set(i.tuned_nested_cv['XGBoost']?.mean ?? null);
    });
  }
}
