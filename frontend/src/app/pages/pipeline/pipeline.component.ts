import { Component } from '@angular/core';
import { buildInfo } from '../../../environments/build-info';

@Component({
  selector: 'app-pipeline',
  standalone: true,
  template: `
    <h1>Pipeline</h1>
    <p class="t-muted">The lab workflow end to end: develop in VS Code, push to GitHub, CI builds and tests,
      CD deploys the frontend to GitHub Pages, and a local Flask API serves predictions.</p>

    <ol class="steps">
      @for (s of steps; track s.n) {
        <li class="panel step">
          <span class="n">{{ s.n }}</span>
          <div><div class="st">{{ s.title }}</div><div class="t-muted t-13">{{ s.detail }}</div></div>
        </li>
      }
    </ol>

    <div class="panel">
      <h3>Current build</h3>
      <p class="mono">SHA {{ build.sha }}{{ build.runNumber ? ' · run #' + build.runNumber : '' }}
        {{ build.builtAt ? ' · ' + build.builtAt : ' · local dev build' }}</p>
      <p class="t-muted t-12">The build tag changes after each push once CI/CD is live (proving CD worked).
        Repo, Actions and live-site links are added when the repository is pushed (Phase 9).</p>
    </div>
  `,
  styles: [`
    .steps { list-style: none; padding: 0; display: flex; flex-direction: column; gap: 10px; }
    .step { display: flex; align-items: flex-start; gap: 12px; }
    .n { flex: none; width: 26px; height: 26px; border-radius: 50%; background: var(--accent); color: #fff;
         display: grid; place-items: center; font-size: 13px; font-weight: 600; }
    .st { font-weight: 600; }
    .mono { font-family: var(--font-mono); }
  `],
})
export class PipelineComponent {
  build = buildInfo;
  steps = [
    { n: 1, title: 'Develop in VS Code', detail: 'Python ML pipeline + Flask API + Angular app on Windows.' },
    { n: 2, title: 'Commit and push', detail: 'Phased, stop-gated commits to the main branch on GitHub.' },
    { n: 3, title: 'GitHub repository', detail: 'Source of truth; raw dataset and large files are gitignored.' },
    { n: 4, title: 'CI (GitHub Actions)', detail: 'compileall + pytest (backend + model smoke) and an Angular production build.' },
    { n: 5, title: 'CD to GitHub Pages', detail: 'On green CI, generate build-info, build with the repo base-href, deploy the browser bundle.' },
    { n: 6, title: 'GitHub Pages site', detail: 'Static research pages render from exported JSON; Predict calls the API.' },
    { n: 7, title: 'Flask ML API', detail: 'Runs locally (python backend/app.py); the top bar shows API online/offline.' },
    { n: 8, title: 'End to end', detail: 'Enter metrics or upload a CSV, get a smell-risk prediction with threshold rulers.' },
  ];
}
