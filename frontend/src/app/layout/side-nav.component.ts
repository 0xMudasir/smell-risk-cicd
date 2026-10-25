import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { LucideAngularModule, LayoutGrid, Database, ListTree, Ruler, BarChart3, Play, GitBranch } from 'lucide-angular';

@Component({
  selector: 'app-side-nav',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, LucideAngularModule],
  template: `
    <nav class="rail" aria-label="Sections">
      @for (item of items; track item.path) {
        <a [routerLink]="item.path" routerLinkActive="active" class="link">
          <lucide-angular [img]="item.icon" class="ico" [size]="18"></lucide-angular>
          <span>{{ item.label }}</span>
        </a>
      }
    </nav>
  `,
  styles: [`
    .rail { display: flex; flex-direction: column; gap: 2px; padding: 12px 8px; width: 220px;
            border-right: 1px solid var(--rule); background: var(--surface); min-height: 100%; }
    .link { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: var(--radius-sm);
            color: var(--ink); font-size: 14px; }
    .link:hover { background: var(--paper); text-decoration: none; }
    .link.active { background: var(--paper); color: var(--accent); font-weight: 600;
                   box-shadow: inset 2px 0 0 var(--accent); }
    .ico { color: currentColor; }
    @media (max-width: 900px) {
      .rail { flex-direction: row; width: 100%; overflow-x: auto; border-right: none;
              border-bottom: 1px solid var(--rule); padding: 8px; min-height: 0; }
      .link span { white-space: nowrap; }
    }
  `],
})
export class SideNavComponent {
  items = [
    { path: '/overview', label: 'Overview', icon: LayoutGrid },
    { path: '/dataset', label: 'Dataset', icon: Database },
    { path: '/features', label: 'Features', icon: ListTree },
    { path: '/thresholds', label: 'Thresholds', icon: Ruler },
    { path: '/models', label: 'Models', icon: BarChart3 },
    { path: '/predict', label: 'Predict', icon: Play },
    { path: '/pipeline', label: 'Pipeline', icon: GitBranch },
  ];
}
