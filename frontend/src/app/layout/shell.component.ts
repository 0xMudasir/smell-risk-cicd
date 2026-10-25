import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { TopBarComponent } from './top-bar.component';
import { SideNavComponent } from './side-nav.component';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [RouterOutlet, TopBarComponent, SideNavComponent],
  template: `
    <app-top-bar></app-top-bar>
    <div class="body">
      <app-side-nav></app-side-nav>
      <main class="content"><router-outlet></router-outlet></main>
    </div>
  `,
  styles: [`
    :host { display: flex; flex-direction: column; min-height: 100vh; }
    .body { display: flex; flex: 1; align-items: stretch; }
    .content { flex: 1; min-width: 0; padding: 20px; max-width: var(--content-max); width: 100%; }
    @media (max-width: 900px) { .body { flex-direction: column; } }
  `],
})
export class ShellComponent {}
