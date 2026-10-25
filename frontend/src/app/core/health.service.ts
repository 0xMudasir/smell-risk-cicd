import { Injectable, inject, signal } from '@angular/core';
import { ApiService } from './api.service';

// Polls /api/health every 30s and exposes an 'online' signal for the top bar.
@Injectable({ providedIn: 'root' })
export class HealthService {
  private api = inject(ApiService);
  readonly online = signal<boolean | null>(null); // null = unknown/checking
  private started = false;

  start(): void {
    if (this.started) return;
    this.started = true;
    this.check();
    setInterval(() => this.check(), 30_000);
  }

  check(): void {
    this.api.health().subscribe({
      next: (h) => this.online.set(h.model_loaded === true),
      error: () => this.online.set(false),
    });
  }
}
