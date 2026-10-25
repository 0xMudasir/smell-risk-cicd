import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, shareReplay } from 'rxjs';

// Reads the static pipeline JSON exported to assets/results. Cached per file so
// the research pages work with no API (CLAUDE.md 7.6).
@Injectable({ providedIn: 'root' })
export class ResultsService {
  private http = inject(HttpClient);
  private cache = new Map<string, Observable<any>>();
  private base = 'assets/results/';

  get<T>(file: string): Observable<T> {
    if (!this.cache.has(file)) {
      this.cache.set(file, this.http.get<T>(this.base + file).pipe(shareReplay(1)));
    }
    return this.cache.get(file)!;
  }
}
