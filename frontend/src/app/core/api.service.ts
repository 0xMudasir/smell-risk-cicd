import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { FeatureSchema, Health, ModelCard, PredictResponse } from './models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  readonly base = environment.apiBaseUrl;

  health(): Observable<Health> { return this.http.get<Health>(`${this.base}/api/health`); }
  modelInfo(): Observable<ModelCard & { features: string[] }> {
    return this.http.get<ModelCard & { features: string[] }>(`${this.base}/api/model-info`);
  }
  schema(): Observable<FeatureSchema> { return this.http.get<FeatureSchema>(`${this.base}/api/schema`); }

  predict(metrics: Record<string, number>): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(`${this.base}/api/predict`, { metrics });
  }
  predictBatch(file: File): Observable<{ count: number; results: PredictResponse[] }> {
    const fd = new FormData();
    fd.append('file', file);
    return this.http.post<{ count: number; results: PredictResponse[] }>(`${this.base}/api/predict/batch`, fd);
  }
}
