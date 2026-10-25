import { Routes } from '@angular/router';
import { ShellComponent } from './layout/shell.component';

export const routes: Routes = [
  {
    path: '',
    component: ShellComponent,
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'overview' },
      { path: 'overview', loadComponent: () => import('./pages/overview/overview.component').then(m => m.OverviewComponent) },
      { path: 'dataset', loadComponent: () => import('./pages/dataset/dataset.component').then(m => m.DatasetComponent) },
      { path: 'features', loadComponent: () => import('./pages/features/features.component').then(m => m.FeaturesComponent) },
      { path: 'thresholds', loadComponent: () => import('./pages/thresholds/thresholds.component').then(m => m.ThresholdsComponent) },
      { path: 'models', loadComponent: () => import('./pages/models/models.component').then(m => m.ModelsComponent) },
      { path: 'predict', loadComponent: () => import('./pages/predict/predict.component').then(m => m.PredictComponent) },
      { path: 'pipeline', loadComponent: () => import('./pages/pipeline/pipeline.component').then(m => m.PipelineComponent) },
    ],
  },
  { path: '**', redirectTo: 'overview' },
];
