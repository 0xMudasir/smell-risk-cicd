// Typed interfaces for the pipeline JSON (frontend/src/assets/results/*.json)
// and the API. Kept intentionally permissive where the pipeline may add keys.

export interface Manifest {
  generated_at: string;
  files: string[];
  model: { name: string; version: string } | null;
}

export interface ModelCard {
  model: { name: string; version: string };
  trained_at: string;
  target: { name: string; definition: string };
  dataset: { name: string; rows: number; smelly_rate: number; n_projects: number };
  cv: { strategy: string; n_splits: number; group: string };
  cv_metrics_mean: Metrics;
  cv_metrics_std: Metrics;
  paper_reference_same_model?: PaperMetrics | null;
  features: string[];
  n_features: number;
  library_versions: Record<string, string>;
  notes: string;
}

export interface Metrics {
  accuracy: number; precision: number; recall: number;
  f1: number; mcc: number; roc_auc: number; pr_auc?: number;
}

// Paper (Table 15) metrics use `auc` rather than roc_auc/pr_auc.
export interface PaperMetrics {
  accuracy: number; precision: number; recall: number;
  f1: number; mcc: number; auc: number;
}

export interface FeatureSchema {
  order: string[];
  features: { name: string; dtype: string; min: number; median: number; max: number; description: string }[];
}

export interface DatasetAudit {
  shape: { rows: number; columns_total: number };
  target_has_smell: { positives: number; negatives: number; rate: number };
  per_smell: Record<string, { positives: number; rate: number }>;
  cooccurrence: Record<string, Record<string, number>>;
  projects: { n_projects: number; top: ProjectRow[] } | null;
  separability: Record<string, { clean_median: number; smelly_median: number; cohen_d: number | null }>;
  metric_map: Record<string, { family: string; paper_understand: string; note: string }>;
  correlated_pairs_gt_0_9: { a: string; b: string; r: number }[];
  granularity: any;
  columns: { numeric_metrics: string[]; smell_labels: string[] };
}
export interface ProjectRow { Project: string; files: number; smelly: number; smelly_rate: number; }

export interface Rq1 {
  features: string[];
  methods: Record<string, { ranked: { feature: string; score: number; rank: number }[]; top10: string[] }>;
  pearson_target: Record<string, number>;
  consensus: { appears_in_top10_counts: Record<string, number>; strict_intersection_5of5: string[]; at_least_4of5: string[] };
  fixes: { permutation_importance: { ranked: { feature: string; mean: number; std: number }[] }; shap: { ranked: { feature: string; score: number }[] }; correlated_pairs_gt_0_9: { a: string; b: string; r: number }[] };
  paper_comparison: { paper_core_metrics: string[]; note: string };
}

export interface Rq1Corr { features: string[]; matrix: number[][]; pearson_target: Record<string, number>; }

export interface Rq2Stats {
  consensus_metrics: string[];
  stats: Record<string, {
    clean: Quantiles; smelly: Quantiles;
    median_ratio_smelly_over_clean: number | string; approx_3x_rule: boolean;
  }>;
}
export interface Quantiles { count: number; mean: number; std: number; min: number; q1: number; median: number; q3: number; max: number; p90: number; p95: number; }

export interface Rq2Tests {
  consensus_metrics: string[];
  tests: Record<string, { full: MwuResult; balanced_undersample: MwuResult }>;
}
export interface MwuResult { U: number; p: number; p_holm: number; rank_biserial: number; effect_magnitude: string; }

export interface Rq2Thresholds {
  consensus_metrics: string[];
  thresholds: Record<string, ThresholdEntry>;
}
export interface ThresholdEntry {
  warning: number; critical: number; clean_median: number;
  ruler_quantiles: { min: number; clean_median: number; warning: number; critical: number; p95_smelly: number; max: number };
  paper_equivalent: { metric: string; warning: number; critical: number; note: string } | null;
  as_classifier_cv: Record<string, { threshold_mean: number; precision_mean: number; recall_mean: number; f1_mean: number }>;
}

export interface Rq3Results {
  metrics: string[];
  models: Record<string, { ours_mean: Metrics; ours_std: Metrics; paper: PaperMetrics | null; per_fold: Metrics[]; train_cap: number | null; deviation?: string }>;
}
export interface Rq3Nemenyi { metric: string; friedman_stat: number; friedman_p: number; mean_ranks: Record<string, number>; nemenyi_p_matrix: Record<string, Record<string, number>>; }
export interface Rq3Roc { top3: string[]; curves: Record<string, { fpr: number[]; tpr: number[]; auc_oof: number }>; }
export interface Rq3Confusion { [model: string]: { tn: number; fp: number; fn: number; tp: number }; }
export interface Rq3Improvements {
  tuned_nested_cv: Record<string, { mean: Metrics; std: Metrics }>;
  smote_vs_scale_pos_weight_xgboost: Record<string, any>;
  threshold_tuning_xgboost_oof: Record<string, any>;
  size_ablation_xgboost: { dropped_metrics: string[]; kept_metrics: string[]; full_features: Metrics; ablated: Metrics; note: string };
}

// --- API ---
export interface PredictResponse {
  probability: number; label: number; risk_tier: 'low' | 'elevated' | 'critical';
  threshold_flags: { metric: string; value: number; warning: number; critical: number; level: string }[];
  top_contributors: { metric: string; value: number; impact: number }[];
  model: { name: string; version: string };
  ignored?: string[];
  row?: number;
}
export interface Health { status: string; model_loaded: boolean; version: string; }
