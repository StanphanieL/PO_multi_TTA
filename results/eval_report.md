Evaluating category-only subset: "bottle1" | samples: 34 (was 1312)
[Cluster] cluster_norm=True -> Predicted clusters will be used for conditioning and per-cluster metrics.
Test samples: 34
[Cluster-Conditioning][Summary] used_predicted=34/34, fallback_true=0, fallback_unconditional=0
[Cluster-Class][Confusion] valid=34 classes=40 clusters=40
[Cluster-Class][Index-consistency] diag_frac=1.0000 (assuming cluster_id == class_id)
  cluster 3 -> class 3 hit=34/34 (100.00%)
[Global] pos_rate=0.014521
[Global] object AUC-ROC: 0.9263157894736842, point AUC-ROC: 0.9372509218574376, object AUCP-PR: 0.931794725966216, point AUCP-PR: 0.41304588087464916
[TTA-Time] pre_infer_total=9.1434s, per_sample=0.2689s
[TTA-Time] tta_infer_total=38.3971s, per_sample=1.1293s, views=4
[Global-preTTA] object AUC-ROC: 0.8912280701754385, point AUC-ROC: 0.905794575638584, object AUCP-PR: 0.9185883506517614, point AUCP-PR: 0.3756179248973945
[Global-postTTA] object AUC-ROC: 0.9263157894736842, point AUC-ROC: 0.9372509218574376, object AUCP-PR: 0.931794725966216, point AUCP-PR: 0.41304588087464916
[Sample-macro] point AP (macro over samples) = 0.49044755606047713

[Per-Cluster metrics with cluster-wise mad normalization]
  [cluster 3] N=34 objAUC=0.9263157894736842 ptAUC=0.9372509218131686 objAP=0.931794725966216 ptAP=0.4130458893810359 pos_rate=0.014521
  [cluster-macro] objAUC=0.9263157894736842 ptAUC=0.9372509218131686 objAP=0.931794725966216 ptAP=0.4130458893810359

[Per-Category metrics (by true category) with category-wise mad normalization]
  [cat 3] N=34 objAUC=0.9263157894736842 ptAUC=0.9372509218131686 objAP=0.931794725966216 ptAP=0.4130458893810359 pos_rate=0.014521
  [category-macro] objAUC=0.9263157894736842 ptAUC=0.9372509218131686 objAP=0.931794725966216 ptAP=0.4130458893810359
[Metrics CSV] saved to ./resulst/metrics_mad_2.csv
