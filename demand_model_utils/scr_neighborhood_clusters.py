from neighborhood_clusters.cls_core_model import KMeansCluster
from neighborhood_clusters.scr_neighborhood_features_v2 import df as data

clustering_features = [
    'avg_psf_new_sale_condo',
    'avg_psf_resale_condo',
    'avg_psf_resale_hdb',
    'avg_psf_new_sale_landed',
    'avg_age_completed_condo',
    'avg_size_completed_condo',
]

cluster_model = KMeansCluster(
    features=clustering_features,
    num_of_clusters=10,
    data=data.copy()
)

clustering_res = cluster_model.data[['label', 'cluster']].copy()
clustering_res['neighborhood'] = clustering_res['label'].apply(lambda l: l.lower())
