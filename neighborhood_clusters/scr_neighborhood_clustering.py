from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from neighborhood_clusters.scr_neighborhood_features import data

X_cols = [
    'num_of_completed_units_condo',
    'average_psf_condo',
    'num_of_completed_units_hdb',
    'average_psf_hdb',
    'num_of_completed_units_landed',
    'average_psf_landed',
    'buyer_profile_company_percent',
    'buyer_profile_foreigner_percent',
    'buyer_profile_pr_percent',
    'buyer_profile_sg_percent'
]
X = data[X_cols].fillna(-1)

cs = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    kmeans.fit(X)
    cs.append(kmeans.inertia_)

plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


data['cluster'] = KMeans(10, random_state=42).fit_predict(X)