from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go


@dataclass
class KMeansCluster:
    features: list
    num_of_clusters: int
    data: pd.DataFrame

    def __post_init__(self):
        training_X = self.data[self.features].fillna(-1)
        self.data['cluster'] = KMeans(
            self.num_of_clusters, random_state=42, n_init=10
        ).fit_predict(training_X).astype(str)

    @property
    def clusters_list(self):
        return [
            str(i) for i in np.arange(self.num_of_clusters)
        ]

    @property
    def color_discrete_map(self):
        color_discrete_map = {
            str(clu): col
            for clu, col in zip(
                self.clusters_list,
                px.colors.qualitative.Pastel1[:-1] +
                px.colors.qualitative.Set3
            )
        }
        return color_discrete_map

    def plot_clusters(self):
        hover_data = {
            i: ':.2f' for i in self.features
        }
        hover_data['cluster'] = True
        hover_data['label'] = True

        fig = px.choropleth_mapbox(
            self.data.fillna(-1),
            color='cluster',
            geojson=self.data.geometry,
            locations=self.data.index,
            hover_name=self.data.label,
            hover_data=hover_data,
            color_discrete_map=self.color_discrete_map,
            category_orders={'cluster': list(self.color_discrete_map.keys())},
            mapbox_style="carto-positron",
            zoom=10,
            center={"lat": 1.290270, "lon": 103.851959},
            opacity=0.5
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            autosize=False,
            width=1200,
            height=600
        )

        return fig

    def plot_violin(self, feature):
        fig = go.Figure()
        for c in self.clusters_list:
            cluster_data = self.data[self.data['cluster'] == c]

            fig.add_trace(
                go.Violin(
                    x=cluster_data['cluster'],
                    y=cluster_data[feature],
                    name=c,
                    side='negative',
                    line_color=self.color_discrete_map[c]
                )
            )

        fig.update_traces(
            meanline_visible=True,
            points='all',  # show all points
            jitter=0.05,  # add some jitter on points for better visibility
            scalemode='count'
        )  # scale violin plot area with total count
        fig.update_layout(
            title_text=f"{feature.replace('_', ' ')} distribution",
            violingap=0,
            violingroupgap=0,
            violinmode='overlay',
            autosize=False,
            width=1200,
            height=600
        )

        return fig

    def plot_histogram(self, feature):
        fig = px.histogram(
            self.data,
            y=feature,
            facet_col="cluster",
            color="cluster",
            color_discrete_map=self.color_discrete_map,
            category_orders={'cluster': list(self.color_discrete_map.keys())},
            marginal="rug",
            # opacity=0.6
            # hover_data=self.data.columns
        )

        fig.update_layout(
            title_text=f"{feature.replace('_', ' ')} distribution",
            autosize=False,
            width=1200,
            height=600
        )

        return fig

if False:
    from neighborhood_clusters.scr_neighborhood_features import df as data

    selectbox_feature = [
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

    cluster_model = KMeansCluster(
        features=selectbox_feature,
        num_of_clusters=10,
        data=data.copy()
    )

    A = cluster_model.plot_histogram('average_psf_condo')

    A.show()
