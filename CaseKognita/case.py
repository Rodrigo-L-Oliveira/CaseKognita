import pandas as pd
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from pysal.lib import weights
from pysal.explore import esda
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import numpy as np

# Carregar os dados usando pandas
pontos_interesse_df = pd.read_parquet('_data/pontos_interesse_geograficos.parquet')
unidades_faturamento = pd.read_parquet('_data/unidades_faturamento.parquet')

# Carregar o shapefile dos municípios
municipios_gdf = gpd.read_file('_data/SP_Municipios.shp')

# Filtrar pontos de interesse para buscar 'comercio'
pontos_interesse_comercio = pontos_interesse_df[pontos_interesse_df['tipo_negocio'].str.contains('comercio', case=False)]

# Converter para GeoDataFrame
if 'longitude' in pontos_interesse_comercio.columns and 'latitude' in pontos_interesse_comercio.columns:
    pontos_interesse = gpd.GeoDataFrame(
        pontos_interesse_comercio,
        geometry=gpd.points_from_xy(pontos_interesse_comercio.longitude, pontos_interesse_comercio.latitude),
        crs="EPSG:4326"
    )
else:
    raise KeyError("As colunas 'longitude' e 'latitude' não foram encontradas em pontos_interesse_df")

# Limpar os dados
pontos_interesse.dropna(inplace=True)
unidades_faturamento.dropna(inplace=True)

# Transformar coordenadas em CRS do município para fazer o join espacial
pontos_interesse = pontos_interesse.to_crs(municipios_gdf.crs)

# Realizar o join espacial entre pontos de interesse e municípios
pontos_interesse_municipio = gpd.sjoin(pontos_interesse, municipios_gdf, how='inner', predicate='intersects')

# Agrupar por município utilizando NM_MUN e contar pontos de interesse
pontos_interesse_municipio_agg = pontos_interesse_municipio.groupby('NM_MUN').size().reset_index(name='num_pois')

# Calcular a densidade de Pontos de Interesse por município
municipios_gdf = municipios_gdf.merge(pontos_interesse_municipio_agg, on='NM_MUN', how='left')
municipios_gdf['num_pois'] = municipios_gdf['num_pois'].fillna(0)
municipios_gdf['densidade_pois'] = municipios_gdf['num_pois'] / municipios_gdf['AREA_KM2']

# Identificar as top 5 categorias de Pontos de Interesse
top5_categorias = pontos_interesse_comercio['tipo_negocio'].value_counts().nlargest(5).index

# Filtrar apenas as top 5 categorias
pontos_interesse_top5 = pontos_interesse[pontos_interesse['tipo_negocio'].isin(top5_categorias)]

# Calcular o faturamento médio por unidade
unidades_faturamento = unidades_faturamento.melt(id_vars=['id_unidade', 'latitude', 'longitude'], 
                                                 var_name='data', value_name='faturamento')
unidades_faturamento['data'] = pd.to_datetime(unidades_faturamento['data'])
faturamento_medio = unidades_faturamento.groupby(['id_unidade'])['faturamento'].mean().reset_index()

# Realizar o join espacial entre dados de faturamento e municípios
unidades_faturamento = gpd.GeoDataFrame(unidades_faturamento, geometry=gpd.points_from_xy(unidades_faturamento.longitude, unidades_faturamento.latitude), crs="EPSG:4326")
unidades_faturamento = unidades_faturamento.to_crs(municipios_gdf.crs)
faturamento_municipio = gpd.sjoin(unidades_faturamento, municipios_gdf, how='inner', predicate='intersects')
faturamento_municipio_agg = faturamento_municipio.groupby('NM_MUN')['faturamento'].mean().reset_index(name='faturamento_medio')

# 1. Determinar os pontos de interesse geográficos (POIs) localizados dentro de cada município, e mapear as categorias relacionadas a potenciais concorrentes.
# Já implementado anteriormente no join espacial e agrupamento.

# 2. Plotar um mapa da quantidade total de POIs por município.
fig = px.choropleth(
    municipios_gdf,
    geojson=municipios_gdf.geometry,
    locations=municipios_gdf.index,
    color="num_pois",
    hover_name="NM_MUN",
    title="Quantidade de POIs por Município",
    color_continuous_scale="Viridis"
)
fig.update_geos(fitbounds="locations", visible=False)
fig.show()

# 3. Para cada uma das top 5 categorias de POIs, fazer o mesmo que o item 2.
for categoria in top5_categorias:
    subset = pontos_interesse_top5[pontos_interesse_top5['tipo_negocio'] == categoria]
    subset_municipio = gpd.sjoin(subset, municipios_gdf, how='inner', predicate='intersects')
    subset_municipio_agg = subset_municipio.groupby('NM_MUN').size().reset_index(name='num_pois')
    
    municipios_subset = municipios_gdf.merge(subset_municipio_agg, on='NM_MUN', how='left')
    municipios_subset['num_pois'] = municipios_subset['num_pois'].fillna(0)
    
    if 'num_pois' in municipios_subset.columns:
        fig = px.choropleth(
            municipios_subset,
            geojson=municipios_subset.geometry,
            locations=municipios_subset.index,
            color="num_pois",
            hover_name="NM_MUN",
            title=f"Quantidade de POIs de {categoria} por Município",
            color_continuous_scale="Viridis"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.show()
    else:
        print(f"Coluna 'num_pois' não encontrada para a categoria {categoria}")

# 4. Para as categorias de POIs que foram identificadas como possíveis concorrentes, fazer o mesmo que o item 2.
concorrentes_categorias = ['categoria1', 'categoria2', 'categoria3']  # Ajuste conforme necessário
subset_concorrentes = pontos_interesse[pontos_interesse['tipo_negocio'].isin(concorrentes_categorias)]
subset_concorrentes_municipio = gpd.sjoin(subset_concorrentes, municipios_gdf, how='inner', predicate='intersects')
subset_concorrentes_municipio_agg = subset_concorrentes_municipio.groupby('NM_MUN').size().reset_index(name='num_pois')

municipios_concorrentes = municipios_gdf.merge(subset_concorrentes_municipio_agg, on='NM_MUN', how='left')
municipios_concorrentes['num_pois'] = municipios_concorrentes['num_pois'].fillna(0)

fig = px.choropleth(
    municipios_concorrentes,
    geojson=municipios_concorrentes.geometry,
    locations=municipios_concorrentes.index,
    color="num_pois",
    hover_name="NM_MUN",
    title="Quantidade de POIs de Concorrentes por Município",
    color_continuous_scale="Viridis"
)
fig.update_geos(fitbounds="locations", visible=False)
fig.show()

# 5. Determinar a correlação espacial (coeficiente I de Moran global) da quantidade de POIs no nível de município.
w = weights.Queen.from_dataframe(municipios_gdf)
w.transform = 'r'
y = municipios_gdf['densidade_pois']
moran = esda.Moran(y, w)

print(f"Coeficiente I de Moran: {moran.I}")
print(f"Valor p: {moran.p_sim}")

# Plotar os resultados do Moran
plt.figure(figsize=(10, 4))
plt.scatter(moran.z, weights.spatial_lag.lag_spatial(w, moran.z), edgecolor='k', facecolor='none')
plt.axhline(0, color='r', linestyle='--')
plt.axvline(0, color='r', linestyle='--')
plt.title('Diagrama de Dispersão de Moran')
plt.xlabel('Valores Padronizados')
plt.ylabel('Valores Padronizados de Lag')
plt.show()

# 6. Plotar os dados de faturamento exibindo informações de lat long e o faturamento no "hover" do mouse sobre o ponto.
unidades_faturamento_mediana = unidades_faturamento.groupby(['id_unidade', 'latitude', 'longitude']).median().reset_index()
fig = px.scatter_mapbox(
    unidades_faturamento_mediana,
    lat='latitude',
    lon='longitude',
    hover_name='faturamento',
    zoom=6,
    mapbox_style="carto-positron",
    title="Faturamento Mediano das Unidades"
)
fig.show()

# 7. Determinar unidades com tendência de alta ou de baixa no faturamento.
unidades_faturamento['trend'] = np.nan
unidades = unidades_faturamento['id_unidade'].unique()
for unidade in unidades:
    serie = unidades_faturamento[unidades_faturamento['id_unidade'] == unidade].set_index('data')['faturamento']
    if len(serie) > 1:
        result = seasonal_decompose(serie, model='additive', period=12)
        model = LinearRegression()
        X = np.arange(len(result.trend.dropna())).reshape(-1, 1)
        y = result.trend.dropna().values
        model.fit(X, y)
        unidades_faturamento.loc[unidades_faturamento['id_unidade'] == unidade, 'trend'] = model.coef_[0]

unidades_faturamento['trend_direction'] = ['Alta' if x > 0 else 'Baixa' for x in unidades_faturamento['trend']]

fig = px.scatter_mapbox(
    unidades_faturamento,
    lat='latitude',
    lon='longitude',
    hover_name='faturamento',
    color='trend_direction',
    zoom=6,
    mapbox_style="carto-positron",
    title="Tendência de Faturamento das Unidades"
)
fig.show()

# 8. Plotar o faturamento médio por município.
municipios_gdf = municipios_gdf.merge(faturamento_municipio_agg, on='NM_MUN', how='left')
municipios_gdf['faturamento_medio'] = municipios_gdf['faturamento_medio'].fillna(0)

fig = px.choropleth(
    municipios_gdf,
    geojson=municipios_gdf.geometry,
    locations=municipios_gdf.index,
    color="faturamento_medio",
    hover_name="NM_MUN",
    title="Faturamento Médio por Município",
    color_continuous_scale="Viridis"
)
fig.update_geos(fitbounds="locations", visible=False)
fig.show()

# 9. Plotar o faturamento médio por 100 mil habitantes.
if 'populacao' in municipios_gdf.columns:
    municipios_gdf['faturamento_por_100k'] = (municipios_gdf['faturamento_medio'] / municipios_gdf['populacao']) * 100000
    fig = px.choropleth(
        municipios_gdf,
        geojson=municipios_gdf.geometry,
        locations=municipios_gdf.index,
        color="faturamento_por_100k",
        hover_name="NM_MUN",
        title="Faturamento Médio por 100 mil Habitantes",
        color_continuous_scale="Viridis"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()

# 10. Determinar a correlação espacial (coeficiente I de Moran global) do faturamento total no nível de município.
w_faturamento = weights.Queen.from_dataframe(municipios_gdf)
w_faturamento.transform = 'r'
y_faturamento = municipios_gdf['faturamento_medio']
moran_faturamento = esda.Moran(y_faturamento, w_faturamento)

print(f"Coeficiente I de Moran para faturamento: {moran_faturamento.I}")
print(f"Valor p: {moran_faturamento.p_sim}")
