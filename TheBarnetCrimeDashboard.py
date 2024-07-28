#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import ast
import geopandas as gpd
import folium
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from folium.plugins import HeatMap, MarkerCluster
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import requests
from datetime import datetime
import os
from flask import send_from_directory

# Load the GeoJSON files from URLs
try:
    lsoa_geojson_data = requests.get('https://raw.githubusercontent.com/samab74/Trying-New-Data/main/lsoa_with_crime_and_hmo_counts.geojson').json()
    wards_geojson_data = requests.get('https://raw.githubusercontent.com/samab74/Barnet-Dashboard/main/OSBoundaryLine%20-%20BarnetWards.geojson').json()
except Exception as e:
    print(f"Error loading GeoJSON files: {e}")

# Convert GeoJSON to GeoDataFrames
try:
    lsoa_gdf = gpd.GeoDataFrame.from_features(lsoa_geojson_data["features"])
    wards_gdf = gpd.GeoDataFrame.from_features(wards_geojson_data["features"])

    # Ensure both GeoDataFrames use the same CRS (Coordinate Reference System)
    lsoa_gdf = lsoa_gdf.set_crs("EPSG:4326", allow_override=True)
    wards_gdf = wards_gdf.set_crs("EPSG:4326", allow_override=True)

    # Re-project to a projected CRS for accurate area calculation
    projected_crs = "EPSG:3395"  # World Mercator projection
    lsoa_gdf = lsoa_gdf.to_crs(projected_crs)
    wards_gdf = wards_gdf.to_crs(projected_crs)

    # Perform spatial overlay to get intersection areas
    intersection_gdf = gpd.overlay(lsoa_gdf, wards_gdf, how='intersection')

    # Calculate the area of each intersection
    intersection_gdf['area'] = intersection_gdf.geometry.area

    # Find the ward with the largest intersection area for each LSOA
    majority_ward_gdf = intersection_gdf.loc[intersection_gdf.groupby('LSOA21CD')['area'].idxmax()]

    # Merge this information back to the original LSOA GeoDataFrame (re-project back to original CRS)
    lsoa_gdf = lsoa_gdf.to_crs("EPSG:4326")
    majority_ward_gdf = majority_ward_gdf.to_crs("EPSG:4326")
    lsoa_gdf = lsoa_gdf.merge(majority_ward_gdf[['LSOA21CD', 'WardName']], on='LSOA21CD', how='left')

    # Convert the result back to a DataFrame
    df = pd.DataFrame(lsoa_gdf.drop(columns='geometry'))

    # Verify the assignment
    if df.isnull().values.any():
        print("Warning: Some LSOAs were not assigned to any ward.")
    else:
        print("All LSOAs successfully assigned to wards.")
except Exception as e:
    print(f"Error processing GeoDataFrames: {e}")

# Load crime dataset from URL
try:
    crime_data = pd.read_csv('https://raw.githubusercontent.com/samab74/Barnet-Dashboard/main/barnet_crimes.csv')
except Exception as e:
    print(f"Error loading crime data: {e}")

# Extract latitude and longitude
def extract_lat_lon(location_str):
    try:
        location_dict = ast.literal_eval(str(location_str).replace('null', 'None'))
        latitude = float(location_dict.get('latitude', None))
        longitude = float(location_dict.get('longitude', None))
        return pd.Series([latitude, longitude])
    except (ValueError, SyntaxError):
        return pd.Series([None, None])

try:
    crime_data[['latitude', 'longitude']] = crime_data['location'].apply(extract_lat_lon)
    filtered_crime_data = crime_data.dropna(subset=['latitude', 'longitude'])
    crime_categories = filtered_crime_data['category'].unique().tolist()
except Exception as e:
    print(f"Error processing crime data: {e}")

# Create the dictionary for renaming columns
short_names = {
    'FeatureID': 'Feat ID',
    'LSOA21CD': 'LSOA Code',
    'LSOA21NM': 'LSOA Name',
    'Lower layer Super Output Areas Code': 'LSOA Code',
    'Lower layer Super Output Areas_deprivation': 'LSOA Deprivation',
    'Does not apply_deprivation': 'No Deprivation',
    'Household is deprived in four dimensions': 'Deprived 4D',
    'Household is deprived in one dimension': 'Deprived 1D',
    'Household is deprived in three dimensions': 'Deprived 3D',
    'Household is deprived in two dimensions': 'Deprived 2D',
    'Household is not deprived in any dimension': 'Not Deprived',
    'Does not apply_economic': 'No Economic',
    'Economically active (excluding full-time students): In employment: Employee: Full-time': 'Econ Active FT',
    'Economically active (excluding full-time students): In employment: Employee: Part-time': 'Econ Active PT',
    'Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time': 'Econ Self-Empl FT',
    'Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time': 'Econ Self-Empl PT',
    'Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time': 'Econ Self-NoEmp FT',
    'Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time': 'Econ Self-NoEmp PT',
    'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks': 'Econ Unemp Seek',
    'Economically active and a full-time student: In employment: Employee: Full-time': 'Econ Active FT Student Emp FT',
    'Economically active and a full-time student: In employment: Employee: Part-time': 'Econ Active FT Student Emp PT',
    'Economically active and a full-time student: In employment: Self-employed with employees: Full-time': 'Econ Active FT Student SelfEmp FT',
    'Economically active and a full-time student: In employment: Self-employed with employees: Part-time': 'Econ Active FT Student SelfEmp PT',
    'Economically active and a full-time student: In employment: Self-employed without employees: Full-time': 'Econ Active FT Student NoEmp FT',
    'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks': 'Econ Active FT Student Unemp',
    'Economically inactive: Long-term sick or disabled': 'Econ Inactive Sick',
    'Economically inactive: Looking after home or family': 'Econ Inactive Home',
    'Economically inactive: Other': 'Econ Inactive Other',
    'Economically inactive: Retired': 'Econ Inactive Retired',
    'Economically inactive: Student': 'Econ Inactive Student',
    'Apprenticeship': 'Apprenticeship',
    'Does not apply': 'No Apply',
    'Level 1 and entry level qualifications: 1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills': 'Qual Level 1',
    'Level 2 qualifications: 5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma': 'Qual Level 2',
    'Level 3 qualifications: 2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma': 'Qual Level 3',
    'Level 4 qualifications or above: degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)': 'Qual Level 4+',
    'No qualifications': 'No Qual',
    'Other: vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales (equivalent not stated or unknown)': 'Other Qual',
    'Asian, Asian British or Asian Welsh: Bangladeshi': 'Asian Bangladeshi',
    'Asian, Asian British or Asian Welsh: Chinese': 'Asian Chinese',
    'Asian, Asian British or Asian Welsh: Indian': 'Asian Indian',
    'Asian, Asian British or Asian Welsh: Other Asian': 'Asian Other',
    'Asian, Asian British or Asian Welsh: Pakistani': 'Asian Pakistani',
    'Black, Black British, Black Welsh, Caribbean or African: African': 'Black African',
    'Black, Black British, Black Welsh, Caribbean or African: Caribbean': 'Black Caribbean',
    'Black, Black British, Black Welsh, Caribbean or African: Other Black': 'Black Other',
    'Does not apply_ethnicity': 'No Ethnicity',
    'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups': 'Mixed Other',
    'Mixed or Multiple ethnic groups: White and Asian': 'Mixed White Asian',
    'Mixed or Multiple ethnic groups: White and Black African': 'Mixed White Black African',
    'Mixed or Multiple ethnic groups: White and Black Caribbean': 'Mixed White Black Caribbean',
    'Other ethnic group: Any other ethnic group': 'Other Ethnic',
    'Other ethnic group: Arab': 'Ethnic Arab',
    'White: English, Welsh, Scottish, Northern Irish or British': 'White British',
    'White: Gypsy or Irish Traveller': 'White Gypsy/Traveller',
    'White: Irish': 'White Irish',
    'White: Other White': 'White Other',
    'White: Roma': 'White Roma',
    'Female': 'Female',
    'Male': 'Male',
    'Population': 'Population',
    'total_crime': 'Total Crime',
    'hmo_count': 'HMO Count'  # Add this line to include HMO count
}


# List of variables to exclude
variables_to_exclude = [
    'FeatureID', 'LSOA21CD', 'LSOA21NM', 
    'Lower layer Super Output Areas Code', 'Lower layer Super Output Areas_deprivation',
    'Does not apply_deprivation', 'index_right', 'ONSWardCode', 'WardName'
]

# Initialize the Dash app
app = Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# Define the layout
app.layout = html.Div([
    html.H1("LSOA Dashboard", style={'textAlign': 'center', 'padding': '10px'}),
    dcc.Tabs([
        dcc.Tab(label='Lower Layer Super Output Area (LSOA)', children=[  # Change the tab label here
            html.Div([
                html.H2("Lower Layer Super Output Area (LSOA) Variable Heatmap", style={'textAlign': 'center', 'padding': '10px'}),  # Change the heading here
                html.Label("Select Variable for Map:"),
                dcc.Dropdown(
                    id='map-variable-dropdown',
                    options=[
                        {'label': short_names.get(var, var), 'value': var} 
                        for var in df.columns if var not in variables_to_exclude
                    ],
                    value='total_crime',  # Set default value
                    clearable=False
                ),
                html.Iframe(id='map', width='100%', height='600', style={'border': 'none'}),
                html.Br(),
                html.Label("Enter LSOA Name:"),
                dcc.Input(id='lsoa-input', type='text', placeholder='Enter LSOA name', style={'width': '50%'}),
                html.Button(id='submit-button', n_clicks=0, children='Submit', style={'margin-left': '10px'}),
                html.Div(id='lsoa-info', style={'margin-top': '20px'}),
                html.Div("Use the dropdown to select a variable for the map. Enter an LSOA name to get detailed information.", style={'margin-top': '20px', 'color': 'grey'}),
                html.Div("Information is current as of April 2024.", style={'position': 'fixed', 'bottom': '10px', 'right': '10px', 'color': 'grey'})
            ], style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'margin-bottom': '20px'}),
        ]),
        dcc.Tab(label='Bar Charts', children=[
            html.Div([
                html.H2("Bar Charts", style={'textAlign': 'center', 'padding': '10px'}),
                html.Label("Select Variable for Bar Charts:"),
                dcc.Dropdown(
                    id='bar-variable-dropdown',
                    options=[
                        {'label': short_names.get(var, var), 'value': var} 
                        for var in df.columns if var not in variables_to_exclude
                    ],
                    value='total_crime',  # Set default value
                    clearable=False
                ),
                dcc.Graph(id='bar-chart'),
                dcc.Graph(id='ward-bar-chart'),
                html.Div("Select a variable to view bar charts for each LSOA and ward.", style={'margin-top': '20px', 'color': 'grey'}),
                html.Div("Information is current as of April 2024.", style={'position': 'fixed', 'bottom': '10px', 'right': '10px', 'color': 'grey'})
            ], style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'margin-bottom': '20px'}),
        ]),
        dcc.Tab(label='Barnet Crime Heatmap', children=[
            html.Div([
                html.H2("Barnet Crime Heatmap", style={'textAlign': 'center', 'padding': '10px'}),
                html.Label("Select Date:"),
                dcc.Input(id='date-input', type='text', placeholder='Enter date (YYYY-MM)', style={'width': '50%'}),
                html.Button(id='submit-date-button', n_clicks=0, children='Submit', style={'margin-left': '10px'}),
                html.Label("Select Crime Category:"),
                dcc.Dropdown(
                    id='crime-category-dropdown',
                    options=[{'label': category, 'value': category} for category in crime_categories],
                    value=crime_categories[0],  # Set default value to the first category
                    clearable=False
                ),
                html.Br(),
                html.A("Download Crime Heatmap", id="download-link", href="", target="_blank", style={'margin-top': '20px'}),
                html.Div("Enter a date in YYYY-MM format and select a crime category to update the heatmap.", style={'margin-top': '20px', 'color': 'grey'}),
                html.Div("You can select dates from 2021-06 to 2024-05.", style={'margin-top': '20px', 'color': 'grey'}),
                html.Div("Please wait for the link to change. This process may take up to a minute.", style={'margin-top': '20px', 'color': 'red'}),
                html.Div("Information is current as of April 2024.", style={'position': 'fixed', 'bottom': '10px', 'right': '10px', 'color': 'grey'})
            ], style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'margin-bottom': '20px'}),
        ]),
        dcc.Tab(label='Correlations', children=[
            html.Div([
                html.H2("Correlations with Total Crime", style={'textAlign': 'center', 'padding': '10px'}),
                dcc.Dropdown(
                    id='correlation-variable-dropdown',
                    options=[
                        {'label': short_names.get(var, var), 'value': var} 
                        for var in df.columns if var not in variables_to_exclude
                    ],
                    value='total_crime',  # Set default value
                    clearable=False
                ),
                html.Br(),
                html.Label("Select Ward:"),
                dcc.Dropdown(
                    id='correlation-lsoa-dropdown',
                    options=[
                        {'label': 'All Barnet', 'value': 'All Barnet'}] + 
                        [{'label': ward, 'value': ward} for ward in df['WardName'].unique()
                    ],
                    value='All Barnet',  # Default value
                    clearable=False
                ),
                dcc.Graph(id='correlation-scatter-plot'),
                html.Div("Select a variable and ward to view correlation scatter plots with total crime.", style={'margin-top': '20px', 'color': 'grey'}),
                html.Div("Information is current as of April 2024.", style={'position': 'fixed', 'bottom': '10px', 'right': '10px', 'color': 'grey'})
            ], style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'margin-bottom': '20px'}),
        ]),
        dcc.Tab(label='Key for Variable Names', children=[
            html.Div([
                html.H2("Key for Variable Names", style={'textAlign': 'center', 'padding': '10px'}),
                html.Table([
                    html.Thead(
                        html.Tr([html.Th("Short Name"), html.Th("Full Name")])
                    ),
                    html.Tbody([
                        html.Tr([html.Td(short), html.Td(full)]) for full, short in short_names.items()
                    ])
                ], style={'width': '100%', 'border': '1px solid #ccc', 'border-collapse': 'collapse', 'margin-bottom': '20px'}),
                html.Div("Information is current as of April 2024.", style={'position': 'fixed', 'bottom': '10px', 'right': '10px', 'color': 'grey'})
            ], style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'margin-bottom': '20px'}),
        ]),
    ])
])

def fetch_crime_data(date):
    coordinates = "51.55519092818953,-0.30557383443798025:51.670170250593905,-0.30557383443798025:51.670170250593905,-0.12909406402138046:51.55519092818953,-0.12909406402138046:51.55519092818953,-0.30557383443798025"
    url = f"https://data.police.uk/api/crimes-street/all-crime?poly={coordinates}&date={date}"
    response = requests.get(url)
    if response.status_code == 200:
        crimes = response.json()
        if crimes:
            df = pd.DataFrame(crimes)
            df[['latitude', 'longitude']] = df['location'].apply(extract_lat_lon)
            return df.dropna(subset=['latitude', 'longitude'])
    return pd.DataFrame()

import os
from flask import send_from_directory

# Ensure 'static' directory exists
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Create a Flask server instance
server = app.server

# Define a route to serve files from the 'static' directory
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.callback(
    Output('download-link', 'href'),
    Output('download-link', 'children'),
    [Input('submit-date-button', 'n_clicks')],
    [State('date-input', 'value'), State('crime-category-dropdown', 'value')]
)
def update_crime_map(n_clicks, date_input, selected_category):
    if n_clicks > 0 and date_input:
        crime_data = fetch_crime_data(date_input)
        filtered_data_by_category = crime_data[crime_data['category'] == selected_category]

        if not filtered_data_by_category.empty:
            center_lat, center_lon = filtered_data_by_category['latitude'].mean(), filtered_data_by_category['longitude'].mean()
            map_barnet = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
            heat_data = [[row['latitude'], row['longitude']] for idx, row in filtered_data_by_category.iterrows()]
            heatmap_layer = HeatMap(heat_data, name='Crime Heatmap').add_to(map_barnet)
            marker_cluster = MarkerCluster(name='Crime Markers').add_to(map_barnet)
    
            def get_marker_color(crime_category):
                category_colors = {
                    'anti-social-behaviour': 'blue',
                    'burglary': 'purple',
                    'criminal-damage-arson': 'orange',
                    'drugs': 'darkred',
                    'other-theft': 'green',
                    'possession-of-weapons': 'cadetblue',
                    'public-order': 'lightred',
                    'robbery': 'darkpurple',
                    'shoplifting': 'lightblue',
                    'theft-from-the-person': 'darkgreen',
                    'vehicle-crime': 'black',
                    'violent-crime': 'red',
                    'other-crime': 'gray'
                }
                return category_colors.get(crime_category, 'black')
    
            for idx, row in filtered_data_by_category.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f'Category: {row["category"]}',
                    icon=folium.Icon(color=get_marker_color(row['category']))
                ).add_to(marker_cluster)
    
            crime_map_file_path = os.path.join(STATIC_DIR, f'crime_heatmap_{date_input}_{selected_category}.html')
            map_barnet.save(crime_map_file_path)
    
            download_link = f'/static/crime_heatmap_{date_input}_{selected_category}.html'
            download_text = f'Download Crime Heatmap for {date_input} ({selected_category})'
    
            return download_link, download_text
    return "", "Download Crime Heatmap"

@app.callback(
    Output('bar-chart', 'figure'),
    Output('ward-bar-chart', 'figure'),
    [Input('bar-variable-dropdown', 'value')]
)
def update_bar_charts(selected_variable):
    # Ensure the selected variable exists in the dataframe
    if selected_variable not in df.columns:
        raise ValueError(f"Selected variable {selected_variable} does not exist in the DataFrame")

    # Sort the dataframe by the selected variable
    sorted_df = df.sort_values(by=selected_variable, ascending=False)
    
    # Update Lower Super Output Area (LSOA) bar chart
    lsoa_fig = px.bar(
        sorted_df,
        x='LSOA21NM',
        y=selected_variable,
        title=f'{short_names.get(selected_variable, selected_variable)} by Lower Super Output Area (LSOA)',
        labels={'LSOA21NM': 'Lower Super Output Area (LSOA) Name', selected_variable: short_names.get(selected_variable, selected_variable)},
        custom_data=['WardName']  # Include ward name in the custom data
    )
    lsoa_fig.update_layout(
        xaxis_title='Lower Super Output Area (LSOA) Name',
        yaxis_title=short_names.get(selected_variable, selected_variable),
        xaxis={'categoryorder': 'total descending'},
        margin={'l': 40, 'r': 40, 't': 40, 'b': 80},
        height=600,
        annotations=[dict(
            x=0.5, y=-0.25,
            showarrow=False,
            text="Data is current as of April 2024",
            xref="paper", yref="paper",
            xanchor="center", yanchor="top",
            font=dict(size=12)
        )]
    )
    
    # Update hover template to include ward information
    lsoa_fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Value: %{y}<br>Ward: %{customdata[0]}"
    )
    
    # Update Ward bar chart
    ward_totals = df.groupby('WardName')[selected_variable].sum().reset_index()
    ward_fig = px.bar(
        ward_totals,
        x='WardName',
        y=selected_variable,
        title=f'Total {short_names.get(selected_variable, selected_variable)} by Ward',
        labels={'WardName': 'Ward Name', selected_variable: short_names.get(selected_variable, selected_variable)}
    )
    ward_fig.update_layout(
        xaxis_title='Ward Name',
        yaxis_title=short_names.get(selected_variable, selected_variable),
        xaxis={'categoryorder': 'total descending'},
        margin={'l': 40, 'r': 40, 't': 40, 'b': 80},
        height=600,
        annotations=[dict(
            x=0.5, y=-0.25,
            showarrow=False,
            text="Data is current as of April 2024",
            xref="paper", yref="paper",
            xanchor="center", yanchor="top",
            font=dict(size=12)
        )]
    )
    
    return lsoa_fig, ward_fig

@app.callback(
    Output('map', 'srcDoc'),
    [Input('map-variable-dropdown', 'value')]
)
def update_variable_map(selected_variable):
    m = folium.Map(location=[51.6, -0.2], zoom_start=12)
    
    if selected_variable:
        values = [feature['properties'][selected_variable] for feature in lsoa_geojson_data['features']]
        percentiles = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]  # Reversed percentiles
        colors = ['#800026', '#BD0026', '#E31A1C', '#FC4E2A', '#FD8D3C', '#FEB24C', '#FED976', '#FFEDA0', '#FFFFCC', '#FFFFFF']
        thresholds = [np.percentile(values, p) for p in percentiles]
        
        def get_color(value):
            for i, threshold in enumerate(thresholds):
                if value >= threshold:  # Reversed comparison
                    return colors[i]
            return colors[-1]
        
        for feature in lsoa_geojson_data['features']:
            lsoa_code = feature['properties']['LSOA21CD']
            matching_row = df[df['LSOA21CD'] == lsoa_code]
            if not matching_row.empty:
                feature['properties']['WardName'] = matching_row.iloc[0]['WardName']
        
        folium.GeoJson(
            lsoa_geojson_data,
            name='Lower Super Output Area (LSOA) Variable',
            style_function=lambda x: {
                'fillColor': get_color(x['properties'][selected_variable]),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['LSOA21NM', selected_variable, 'WardName'],
                aliases=['Lower Super Output Area (LSOA):', short_names.get(selected_variable, selected_variable), 'Ward:']
            )
        ).add_to(m)
        
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: auto; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                    ">
        <h4 style="margin-top: 5px; text-align: center;">Legend</h4>
        <ul style="list-style-type:none; padding-left: 0;">
        '''
        
        for i, percentile in enumerate(percentiles):
            color = colors[i]
            legend_html += f'<li><span style="background:{color}; width: 20px; height: 20px; display: inline-block;"></span> Top {100 - percentile}%</li>'
        
        legend_html += '</ul></div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    ward_style_function = lambda x: {
        'fillColor': 'none',
        'color': 'black',
        'weight': 3,
        'fillOpacity': 0.5,
    }
    
    folium.GeoJson(
        wards_geojson_data,
        name='Electoral Wards',
        style_function=ward_style_function,
        tooltip=folium.GeoJsonTooltip(fields=['WardName'], aliases=['Ward:'])
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    map_file_path = 'lsoa_variable_map.html'
    m.save(map_file_path)
    
    with open(map_file_path, 'r') as f:
        html_map = f.read()
    
    return html_map

@app.callback(
    Output('lsoa-info', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('lsoa-input', 'value'), State('map-variable-dropdown', 'value')]
)
def display_lsoa_info(n_clicks, lsoa_name, selected_variable):
    if n_clicks > 0 and lsoa_name:
        # Remove any leading/trailing spaces and convert to upper case for consistency
        lsoa_name = lsoa_name.strip().upper()
        
        # Filter the DataFrame for the given LSOA name
        lsoa_data = df[df['LSOA21NM'].str.contains(lsoa_name, case=False, na=False)]
        
        if not lsoa_data.empty:
            value = lsoa_data[selected_variable].values[0]

            # Remove duplicates
            df_unique = df.drop_duplicates(subset=['LSOA21NM'])

            # Compute the rank
            df_sorted = df_unique.sort_values(by=selected_variable, ascending=False).reset_index(drop=True)
            df_sorted['rank'] = df_sorted[selected_variable].rank(method='min', ascending=False)
            rank = df_sorted[df_sorted['LSOA21NM'].str.contains(lsoa_name, case=False, na=False)]['rank'].values[0]

            total_lsoas = df_unique['LSOA21NM'].nunique()  # Correctly count the number of unique LSOAs
            ward_name = lsoa_data['WardName'].values[0]  # Get ward name

            return [
                html.H4(f"Information for {lsoa_name}:"),
                html.P(f"Value of {short_names.get(selected_variable, selected_variable)}: {value}"),
                html.P(f"Rank: {int(rank)} out of {total_lsoas} LSOAs"),
                html.P(f"Ward: {ward_name}")  # Display ward name
            ]
        else:
            return html.P(f"No information found for LSOA: {lsoa_name}")
    return ""

@app.callback(
    Output('correlation-scatter-plot', 'figure'),
    [Input('correlation-variable-dropdown', 'value'), Input('correlation-lsoa-dropdown', 'value')]
)
def update_correlation_scatter_plot(selected_variable, selected_ward):
    # Filter numeric data
    numeric_df = df.select_dtypes(include=[np.number])
    
    # If a specific ward is selected, filter the data for that ward
    if selected_ward and selected_ward != "All Barnet":
        ward_data = df[df['WardName'] == selected_ward]
        if not ward_data.empty:
            numeric_df = ward_data.select_dtypes(include=[np.number])

    # Compute correlations
    correlation = numeric_df.corr()[selected_variable].sort_values(ascending=False).reset_index()
    correlation = correlation.rename(columns={'index': 'Variable', selected_variable: 'Correlation'})
    top_correlation = correlation.head(10).copy()

    # Determine the layout of the subplots dynamically
    num_plots = len(top_correlation)
    rows = (num_plots // 5) + (1 if num_plots % 5 else 0)
    cols = min(num_plots, 5)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[
            f"{short_names.get(var, var)}" 
            for var in top_correlation['Variable']
        ],
        horizontal_spacing=0.1,  # Adjust horizontal spacing
        vertical_spacing=0.3  # Adjust vertical spacing
    )
    
    for i, variable in enumerate(top_correlation['Variable']):
        row = i // 5 + 1
        col = i % 5 + 1

        scatter = go.Scatter(
            x=numeric_df[variable],
            y=numeric_df[selected_variable],
            mode='markers',
            name=short_names.get(variable, variable)
        )

        # Calculate the line of best fit
        fit = np.polyfit(numeric_df[variable], numeric_df[selected_variable], deg=1)
        fit_fn = np.poly1d(fit)
        line = go.Scatter(
            x=numeric_df[variable],
            y=fit_fn(numeric_df[variable]),
            mode='lines',
            name=f"Fit: {short_names.get(variable, variable)}",
            line=dict(dash='dash')
        )

        fig.add_trace(scatter, row=row, col=col)
        fig.add_trace(line, row=row, col=col)

        # Add correlation coefficient below the graph
        fig.add_annotation(
            xref=f'x{i+1}', yref=f'y{i+1}',
            x=0.5, y=-0.35,
            text=f"r = {top_correlation['Correlation'][i]:.2f}",
            showarrow=False,
            xanchor='center',
            yanchor='top',
            font=dict(size=12),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f'Top 10 Correlations with {short_names.get(selected_variable, selected_variable)} in {selected_ward}',
        height=400 * rows,
        showlegend=False,
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
        font=dict(size=10),  # Set the font size for the subplot titles
        annotations=[dict(font=dict(size=8)) for _ in range(len(top_correlation['Variable']))]  # Smaller font for subplot titles
    )

    for i, variable in enumerate(top_correlation['Variable']):
        fig['layout'][f'xaxis{i+1}'].update(title=short_names.get(variable, variable), title_font_size=8)
        fig['layout'][f'yaxis{i+1}'].update(title=short_names.get(selected_variable, selected_variable), title_font_size=8)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)



# In[ ]:




