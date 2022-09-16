## Import Libraries

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Import Python Libraries
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from dash import dcc
from dash import html
from dash import Dash
from dash import dash_table
from datetime import timedelta
from dash.dependencies import Output, Input

## Start Application

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Start Application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

## Import Data from GSheets

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Import data from GSheets
def data_from_GSheets():
    import warnings
    warnings.filterwarnings("ignore")
    """
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                         '1P3Xx65It48JOpmVyDuamWpNzShSjbeZAshhHupFKc8g' +
                         '/export?gid=363258001' + 
                         '&format=csv',
                         # index_col=0,
                         # Set first column as rownames in data frame
                         low_memory=False,
                        )
                        
    """
    df = pd.read_csv('raw_data.csv', sep=';', encoding='latin-1')
    #############################################
    dataSet = df.iloc[: , :45]
    dataSet.columns = dataSet.columns.str.replace(" ", "_")
    #############################################
    p_names = []
    for p in dataSet['Platform']:
        if "Mentoring Platform" in p:
            p_names.append(p.replace("Mentoring Platform", "").strip())
        elif "Mentoring Program" in p:
            p_names.append(p.replace("Mentoring Program", "").strip())
        elif "Executive Mentorship" in p:
            p_names.append(p.replace("Executive Mentorship", "").strip())
        else:
            p_names.append(p.strip())
    
    #############################################
    dataSet['Platform'] = p_names
    dataSet['Date'] = pd.to_datetime(dataSet['Date'])
    dataSet['Corrected_Date'] = (pd.to_datetime(dataSet['Date']) + timedelta(days=-7))
    #############################################
    dataSet.sort_values(['Platform', 'Corrected_Date'], axis = 0, ascending = True, inplace = True)
    dataSet.sort_index()
    #############################################
    dFrames = []
    subsets = {}
    for p in sorted(dataSet['Platform'].unique()):
        #############################################
        subsets[p] = dataSet.loc[dataSet["Platform"] == p].reset_index()
        subsets[p]['Corrected_Date'] = pd.to_datetime(subsets[p]['Corrected_Date'])
        subsets[p]['Date_Order'] = "-"
        #############################################
        subsets[p].sort_values(by=['Corrected_Date'], axis = 0, ascending=True, inplace=True)
        subsets[p].reset_index(inplace=True)
        subsets[p]['index'] = subsets[p].index
        #############################################
        subsets[p]['Change_in_Users'] = subsets[p]['Users'] - subsets[p]['Users'].shift(1)
        subsets[p]['Change_in_Users'].iloc[0] = subsets[p]['Users'].iloc[0]
        #############################################
        subsets[p]['Change_in_Profiled'] = subsets[p]['Total_Profiled'] - subsets[p]['Total_Profiled'].shift(1)
        subsets[p]['Change_in_Profiled'].iloc[0] = subsets[p]['Total_Profiled'].iloc[0]
        #############################################
        subsets[p]['Change_in_Profiled_Mentors'] = subsets[p]['Profiled_Mentors'] - subsets[p]['Profiled_Mentors'].shift(1)
        subsets[p]['Change_in_Profiled_Mentors'][0] = subsets[p]['Profiled_Mentors'][0]
        #############################################
        subsets[p]['Change_in_Profiled_Mentees'] = subsets[p]['Profiled_Mentees'] - subsets[p]['Profiled_Mentees'].shift(1)
        subsets[p]['Change_in_Profiled_Mentees'].iloc[0] = subsets[p]['Profiled_Mentees'].iloc[0]
        #############################################
        subsets[p]['Change_in_Non-Profiled'] = subsets[p]['Total_Non-Profiled'] - subsets[p]['Total_Non-Profiled'].shift(1)
        subsets[p]['Change_in_Non-Profiled'].iloc[0] = subsets[p]['Total_Non-Profiled'].iloc[0]
        #############################################
        subsets[p]['Profile_as_Both'] = (subsets[p]['Profiled_Mentors'] + subsets[p]['Profiled_Mentees']) - subsets[p]['Total_Profiled']
        #############################################
        subsets[p]['Change_in_Profile_as_Both'] = subsets[p]['Profile_as_Both'] - subsets[p]['Profile_as_Both'].shift(1)
        subsets[p]['Change_in_Profile_as_Both'].iloc[0] = subsets[p]['Profile_as_Both'].iloc[0]
        #############################################
        subsets[p]['Change_in_Completed_Meetings'] = subsets[p]['Meetings_(Completed)'] - subsets[p]['Meetings_(Completed)'].shift(1)
        subsets[p]['Change_in_Completed_Meetings'].iloc[0] = subsets[p]['Meetings_(Completed)'].iloc[0]
        #############################################
        subsets[p]['Diff_of_Request_Initiated'] = subsets[p]['Total_no._of_Request_Initiated'] - subsets[p]['Total_no._of_Request_Initiated'].shift(1)
        subsets[p]['Diff_of_Request_Initiated'].iloc[0] = subsets[p]['Total_no._of_Request_Initiated'].iloc[0]
        #############################################
        subsets[p]['Diff_of_Request_Pending'] = subsets[p]['Current_no._of_Requests_Pending'] - subsets[p]['Current_no._of_Requests_Pending'].shift(1)
        subsets[p]['Diff_of_Request_Pending'].iloc[0] = subsets[p]['Current_no._of_Requests_Pending'].iloc[0]
        #############################################
        subsets[p]['Diff_of_Request_Accepted'] = subsets[p]['Current_no._of_Requests_Accepted'] - subsets[p]['Current_no._of_Requests_Accepted'].shift(1)
        subsets[p]['Diff_of_Request_Accepted'].iloc[0] = subsets[p]['Current_no._of_Requests_Accepted'].iloc[0]
        #############################################
        subsets[p]['Diff_of_Request_Completed'] = subsets[p]['Current_no._of_Requests_Completed'] - subsets[p]['Current_no._of_Requests_Completed'].shift(1)
        subsets[p]['Diff_of_Request_Completed'].iloc[0] = subsets[p]['Current_no._of_Requests_Completed'].iloc[0]
        #############################################
        subsets[p]['Diff_of_Request_Expired'] = subsets[p]['Current_no._of_Requests_Expired'] - subsets[p]['Current_no._of_Requests_Expired'].shift(1)
        subsets[p]['Diff_of_Request_Expired'].iloc[0] = subsets[p]['Current_no._of_Requests_Expired'].iloc[0]
        #############################################
        subsets[p]['Diff_of_Request_Removed'] = subsets[p]['Current_no._of_Requests_Removed'] - subsets[p]['Current_no._of_Requests_Removed'].shift(1)
        subsets[p]['Diff_of_Request_Removed'].iloc[0] = subsets[p]['Current_no._of_Requests_Removed'].iloc[0]
        #############################################
        dFrames.append(subsets[p])
        #############################################
    big_df = pd.concat(dFrames, ignore_index=False).drop(columns=['index']).reset_index()
    big_df.fillna(0, inplace=True)
    col = big_df.pop("Corrected_Date")
    big_df.insert(2, col.name, col)
    #############################################
    big_df.drop(columns=['level_0'], inplace=True)
    #############################################
    big_df['Date'] = pd.to_datetime(big_df['Date'])
    big_df['Year'] = pd.DatetimeIndex(big_df['Date']).year
    big_df['Month'] = pd.DatetimeIndex(big_df['Date']).month
    big_df['Day'] = pd.DatetimeIndex(big_df['Date']).day
    big_df['Week'] = pd.DatetimeIndex(big_df['Date']).week
    big_df['Month_Year'] = big_df['Date'].dt.strftime('%Y, %m')
    #############################################
    big_df['Corrected_Date'] = pd.to_datetime(big_df['Corrected_Date'])
    big_df['Corrected_Year'] = pd.DatetimeIndex(big_df['Corrected_Date']).year
    big_df['Corrected_Month'] = pd.DatetimeIndex(big_df['Corrected_Date']).month
    big_df['Corrected_Day'] = pd.DatetimeIndex(big_df['Corrected_Date']).day
    big_df['Corrected_Week'] = pd.DatetimeIndex(big_df['Corrected_Date']).week
    big_df['Corrected_Month_Year'] = big_df['Corrected_Date'].dt.strftime('%Y, %m')
    #############################################
    big_df.drop_duplicates(inplace=True)
    #############################################
    return big_df
#################################################################################################################################################
complete_df = data_from_GSheets()

## Totals in Profiles Figure

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- TAB 1 FIGURES
######################### ----------- TAB 1 FIGURES
######################### ----------- TAB 1 FIGURES
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Totals in # Profiles Figure
def total_in_profiles_fig(df, group_by, selected_platforms):
    #############################################
    df.sort_values(['Corrected_Date'], ascending = True, inplace = True)
    y_params = ['cum_profiled','cum_mentees', 'cum_mentors']
    #############################################
    if len(selected_platforms) == 1:
        company = selected_platforms[0]
    elif len(selected_platforms) > 1:
        company = 'Various'
    #############################################
    if group_by == 'Corrected_Date':
        x_title = 'Week-Date'
    elif group_by == 'Corrected_Month_Year':
        x_title = 'Year, Month'
    #############################################
    sub_df = df.loc[df["Platform"].isin(selected_platforms)].reset_index()
    sub_df.drop(columns='Platform', inplace=True)
    sub_df.sort_values(['Corrected_Date'], ascending = True, inplace = True)
    sub_df.drop(columns=['index'], inplace=True)
    df_1 = sub_df.groupby(group_by).sum().reset_index()
    df_1['cum_profiled'] = df_1['Change_in_Profiled'].cumsum()
    df_1['cum_mentors'] = df_1['Change_in_Profiled_Mentors'].cumsum()
    df_1['cum_mentees'] = df_1['Change_in_Profiled_Mentees'].cumsum()
    #############################################
    fig = px.line(df_1,
                  x=group_by,
                  y=y_params,
                  height=500,
                  markers=True,
                  color_discrete_sequence=["#58A65C", "#69BBC4", "#EB752F", "magenta"]
                 )
    #############################################
    fig.update_traces(
        hovertemplate='%{y}',
        textposition="bottom right",
    )
    #############################################
    fig.update_layout(
        transition_duration=500,
        title=f'Total in # Profiles ({company})',
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title="Legend Title",
        font=dict(
            family="Arial",
            color="Black"
        ),
        #paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#fff',
        legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
    )
    #############################################
    new_labels={
                "cum_profiled": "Total Profiled", 
                "cum_mentors": "Profiled Mentors",
                "cum_mentees":"Profiled Mentees"
                }
    fig.for_each_trace(
        lambda t: t.update(name = new_labels[t.name],
                           legendgroup = new_labels[t.name],
                           hovertemplate = t.hovertemplate.replace(t.name, new_labels[t.name]))
    )
    
    #############################################
    fig.update_layout(
        hovermode="x unified",
        legend_title = None,
        transition_duration=500,
    )
    return fig

## New Profiles Figure

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- New Profiles Figure
def new_profiles_fig(df, group_by, selected_platforms):
    #############################################
    df.sort_values(['Corrected_Date'], ascending = True, inplace = True)
    y_params = ['Change_in_Profiled','Change_in_Profiled_Mentees', 'Change_in_Profiled_Mentors']
    #############################################
    if len(selected_platforms) == 1:
        company = selected_platforms[0]
    elif len(selected_platforms) > 1:
        company = 'Various' 
    #############################################
    if group_by == 'Corrected_Date':
        x_title = 'Week-Date'
    elif group_by == 'Corrected_Month_Year':
        x_title = 'Year, Month'
        
    #############################################
    dFrames = []
    subsets = {}
    for p in sorted(df['Platform'].unique()):
        #############################################
        subsets[p] = (df.loc[df["Platform"] == p]).iloc[1: , :].reset_index()        
        subsets[p]['Corrected_Date'] = pd.to_datetime(subsets[p]['Corrected_Date'])
        #############################################
        dFrames.append(subsets[p])
        #############################################
    sub_df = pd.concat(dFrames, ignore_index=False).drop(columns=['index']).reset_index()#df.loc[df["Platform"].isin(selected_platforms)].reset_index()
    sub_df.drop(columns='Platform', inplace=True)
    sub_df.sort_values(['Corrected_Date'], ascending = True, inplace = True)
    sub_df.drop(columns=['index'], inplace=True)
    df_1 = sub_df.groupby(group_by).sum().reset_index()
    #############################################
    fig = px.bar(df_1,
                 barmode='group',
                 x=group_by,
                 y=y_params,
                 height=600,
                 text_auto='.2s',
                 color_discrete_sequence=["#58A65C", "#69BBC4", "#EB752F", "magenta"],
                 )
    #############################################
    fig.update_traces(
        hovertemplate='%{y}',
        width=0.25,
        textangle=-70,
        textposition="outside",
        cliponaxis=False,
        #textposition="bottom right",
    )
    #############################################
    fig.update_layout(
        transition_duration=500,
        title=f'# of New Profiles - ({company})',
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title="Legend Title",
        font=dict(
            family="Arial",
            color="Black"
        ),
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='#fff',
        legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
    )
    #############################################
    new_labels={
                "Change_in_Profiled" : "Total Profiled", 
                "Change_in_Profiled_Mentees" : "Profiled Mentors",
                "Change_in_Profiled_Mentors" : "Profiled Mentees"
                }
    fig.for_each_trace(
        lambda t: t.update(name = new_labels[t.name],
                           legendgroup = new_labels[t.name],
                           hovertemplate = t.hovertemplate.replace(t.name, new_labels[t.name]))
    )
    #############################################
    fig.update_layout(
        #hovermode="x unified",
        transition_duration=500,
        legend_title = None,
    )
    return fig
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

## Modify Dataframe for USER GROWTH

#################################################################################################################################################
charts_df = complete_df[['Date','Corrected_Date', 'Corrected_Month_Year', 'Platform', 'Change_in_Users', 'Change_in_Profiled', 'Change_in_Profiled_Mentors', 'Change_in_Profiled_Mentees']]
#################################################################################################################################################
selected_platforms = sorted(complete_df['Platform'].unique()) #['YCP', 'Wipro', 'Wine Industry', 'Expert Impact', 'Amazon']
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
group_by = 'Corrected_Date'
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
group_by = 'Corrected_Month_Year'

## Dash Body Build

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Tabs Design

#################################################################
#################################################################
######################### -----------  Tab 1
fig_1 = total_in_profiles_fig(charts_df, group_by, selected_platforms)
fig_2 = new_profiles_fig(charts_df, group_by, selected_platforms)
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            # html.Button('Update', id='btn-update'),
            html.Div(dcc.Graph(figure=fig_1)),
            html.Div(dcc.Graph(figure=fig_2)),
        ]
    ),
    className="mt-3",
)
#################################################################
#################################################################
######################### -----------  Tab 2
tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
        ]
    ),
    className="mt-3",
)
#################################################################
#################################################################
######################### -----------  Tab 3
tab3_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 3!", className="card-text"),
        ]
    ),
    className="mt-3",
)
#################################################################
#################################################################
######################### -----------  Tab 4
tab4_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 4!", className="card-text"),
        ]
    ),
    className="mt-3",
)
#################################################################
#################################################################
######################### -----------  Tab 5
tab5_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 5!", className="card-text"),
        ]
    ),
    className="mt-3",
)
#################################################################
#################################################################
######################### -----------  Tab 6
tab6_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 6!", className="card-text"),
        ]
    ),
    className="mt-3",
)
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Tabs Disposition
tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="User Growth", tab_id="tab-1"),
                dbc.Tab(label="Mentoring Requests", tab_id="tab-2"),
                dbc.Tab(label="Meetings", tab_id="tab-3"),
                dbc.Tab(label="Relationships Monthly", tab_id="tab-4"),
                dbc.Tab(label="Weekly Active Relationships", tab_id="tab-5"),
                dbc.Tab(label="Mentors, Mentees Relationship", tab_id="tab-6"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ]
)

## Callbacks

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Callbacks

#################################################################
######################### -----------  Update Table with Button
@app.callback(
    Output('table_1', 'children'),
    Input('btn-update', 'n_clicks')
)
def displayClick(btn1):
    table_1 = generate_table(data_from_GSheets(), max_rows=25)
    return table_1

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Tabs Callback
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-1":
        return tab1_content
    elif at == "tab-2":
        return tab2_content
    elif at == "tab-3":
        return tab3_content
    elif at == "tab-4":
        return tab4_content
    elif at == "tab-5":
        return tab5_content
    elif at == "tab-6":
        return tab6_content
    return #html.P("This shouldn't ever be displayed...")

## Run App

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### -----------  App Layout
app.layout = html.Div([
    tabs
])
#################################################################
#################################################################
######################### -----------  Server Initiation
if __name__ == '__main__':
    app.run_server(debug=False)