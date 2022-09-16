# MENTORCLOUD DASH APP


## Import Python Libraries


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Import Python Libraries
import dash
import calendar
import numpy as np
import pandas as pd
import dash_daq as daq
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from dash_iconify import DashIconify
from plotly.subplots import make_subplots
from dash.dependencies import Output, Input
from dash import Dash, dash_table, dcc, html
from datetime import datetime, timedelta, date

## Import DataFrames


def categories_from_GSheets():
    import warnings
    warnings.filterwarnings("ignore")
    #############################################
    categories_df = pd.read_csv("Categories.csv", delimiter=';', low_memory=False)
    return categories_df

def dataFrame_from_GSheets():
    import warnings
    warnings.filterwarnings("ignore")
    #############################################
    df = pd.read_csv("Example_01_data.csv", delimiter=';', low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Corrected_Date"] = pd.to_datetime(df["Corrected_Date"])
    return df

## Functions

#################################################################################################################################################
#################################################################################################################################################
def filter_selected_platforms(selected_platforms, range_dates):
    platforms_df = complete_df.loc[complete_df['Platform'].isin(selected_platforms)]
    platforms_df['Corrected_Date'] = pd.to_datetime(platforms_df['Corrected_Date'])
    mask = (platforms_df['Corrected_Date'] >= pd.to_datetime(range_dates[0])) & (platforms_df['Corrected_Date'] <= pd.to_datetime(range_dates[1]))
    filtered_df = platforms_df.loc[mask]
    return filtered_df

#################################################################################################################################################
#################################################################################################################################################
def platforms_list(selected_categories):
    if len(selected_categories) < 1:
        return
    categories_df = categories_from_GSheets()
    platform_dict = dict(zip(categories_df["Platform"], categories_df["Category"]))
    selected_platforms = []
    for plat, categ in platform_dict.items():
        for s in selected_categories:
            if categ == s:
                selected_platforms.append(plat)
    return selected_platforms

# Chart Functions

## MentorCloud LINES Plot Function


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- MentorCloud Bar Plot Function
def mentorCloud_line_plot(
    chart_title,
    y_params,
    group_by,
    selected_platforms,
    labels,
    fill_colors,
    hovermode,
    markers,
    mk_size,
    box_line,
    user_growth,
    legend,
    he,
    range_dates
):
    df = filter_selected_platforms(selected_platforms, range_dates)
    #############################################
    df.sort_values(["Corrected_Date"], ascending=True, inplace=True)
    #############################################
    if len(selected_platforms) == 1:
        company = selected_platforms[0]
    elif len(selected_platforms) == len(complete_df["Platform"].unique()):
        company = "All Platforms"
    elif len(selected_platforms) > 1:
        company = "Multiple Platforms"
    #############################################
    sub_df = df.loc[df["Platform"].isin(selected_platforms)].reset_index()
    sub_df.drop(columns="Platform", inplace=True)
    sub_df.sort_values(["Corrected_Date"], ascending=True, inplace=True)
    sub_df.drop(columns=["index"], inplace=True)
    #############################################
    if group_by == "Corrected_Date":
        x_title = "Weeks"
        df_1 = sub_df.groupby(pd.Grouper(key='Corrected_Date', freq='W-MON')).sum().reset_index().sort_values('Corrected_Date')
    elif group_by == "Corrected_Month_Year":
        x_title = "Year, Month*"
        df_1 = sub_df.groupby("Corrected_Month_Year").sum().reset_index()
    #############################################
    if chart_title == "Messages":
        df_1['messages_rate'] = (df_1["Total_Messages"]/df_1["Message_Posts"]).astype(float)
    if user_growth:
        df_1["cum_profiled"] = df_1["Change_in_Profiled"].cumsum()
        df_1["cum_mentors"] = df_1["Change_in_Profiled_Mentors"].cumsum()
        df_1["cum_mentees"] = df_1["Change_in_Profiled_Mentees"].cumsum()
    #############################################
    if markers:
        mk = True
    else:
        mk = False
    fig = px.line(
        df_1,
        x=group_by,
        y=y_params,
        height=he,
        markers=mk,
        color_discrete_sequence=fill_colors,
    )
    #############################################
    fig.update_traces(
        hovertemplate="%{y}",
        textposition="bottom right",
        marker=dict(
            size=mk_size,
        ),
    )
    #############################################
    if legend:
        legend = True
    else:
        legend = False
    fig.update_layout(
        title="%s - (%s)" % (chart_title, company),
        hovermode=hovermode,
        showlegend=legend,
        transition_duration=500,
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title=None,
        font=dict(family="Arial", color="Black"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
        hoverlabel=dict(
            bgcolor="#F3F3F3",
            font_size=12,
        ),
    )
    #############################################
    new_labels = labels
    fig.for_each_trace(
        lambda t: t.update(
            name=new_labels[t.name],
            legendgroup=new_labels[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_labels[t.name]),
        )
    )
    #############################################
    if box_line:
        fig.add_shape(
            # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="#363435",
                width=1,
            ),
        )
    #############################################
    return fig

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

## MentorCloud BARS Plot Function


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- MentorCloud Bar Plot Function
def mentorCloud_bar_plot(
    chart_title,
    y_params,
    group_by,
    selected_platforms,
    labels,
    fill_colors,
    hovermode,
    barmode,
    line_color,
    bar_line,
    box_line,
    legend,
    he,
    range_dates
):
    df = filter_selected_platforms(selected_platforms, range_dates)
    #############################################
    df.sort_values(["Corrected_Date"], ascending=True, inplace=True)
    #############################################
    if len(selected_platforms) == 1:
        company = selected_platforms[0]
    elif len(selected_platforms) == len(complete_df["Platform"].unique()):
        company = "All Platforms"
    elif len(selected_platforms) > 1:
        company = "Multiple Platforms"
    #############################################
    sub_df = df.loc[df["Platform"].isin(selected_platforms)].reset_index()
    sub_df.drop(columns="Platform", inplace=True)
    sub_df.sort_values(["Corrected_Date"], ascending=True, inplace=True)
    sub_df.drop(columns=["index"], inplace=True)
    #############################################
    if group_by == "Corrected_Date":
        x_title = "Weeks"
        df_1 = sub_df.groupby(pd.Grouper(key='Corrected_Date', freq='W-MON')).sum().reset_index().sort_values('Corrected_Date')
    elif group_by == "Corrected_Month_Year":
        x_title = "Year, Month*"
        df_1 = sub_df.groupby("Corrected_Month_Year").sum().reset_index()
    #############################################
    fig = px.bar(
        df_1,
        barmode=barmode,
        x=group_by,
        y=y_params,
        height=he,
        text_auto=".2s",
        color_discrete_sequence=fill_colors,
    )
    #############################################
    if bar_line:
        l_color = line_color
        l_width = 3
    else:
        l_color = "#fff"
        l_width = 0
    fig.update_traces(
        hovertemplate="%{y}",
        textangle=-70,
        textposition="outside",
        texttemplate="%{y:.0f}",
        cliponaxis=False,
        marker_line_color=l_color,
        marker_line_width=l_width,
        # textposition="bottom rightd",
    )
    #############################################
    if legend:
        legend = True
    else:
        legend = False
    fig.update_layout(
        title="%s - (%s)" % (chart_title, company),
        hovermode=hovermode,
        showlegend=legend,
        transition_duration=500,
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title=None,
        font=dict(family="Arial", color="Black"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0.6)",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
        hoverlabel=dict(
            bgcolor="#F3F3F3",
            font_size=12,
        ),
    )
    #############################################
    new_labels = labels
    fig.for_each_trace(
        lambda t: t.update(
            name=new_labels[t.name],
            legendgroup=new_labels[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_labels[t.name]),
        )
    )
    #############################################
    if box_line:
        fig.add_shape(
            # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="#363435",
                width=1,
            ),
        )
    #############################################
    return fig

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

## MentorCloud AREA Plot Function


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- MentorCloud Area Plot Function
def mentorCloud_area_plot(
    chart_title,
    y_params,
    group_by,
    selected_platforms,
    labels,
    fill_colors,
    hovermode,
    markers,
    mk_size,
    box_line,
    cum_rela,
    legend,
    he,
    range_dates
):
    df = filter_selected_platforms(selected_platforms, range_dates)
    #############################################
    df.sort_values(["Corrected_Date"], ascending=True, inplace=True)
    #############################################
    if len(selected_platforms) == 1:
        company = selected_platforms[0]
    elif len(selected_platforms) == len(complete_df["Platform"].unique()):
        company = "All Platforms"
    elif len(selected_platforms) > 1:
        company = "Multiple Platforms"
   #############################################
    sub_df = df.loc[df["Platform"].isin(selected_platforms)].reset_index()
    sub_df.drop(columns="Platform", inplace=True)
    sub_df.sort_values(["Corrected_Date"], ascending=True, inplace=True)
    sub_df.drop(columns=["index"], inplace=True)
    #############################################
    if group_by == "Corrected_Date":
        x_title = "Weeks"
        df_1 = sub_df.groupby(pd.Grouper(key='Corrected_Date', freq='W-MON')).sum().reset_index().sort_values('Corrected_Date')
    elif group_by == "Corrected_Month_Year":
        x_title = "Year, Month*"
        df_1 = sub_df.groupby("Corrected_Month_Year").sum().reset_index()
    #############################################
    if cum_rela:
        df_1["cum_active_rela"] = df_1["No:_of_Active_Relationships"].cumsum()
    #############################################
    if markers:
        mk = True
    else:
        mk = False
    fig = px.area(
        df_1,
        x=group_by,
        y=y_params,
        height=he,
        markers=mk,
        color_discrete_sequence=fill_colors,
    )
    #############################################
    fig.update_traces(
        hovertemplate="%{y}",
        textposition="bottom right",
        marker=dict(
            size=mk_size,
        ),
    )
    #############################################
    if legend:
        legend = True
    else:
        legend = False
    fig.update_layout(
        title="%s - (%s)" % (chart_title, company),
        hovermode=hovermode,
        showlegend=legend,
        transition_duration=500,
        xaxis_title=x_title,
        yaxis_title=None,
        legend_title=None,
        font=dict(family="Arial", color="Black"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
        hoverlabel=dict(
            bgcolor="#D0CDCE",
            font_size=12,
        ),
    )
    #############################################
    new_labels = labels
    fig.for_each_trace(
        lambda t: t.update(
            name=new_labels[t.name],
            legendgroup=new_labels[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, new_labels[t.name]),
        )
    )
    #############################################
    if box_line:
        fig.add_shape(
            # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="#F3F3F3",
                width=1,
            ),
        )
    #############################################
    return fig

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

# TAB 1 CONTENT


## 1. Growth in Completed Profiles (Bar)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_1(group_by, selected_platforms, range_dates):
    fig_1_args = [
        "Growth in Completed Profiles",  ## chart_title
        [
            "Change_in_Profiled",
            "Change_in_Profiled_Mentees",
            "Change_in_Profiled_Mentors",
        ],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {
            "Change_in_Profiled": "Total Profiled",
            "Change_in_Profiled_Mentees": "Profiled Mentors",
            "Change_in_Profiled_Mentors": "Profiled Mentees",
        },  ## labels
        ["#58A65C", "#69BBC4", "#EB752F", "#4D9098"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#ffffff",  ## line_color
        False,  ## bar_line
        True,  ## box_line
        True,  ## legend
        460,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_1 = mentorCloud_bar_plot(*fig_1_args)
    return fig_1

## 2. Totals # of Profiles Cumulative (Line)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_2(group_by, selected_platforms, range_dates):
    fig_2_args = [
        "Total # of Profiles",  ## chart_title
        ["cum_profiled", "cum_mentees", "cum_mentors"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {
            "cum_profiled": "Total Profiled",
            "cum_mentors": "Profiled Mentors",
            "cum_mentees": "Profiled Mentees",
        },  ## labels
        ["#58A65C", "#69BBC4", "#EB752F", "#4D9098"],  ## fill_colors
        "x unified",  ## hovermode
        True,  ## markers
        6,  ## mk_size
        True,  ## box_line
        True,  ## user_growth
        True,  ## legend
        460,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_2 = mentorCloud_line_plot(*fig_2_args)
    return fig_2

# TAB 2 CONTENT


## 3. Total Mentoring Requests (Bar)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_3(group_by, selected_platforms, range_dates):
    fig_3_args = [
        "Total Mentoring Requests",  ## chart_title
        ["Tot_mentoring_requests"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"Tot_mentoring_requests": "Total Requests"},  ## labels
        ["#ffffff"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        True,  ## bar_line
        True,  ## box_line
        False,  ## legend
        460,  ## he
        range_dates
    ]

    #################################################################################################################################################
    fig_3 = mentorCloud_bar_plot(*fig_3_args)
    return fig_3

## 4. Started Relationships (Bar)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Started Relationships
def return_fig_4(group_by, selected_platforms, range_dates):
    fig_4_args = [
        "Started Relationships",  ## chart_title
        ["Relationships_started"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"Relationships_started": "Started Relationships"},  ## labels
        ["#ffffff"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        True,  ## bar_line
        True,  ## box_line
        False,  ## legend
        460,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_4 = mentorCloud_bar_plot(*fig_4_args)
    return fig_4

## 5. Expired Requests (Bar)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Started Relationships
def return_fig_5(group_by, selected_platforms, range_dates):
    fig_5_args = [
        "Expired Requests",  ## chart_title
        ["Diff_of_Request_Expired"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"Diff_of_Request_Expired": "Expired Relationships"},  ## labels
        ["#ffffff"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        True,  ## bar_line
        True,  ## box_line
        False,  ## legend
        460,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_5 = mentorCloud_bar_plot(*fig_5_args)
    return fig_5

## 6. Declined Requests (Bar)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Started Relationships
def return_fig_6(group_by, selected_platforms, range_dates):
    fig_6_args = [
        "Declined Requests",  ## chart_title
        ["Diff_of_Request_Declined"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"Diff_of_Request_Declined": "Declined Relationships"},  ## labels
        ["#ffffff"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        True,  ## bar_line
        True,  ## box_line
        False,  ## legend
        460,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_6 = mentorCloud_bar_plot(*fig_6_args)
    return fig_6

## 7. Removed Requests (Bar)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Started Relationships
def return_fig_7(group_by, selected_platforms, range_dates):
    fig_7_args = [
        "Removed Requests",  ## chart_title
        ["Diff_of_Request_Removed"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"Diff_of_Request_Removed": "Removed Relationships"},  ## labels
        ["#ffffff"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        True,  ## bar_line
        True,  ## box_line
        False,  ## legend
        460,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_7 = mentorCloud_bar_plot(*fig_7_args)
    return fig_7

# TAB 3 CONTENT


## 8. Meetings (Line)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_8(group_by, selected_platforms, range_dates):
    if group_by == 'Corrected_Month_Year':
        mk_size = 32
    else:
        mk_size = 12
    fig_8_args = [
        "Meetings",  ## chart_title
        ["Meetings_(Completed)", "Meetings_(Upcoming)"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {
            "Meetings_(Completed)": "Completed Meetings",
            "Meetings_(Upcoming)": "Upcoming Meetings",
        },  ## labels
        ["#58A65C", "#E2B24B"],  ## fill_colors
        "x unified",  ## hovermode
        True,  ## markers
        mk_size,  ## mk_size
        True,  ## box_line
        True,  ## user_growth
        True,  ## legend
        880,  ## he
        range_dates
    ]
    fig_8 = mentorCloud_line_plot(*fig_8_args)
    #################################################################################################################################################
    return fig_8

# TAB 4 CONTENT


## 9. Messages

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_9(group_by, selected_platforms, range_dates):
    fig_9A_args = [
        "Messages",  ## chart_title
        ["messages_rate"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {  ## labels
            "messages_rate": "Tot msg / Post msg",
        },
        ["#BBC2FA"],  ## fill_colors
        "x unified",  ## hovermode
        True,  ## markers
        6,  ## mk_size
        True,  ## box_line
        True,  ## user_growth
        False,  ## legend
        880,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_9_A = mentorCloud_line_plot(*fig_9A_args)

    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    ######################### ----------- Total Mentoring Requests Plot
    fig_9B_args = [
        "Messages",  ## chart_title
        ["Message_Posts", "Total_Messages"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {
            "Message_Posts": "Posted Messages",
            "Total_Messages": "Total Messages"
        },  ## labels
        ["#0E28B7", "#48A0F8"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        False,  ## bar_line
        True,  ## box_line
        True,  ## legend
        880,  ## he
        range_dates
    ]

    #################################################################################################################################################
    fig_9_B = mentorCloud_bar_plot(*fig_9B_args)
    fig_9_A.update_traces(yaxis="y2")

    fig_9 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_9.add_traces(fig_9_A.data + fig_9_B.data)

    fig_9.update_layout(
        height=880,
        title="Messages",  # "%s - (%s)"%(chart_title,company),
        hovermode="x unified",
        showlegend=True,
        transition_duration=500,
        xaxis_title="Time",
        yaxis_title=None,
        yaxis2_title="% Total messages / Posted messages",
        legend_title=None,
        font=dict(family="Arial", color="Black"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0.6)",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
    )
    fig_9.layout.yaxis2.tickformat = ',.1%'
    return fig_9

# TAB 5 CONTENT


## 10. Relationships Monthly / Weekly (Line + Shadowed Area)


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_10(group_by, selected_platforms, range_dates):
    fig_10A_args = [
        "Active Relationships",  ## chart_title
        ["No:_of_Active_Relationships"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"No:_of_Active_Relationships": "Active Relationships"},  ## labels
        ["#E2B24B"],  ## fill_colors
        "x unified",  ## hovermode
        True,  ## markers
        6,  ## mk_size
        True,  ## box_line
        True,  ## user_growth
        False,  ## legend
        450,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_10_A = mentorCloud_line_plot(*fig_10A_args)

    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    ######################### ----------- Total Mentoring Requests Plot
    fig_10B_args = [
        "Relationships by Role",  ## chart_title
        ["cum_active_rela"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"cum_active_rela": "Active (Cumulative)"},  ## labels
        ["#FBEBB5"],  ## fill_colors
        "x unified",  ## hovermode
        False,  ## markers
        0,  ## mk_size
        False,  ## box_line
        True,  ## cum_rela
        True,  ## legend
        450,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_10_B = mentorCloud_area_plot(*fig_10B_args)
    fig_10_A.update_traces(yaxis="y2")

    fig_10 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_10.add_traces(fig_10_A.data + fig_10_B.data)

    fig_10.update_layout(
        height=450,
        title="Relationships",  # "%s - (%s)"%(chart_title,company),
        hovermode="x unified",
        showlegend=True,
        transition_duration=500,
        xaxis_title="Time",
        yaxis_title=None,
        yaxis2_title="Completed Relationships",
        legend_title=None,
        font=dict(family="Arial", color="Black"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0.6)",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
    )
    return fig_10


## 11. Mentors, Mentees Rela (Line and Bars)

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Total Mentoring Requests Plot
def return_fig_11(group_by, selected_platforms, range_dates):
    fig_11A_args = [
        "N° of Relationships",  ## chart_title
        [
            "No:_of_Active_Relationships",
            "Mentors_Currently_in_Relationship",
            "Mentees_Currently_in_Relationship",
        ],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {  ## labels
            "No:_of_Active_Relationships": "Active Relationships",
            "Mentors_Currently_in_Relationship": "Mentors in Relationship",
            "Mentees_Currently_in_Relationship": "Mentees in Relationship",
        },
        ["#58A65C", "#69BBC4", "#EB752F", "#4D9098"],  ## fill_colors
        "x unified",  ## hovermode
        True,  ## markers
        6,  ## mk_size
        True,  ## box_line
        True,  ## user_growth
        False,  ## legend
        450,  ## he
        range_dates
    ]
    #################################################################################################################################################
    fig_11_A = mentorCloud_line_plot(*fig_11A_args)

    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    ######################### ----------- Total Mentoring Requests Plot
    fig_11B_args = [
        "N° of Relationships",  ## chart_title
        ["No:_of_Completed_Relationships"],  ## y_params
        group_by,  ## group_by
        selected_platforms,  ## selected_platforms
        {"No:_of_Completed_Relationships": "Completed Relationships"},  ## labels
        ["rgba(226, 178, 75, 0.7)", "#69BBC4", "#EB752F", "#4D9098"],  ## fill_colors
        "x unified",  ## hovermode
        "group",  ## barmode
        "#4D9098",  ## line_color
        False,  ## bar_line
        True,  ## box_line
        True,  ## legend
        460,  ## he
        range_dates
    ]

    #################################################################################################################################################
    fig_11_B = mentorCloud_bar_plot(*fig_11B_args)
    fig_11_A.update_traces(yaxis="y2")

    fig_11 = make_subplots(specs=[[{"secondary_y": True}]])
    fig_11.add_traces(fig_11_A.data + fig_11_B.data)

    fig_11.update_layout(
        height=460,
        title="Relationships",  # "%s - (%s)"%(chart_title,company),
        hovermode="x unified",
        showlegend=True,
        transition_duration=500,
        xaxis_title="Time",
        yaxis_title=None,
        yaxis2_title="Completed Relationships",
        legend_title=None,
        font=dict(family="Arial", color="Black"),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0.6)",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(
            l=2,
            r=2,
            b=2,
            t=50,
        ),
    )
    return fig_11

# TAB 6 CONTENT


## 12. Report by Dates (Table?)

# TAB 7 CONTENT


## 13. Report by Dates (Table?)

# DASH APPLICATION


## Start Application


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Start Application
external_scripts = [
    '/assets/aos.js',
    ]

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=0.9, maximum-scale=1.2, minimum-scale=0.5,",
        }
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=external_scripts,
)
server = app.server

## Tabs Design


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Tabs Design
#################################################################
#################################################################
######################### -----------  Tab 1
def tab1_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=return_fig_1(
                                            group_by, selected_platforms, range_dates
                                        )
                                    )
                                ),
                                html.Div(
                                    dcc.Graph(
                                        figure=return_fig_2(
                                            group_by, selected_platforms, range_dates
                                        )
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################
#################################################################
######################### -----------  Tab 2
def tab2_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    dcc.Graph(
                                                        figure=return_fig_3(
                                                            group_by, selected_platforms, range_dates
                                                        )
                                                    )
                                                )
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    dcc.Graph(
                                                        figure=return_fig_4(
                                                            group_by, selected_platforms, range_dates
                                                        )
                                                    )
                                                )
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    dcc.Graph(
                                                        figure=return_fig_5(
                                                            group_by, selected_platforms, range_dates
                                                        )
                                                    )
                                                )
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    dcc.Graph(
                                                        figure=return_fig_6(
                                                            group_by, selected_platforms, range_dates
                                                        )
                                                    )
                                                )
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    dcc.Graph(
                                                        figure=return_fig_7(
                                                            group_by, selected_platforms, range_dates
                                                        )
                                                    )
                                                )
                                            ],
                                            width=4,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################
#################################################################
######################### -----------  Tab 3
def tab3_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=return_fig_8(
                                            group_by, selected_platforms, range_dates
                                        )
                                    )
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################
#################################################################
######################### -----------  Tab 4
def tab4_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=return_fig_9(
                                            group_by, selected_platforms, range_dates
                                        )
                                    )
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################
#################################################################
######################### -----------  Tab 5
def tab5_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=return_fig_10(
                                            group_by, selected_platforms, range_dates
                                        )
                                    )
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    dcc.Graph(
                                        figure=return_fig_11(
                                            group_by, selected_platforms, range_dates
                                        )
                                    )
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################
#################################################################
######################### -----------  Tab 6
def tab6_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P("Tab 6!", className="card-text"),
                                ## html.Div(dcc.Graph(figure=fig_9)),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################
#################################################################
######################### -----------  Tab 7
def tab7_content(group_by, selected_platforms, range_dates):
    tab_content = dbc.Card(
        dbc.CardBody(
            [
                #############################################
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P("Tab 7!", className="card-text"),
                                ## html.Div(dcc.Graph(figure=fig_9)),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        className="mt-3",
        style={
            "border-style": "hidden",
            "maxHeight": "90vh",
            "overflow": "scroll",
        },
    )
    return tab_content


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Tabs Disposition
def return_the_tabs():
    tabs = html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(label="User Growth", tab_id="tab-1"),
                    dbc.Tab(label="Mentoring Requests", tab_id="tab-2"),
                    dbc.Tab(label="Relationships", tab_id="tab-5"),
                    dbc.Tab(label="Meetings", tab_id="tab-3"),
                    dbc.Tab(label="Messages", tab_id="tab-4"),
                    dbc.Tab(label="--", tab_id="tab-6"),
                    dbc.Tab(label="KPIs", tab_id="tab-7"),
                ],
                id="tabs",
                active_tab="tab-1",
                style={
                    "maxHeight": "90vh",
                    "overflow": "scroll",
                    "margin": "0px",
                    "padding": "0px",
                },
            ),
            html.Div(id="content"),
        ]
    )
    return tabs

## Layout


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### -----------  App Layout
def serve_layout():
    global complete_df
    global complete_df_interpolated
    global categories
    #############################################
    complete_df = dataFrame_from_GSheets()
    categories = sorted(categories_from_GSheets()['Category'].unique())
    selected_platforms = sorted(complete_df["Platform"].unique())
    all_platforms = sorted(complete_df["Platform"].unique())
    #############################################
    min_date = complete_df['Corrected_Date'].min().date()
    max_date = complete_df['Corrected_Date'].max().date()
    last_6_months = max_date - timedelta(days=181)
    #############################################
    latest_date_srt = str(max_date.day) + ' ' +  max_date.strftime("%b") + ', ' + str(max_date.year)
    #############################################
    csr_layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            #############################################
                            return_the_tabs(),
                            #############################################
                        ]
                    ),
                    #############################################
                    dbc.Col(
                        [
                            dbc.Alert([html.B("Data as of "), latest_date_srt], color="#BFE8D6", style={"maxHeight": "4vh", "margin-top":"-10px"}),
                            dbc.Row([
                                dbc.Col(
                                    DashIconify(
                                        icon="flat-color-icons:calendar",
                                        width=50,
                                        height=50,
                                    ),
                                    width=2,
                                    style={"margin-left":"38px"}
                                ),
                                #############################################
                                dbc.Col(
                                    dcc.DatePickerRange(
                                        id='csr-date-picker',
                                        display_format='MMM DD, YYYY',
                                        min_date_allowed=min_date,
                                        max_date_allowed=max_date,
                                        #initial_visible_month=last_6_months,
                                        start_date=last_6_months,
                                        end_date=max_date,
                                        with_portal=False,
                                        #disabled_days
                                        style={
                                            "margin-bottom": "13px",
                                        },
                                    ),
                                ),
                            ]),
                            #############################################
                            html.P(
                                "Platform Categories",
                                className="card-text",
                                style={"font-size": "18px", "font-weight": "bold", "border-bottom": "2px solid #8D8B8B"},
                            ),
                            #############################################
                            dbc.Row(
                                [
                                    dbc.Col([
                                        #############################################
                                        dcc.Checklist(
                                            ['Select All Categories'],
                                            [],
                                            id='csr-all-categories',
                                            labelStyle=dict(display="block"),
                                            style={
                                                "overflow": "scroll",
                                                "margin": "0px",
                                                "font-size": "13px",
                                            },
                                        ),
                                        #############################################
                                        dcc.Checklist(
                                            sorted(categories),
                                            ['1 - Key Account'],
                                            id='csr-categories',
                                            labelStyle=dict(display="block"),
                                            style={
                                                "overflow": "scroll",
                                                "margin": "0px",
                                                "font-size": "13px",
                                            },
                                        ),
                                        #############################################
                                    ])
                                ],
                                style={
                                    "height": "20vh",
                                    "margin": "0px",
                                    "overflow": "scroll",
                                    "margin-top": "-5px",
                                },
                            ),
                            #############################################
                            html.P(
                                className="card-text",
                                style={"font-size": "18px", "font-weight": "bold", "border-bottom": "2px solid #8D8B8B", "margin-bottom": "10px"},
                            ),
                            #############################################
                            dbc.Row(
                                [
                                    #############################################
                                    dbc.Col(
                                        [
                                            html.P("Monthly*", style={"text-align":"center", "padding-top":"10px"}),
                                            daq.BooleanSwitch(
                                                id="csr-toggle",
                                                color="#9B51E0",
                                                vertical=True,
                                                style={'transform': 'rotate(180deg)'},
                                            ),
                                            html.P("Weekly", style={"text-align":"center", "padding-top":"10px"}),
                                        ],
                                    ),
                                    #############################################
                                    dbc.Col(
                                        [
                                            html.P("Categories", style={"text-align":"center", "padding-top":"10px"}),
                                            daq.BooleanSwitch(
                                                id="csr-toggle-2",
                                                color="#9B51E0",
                                                vertical=True,
                                                style={'transform': 'rotate(180deg)'},
                                            ),
                                            html.P("Platforms", style={"text-align":"center", "padding-top":"10px"}),
                                        ]
                                    ),
                                    #############################################
                                ],
                                style={
                                    "margin-top": "-10px",
                                    "padding": "0px",
                                    "width": "100%",
                                    "display": "inline-flex",
                                },
                            ),
                            #############################################
                            html.P(
                                "Platforms",
                                className="card-text",
                                style={"font-size": "18px", "font-weight": "bold", "border-bottom": "2px solid #8D8B8B", "margin-top":"-15px"},
                            ),
                            #############################################
                            dbc.Button("X", outline=True, color="secondary", id="no-text-btn", className="me-1", style={"position": "absolute", "z-index": "1", "scale": "80%", "left": "97%"}),
                            dbc.Input(placeholder="Search by Platform...", type="text", id='input-platforms', style={'margin':'12px 5px', 'width':'95%'}),
                            #############################################
                            dbc.Row(
                                [
                                    #############################################
                                    dcc.Checklist(
                                        all_platforms,
                                        all_platforms,
                                        id='csr-platforms',
                                        className='filterCSR',
                                        labelStyle=dict(display="block"),
                                    ),
                                    #############################################
                                ],
                                style={
                                    "margin": "0px",
                                    "margin-top": "-5px",
                                    "overflow-y": "scroll",
                                },
                            ),
                        ],
                        #############################################
                        width=3,
                        style={"overflow-y": "scroll"},
                    ),
                ],
                style={"padding-top": "0px", "overflow-y": "hidden"},
            ),
        ],
    style={'backgroundColor':'#FBFBFB'})
    return csr_layout
#############################################
#############################################
app.layout = serve_layout

## Callbacks


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
######################### ----------- Callbacks
###########################################################################################################
######################### ----------- Input Categories Callback
@app.callback(
    Output('input-platforms', 'value'),
    Input('no-text-btn', 'n_clicks')
)
def on_button_click(n):
    if n is None:
        return
    else:
        return ""

###########################################################################################################
###########################################################################################################
######################### ----------- All Categories Callback
@app.callback(
    Output('csr-categories', 'value'),
    Input('csr-all-categories', 'value')
)
def change_values_0(s_all):
    return categories if len(s_all) > 0 else ['1 - Key Account']

###########################################################################################################
###########################################################################################################
######################### ----------- Categories Callback
@app.callback(
    [Output('csr-platforms', 'value'),
    Output('csr-all-categories', 'style')],
    [Input('csr-categories', 'value'),
    Input('csr-all-categories', 'value')]
)
def change_values_1(categor, s_all):
    li = ",".join(list(categor))
    selected_categories = list(li.split(","))
    selected_platforms = platforms_list(selected_categories)
    if len(s_all) > 0 and len(categor)!= len(categories):
        style = {'color': '#BDBDBD', "font-size": "13px"}
    else:
        style = {'color': 'inherit', "font-size": "13px"}
    return selected_platforms, style

###########################################################################################################
###########################################################################################################
######################### ----------- Platforms Callback
@app.callback(
    [Output('csr-platforms', 'style'),
    Output('csr-categories', 'style')],
    Input('csr-toggle-2', 'on')
)
def change_values_2(on):
    if on:
        style_1 = {'color': '#222529', "font-size": "13px", "maxHeight": "40.5vh", "margin": "0px"}
        style_2 = {'color': '#141315', "font-size": "13px"}
    else:
        style_1 = {'color': '#141315', "font-size": "13px", "maxHeight": "40.5vh", "margin": "0px"}
        style_2 = {'color': '#222529', "font-size": "13px"}
    return style_1, style_2

###########################################################################################################
###########################################################################################################
######################### ----------- Tabs Callback
@app.callback(
    Output('content', 'children'),
    [
        Input('tabs', 'active_tab'),
        Input('csr-toggle', 'on'),
        Input('csr-categories', 'value'),
        Input('csr-toggle-2', 'on'),
        Input('csr-platforms', 'value'),
        Input('csr-date-picker', 'start_date'),
        Input('csr-date-picker', 'end_date')
    ]
)
def populate_tabs(at, on, categor, on_2, plats, start_date, end_date):
    aggregation = ("Corrected_Date", "Corrected_Month_Year")
    if on_2:
        li_2 = ",".join(list(plats))
        selected_platforms = list(li_2.split(","))
    else:
        li = ",".join(list(categor))
        selected_categories = list(li.split(","))
        selected_platforms = platforms_list(selected_categories)
    ##################################################
    if on:
        group_by = aggregation[0]
    else:
        group_by = aggregation[1]
    ##################################################
    range_dates = (date.fromisoformat(start_date),date.fromisoformat(end_date))
    ##################################################   
    if at == "tab-1":
        return tab1_content(group_by, selected_platforms, range_dates)
    elif at == "tab-2":
        return tab2_content(group_by, selected_platforms, range_dates)
    elif at == "tab-3":
        return tab3_content(group_by, selected_platforms, range_dates)
    elif at == "tab-4":
        return tab4_content(group_by, selected_platforms, range_dates)
    elif at == "tab-5":
        return tab5_content(group_by, selected_platforms, range_dates)
    elif at == "tab-6":
        return tab6_content(group_by, selected_platforms, range_dates)
    elif at == "tab-7":
        return tab7_content(group_by, selected_platforms, range_dates)
    ##################################################
    return

## Run Application

#################################################################
#################################################################
######################### -----------  Server Initiation
if __name__ == "__main__":
    app.run_server(debug=False)
