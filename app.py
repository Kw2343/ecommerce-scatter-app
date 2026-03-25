from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DEFAULT_FILE = Path('/mnt/data/recommender_scatterplot_inputs.xlsx')
DEFAULT_SHEET = 'PlotData'
TOP_ORDER = ['Top1', 'Top2', 'Top3', 'Top4', 'Top5']


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map supported workbook column names to plotting columns."""
    required_base = ['DisplayLabel', 'Group']
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    x_candidates = [c for c in ['X_MaxCosSim', 'MaxCosine'] if c in df.columns]
    y_candidates = [c for c in ['Y_PredRating', 'Predicted_Rating'] if c in df.columns]
    if not x_candidates or not y_candidates:
        raise ValueError(
            'Workbook must contain X_MaxCosSim/MaxCosine and '
            'Y_PredRating/Predicted_Rating columns.'
        )

    rename_map = {}
    if 'X_MaxCosSim' in df.columns:
        rename_map['X_MaxCosSim'] = 'MaxCosine'
    if 'Y_PredRating' in df.columns:
        rename_map['Y_PredRating'] = 'Predicted_Rating'

    out = df.rename(columns=rename_map).copy()
    return out[['DisplayLabel', 'Group', 'MaxCosine', 'Predicted_Rating']]


def load_excel(file_or_path, sheet_name: str | None = None) -> pd.DataFrame:
    """Read the workbook from a path or Streamlit upload object."""
    if isinstance(file_or_path, (str, Path)):
        excel = pd.ExcelFile(file_or_path)
    else:
        file_or_path.seek(0)
        excel = pd.ExcelFile(file_or_path)

    chosen_sheet = sheet_name if sheet_name in excel.sheet_names else excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=chosen_sheet)
    return standardize_columns(df)


def split_groups(df: pd.DataFrame):
    top = df[df['Group'].isin(TOP_ORDER)].copy()
    top['TopSort'] = top['Group'].map({g: i for i, g in enumerate(TOP_ORDER, start=1)})
    top = top.sort_values('TopSort')

    near = df[df['Group'].eq('Near')].copy()
    far = df[df['Group'].eq('Far')].copy()
    random_pts = df[df['Group'].eq('Random')].copy()
    return top, near, far, random_pts


def hover_text_from_xy(x_vals, y_vals):
    """Return (x,y) text in the user's requested format."""
    return [f'({x:.2f},{y:.2f})' for x, y in zip(x_vals, y_vals)]


def create_interactive_plot(df: pd.DataFrame) -> go.Figure:
    top, near, far, random_pts = split_groups(df)

    x_min, x_max = df['MaxCosine'].min(), df['MaxCosine'].max()
    y_min, y_max = df['Predicted_Rating'].min(), df['Predicted_Rating'].max()
    x_pad = max((x_max - x_min) * 0.08, 0.02)
    y_pad = max((y_max - y_min) * 0.12, 0.05)

    fig = go.Figure()

    if not random_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=random_pts['MaxCosine'],
                y=random_pts['Predicted_Rating'],
                mode='markers',
                name='Random',
                marker=dict(size=7, color='rgba(0, 255, 255, 0.45)'),
                hoverinfo='skip',
                showlegend=True,
            )
        )

    if not top.empty:
        # Connecting line for the top 5.
        fig.add_trace(
            go.Scatter(
                x=top['MaxCosine'],
                y=top['Predicted_Rating'],
                mode='lines',
                name='Top 5 path',
                line=dict(color='blue', width=2),
                hoverinfo='skip',
                showlegend=True,
            )
        )

        # Markers + visible product labels, hover shows only coordinate pair.
        fig.add_trace(
            go.Scatter(
                x=top['MaxCosine'],
                y=top['Predicted_Rating'],
                mode='markers+text',
                name='Top 5',
                text=top['DisplayLabel'],
                textposition='top center',
                textfont=dict(size=11, color='blue'),
                marker=dict(size=12, color='blue'),
                customdata=hover_text_from_xy(top['MaxCosine'], top['Predicted_Rating']),
                hovertemplate='%{customdata}<extra></extra>',
                showlegend=True,
            )
        )

    if not near.empty:
        fig.add_trace(
            go.Scatter(
                x=near['MaxCosine'],
                y=near['Predicted_Rating'],
                mode='markers+text',
                name='Near',
                text=near['DisplayLabel'],
                textposition='bottom right',
                textfont=dict(size=11, color='green'),
                marker=dict(size=13, color='green', line=dict(color='black', width=1)),
                customdata=hover_text_from_xy(near['MaxCosine'], near['Predicted_Rating']),
                hovertemplate='%{customdata}<extra></extra>',
                showlegend=True,
            )
        )

    if not far.empty:
        fig.add_trace(
            go.Scatter(
                x=far['MaxCosine'],
                y=far['Predicted_Rating'],
                mode='markers+text',
                name='Far',
                text=far['DisplayLabel'],
                textposition='top right',
                textfont=dict(size=11, color='red'),
                marker=dict(size=13, color='red', line=dict(color='black', width=1)),
                customdata=hover_text_from_xy(far['MaxCosine'], far['Predicted_Rating']),
                hovertemplate='%{customdata}<extra></extra>',
                showlegend=True,
            )
        )

    fig.update_layout(
        title='XY Scatter Plot of Recommendation Candidates',
        xaxis_title='MaxCosine',
        yaxis_title='Predicted_Rating',
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
        hoverlabel=dict(font_size=10),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    fig.update_xaxes(
        range=[x_min - x_pad, x_max + x_pad],
        showgrid=True,
        gridcolor='rgba(0,0,0,0.12)',
        zeroline=False,
        showline=True,
        linecolor='black',
        ticks='outside',
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        range=[y_min - y_pad, y_max + y_pad],
        showgrid=True,
        gridcolor='rgba(0,0,0,0.12)',
        zeroline=False,
        showline=True,
        linecolor='black',
        ticks='outside',
        tickfont=dict(size=11),
    )

    return fig


def main():
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            'streamlit is required to run this app. Install it with: '
            'pip install streamlit plotly pandas openpyxl'
        ) from exc

    st.set_page_config(page_title='Recommendation Scatter Plot', layout='wide')
    st.title('Recommendation XY Scatter Plot')
    st.write(
        'Hover over any Top 5, Near, or Far point to see the coordinate pair in the '
        'format (MaxCosine, Predicted_Rating). Random points are de-emphasized and '
        'do not show hover text.'
    )

    uploaded_file = st.file_uploader('Upload an Excel file', type=['xlsx'])
    source = uploaded_file if uploaded_file is not None else DEFAULT_FILE

    if uploaded_file is None:
        st.caption(f'Using default workbook: {DEFAULT_FILE}')

    try:
        df = load_excel(source, sheet_name=DEFAULT_SHEET)
    except Exception as exc:
        st.error(f'Unable to read the workbook: {exc}')
        st.stop()

    fig = create_interactive_plot(df)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    important_rows = df[df['Group'].isin(TOP_ORDER + ['Near', 'Far'])].copy()
    if not important_rows.empty:
        important_rows['HoverValue'] = important_rows.apply(
            lambda r: f"({r['MaxCosine']:.2f},{r['Predicted_Rating']:.2f})", axis=1
        )
        st.expander('Rows used for Top 5 / Near / Far').dataframe(
            important_rows[['DisplayLabel', 'Group', 'MaxCosine', 'Predicted_Rating', 'HoverValue']],
            use_container_width=True,
        )

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download plotted data as CSV',
        data=csv_bytes,
        file_name='plotted_scatter_data.csv',
        mime='text/csv',
    )


if __name__ == '__main__':
    main()