import plotly.express as px
import pandas as pd
import os
from pathlib import Path

def generate_sunburst_chart(csv_file_path, output_image_path):
    # Load the true and predicted results data
    results_df = pd.read_csv(csv_file_path)

    # Create a DataFrame to store the counts of each category
    data = {
        'labels': [],
        'sum': [],
        'parent': []
    }

    # Aggregate counts for Y2_test
    y2_counts = results_df['Y2_test'].value_counts()
    for label, count in y2_counts.items():
        data['labels'].append(label)
        data['sum'].append(count)
        data['parent'].append("Overall")

    # Aggregate counts for Y3_test and link them to their respective Y2_test categories
    y3_counts = results_df.groupby(['Y2_test', 'Y3_test']).size().reset_index(name='count')
    for _, row in y3_counts.iterrows():
        data['labels'].append(row['Y3_test'])
        data['sum'].append(row['count'])
        data['parent'].append(row['Y2_test'])

    # Aggregate counts for Y4_test and link them to their respective Y3_test categories
    y4_counts = results_df.groupby(['Y3_test', 'Y4_test']).size().reset_index(name='count')
    for _, row in y4_counts.iterrows():
        data['labels'].append(row['Y4_test'])
        data['sum'].append(row['count'])
        data['parent'].append(row['Y3_test'])

    # Create a DataFrame for visualization
    df_viz = pd.DataFrame(data)

    # Plotting the sunburst chart
    fig = px.sunburst(df_viz, path=['parent', 'labels'], values='sum', color='sum',
                      color_continuous_scale='Viridis',
                      title="Issues Visualization")

    # Save the plot as a file
    fig.write_image(str(output_image_path))

    # Show the plot
    fig.show()
