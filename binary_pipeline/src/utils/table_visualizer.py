# multiclass_pipeline/utils/table_visualizer.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def save_styled_table(df, filename, folder, title=None, highlight_max=True):
    """
    Saves a pandas DataFrame as a visually appealing PNG table using Matplotlib.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Format numbers to 4 decimals
    df_display = df.copy()
    for col in df_display.select_dtypes(include=[np.number]).columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    
    fig, ax = plt.subplots(figsize=(max(10, len(df.columns)*1.5), len(df)*0.5 + 1.5))
    ax.axis('off')
    
    # Define colors
    header_color = '#2c3e50'
    row_even_color = '#ecf0f1'
    row_odd_color = 'white'
    highlight_color = '#27ae60' # Green for best values
    
    # Create the table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Apply styling
    for (row, col), cell in table.get_celld().items():
        # Header
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
        else:
            # Alternating row colors
            if row % 2 == 0:
                cell.set_facecolor(row_even_color)
            else:
                cell.set_facecolor(row_odd_color)
            
            # Highlight max values in numeric columns (if requested)
            if highlight_max:
                col_name = df.columns[col]
                try:
                    val = float(df_display.iloc[row-1, col])
                    # Check if this is the max value in its column
                    if val == df[col_name].max() and df[col_name].dtype in [np.float64, np.int64]:
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor(highlight_color)
                except:
                    pass

    if title:
        plt.title(title, fontsize=16, pad=30, weight='bold', color=header_color)
        
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Table saved: {save_path}")

def export_classification_report(report_dict, filename, folder, title):
    """
    Converts a classification report dictionary to a styled PNG.
    """
    df = pd.DataFrame(report_dict).transpose().reset_index()
    df.columns = ['Metric'] + list(df.columns[1:])
    save_styled_table(df, filename, folder, title, highlight_max=False)
