import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_classification_report(report_input, name):
    """
    Parse and visualize a classification report from sklearn.
    
    Parameters:
    -----------
    report_input : str
        Either the classification report text or a path to a file containing it
    name : str
        The filename to save the plot
    """
    # Check if input is a file path
    try:
        with open(report_input, 'r') as f:
            report_text = f.read()
    except FileNotFoundError:
        # If file doesn't exist, assume it's the text itself
        report_text = report_input
    
    # Parse the report into a list of lines
    lines = report_text.strip().split('\n')
    
    # Extract data rows (skip header and summary rows)
    data = []
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) == 5 and parts[0] not in ['accuracy', 'macro', 'weighted']:
            label = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1_score = float(parts[3])
            support = int(parts[4])
            data.append([label, precision, recall, f1_score, support])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    # Create figure with multiple subplots (now 1x3 instead of 2x2)
    fig = plt.figure(figsize=(18, 6))
    
    # Subplot 1: Bar chart comparing precision, recall, and f1-score
    plt.subplot(1, 3, 1)
    x = np.arange(len(df['Class']))
    width = 0.25
    
    plt.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
    plt.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Classification Metrics by Class')
    plt.xticks(x, df['Class'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Subplot 2: F1-Score sorted
    plt.subplot(1, 3, 2)
    df_sorted = df.sort_values('F1-Score', ascending=True)
    colors = plt.cm.RdYlGn(df_sorted['F1-Score'])
    
    plt.barh(df_sorted['Class'], df_sorted['F1-Score'], color=colors)
    plt.xlabel('F1-Score')
    plt.ylabel('Class')
    plt.title('F1-Score by Class (Sorted)')
    plt.xlim(0, 1.05)
    plt.grid(axis='x', alpha=0.3)
    
    # Subplot 3: Summary statistics table
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('tight')
    ax3.axis('off')
    
    # Create summary data (no empty lines)
    best_f1_idx = df['F1-Score'].idxmax()
    worst_f1_idx = df['F1-Score'].idxmin()
    below_threshold = df[df['F1-Score'] < 0.85]['Class'].tolist()
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Classes', str(len(df))],
        ['Total Samples', str(df['Support'].sum())],
        ['Avg Precision', f"{df['Precision'].mean():.3f}"],
        ['Avg Recall', f"{df['Recall'].mean():.3f}"],
        ['Avg F1-Score', f"{df['F1-Score'].mean():.3f}"],
        ['Best F1 Class', f"{df.loc[best_f1_idx, 'Class']} ({df['F1-Score'].max():.3f})"],
        ['Worst F1 Class', f"{df.loc[worst_f1_idx, 'Class']} ({df['F1-Score'].min():.3f})"],
        ['Classes < 0.85 F1', ', '.join(below_threshold) if below_threshold else 'None']
    ]
    
    table = ax3.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Classification Report Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df