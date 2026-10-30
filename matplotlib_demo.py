import matplotlib.pyplot as plt
import numpy as np

# 1. Define the muted color palette
C_GOOD = "#628cb6"        # good credit risk / approved
C_BAD = "#c97b63"         # bad credit risk / denied
C_LINE = "#4d4d4d"        # threshold lines
C_FAIR = "#cfcfcf"        # "fair" band

# 2. Define the three custom style dictionaries
style_floating = {
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False, 
    "axes.grid": True, "grid.color": "#d4d4d4", "grid.linewidth": 1.0, "grid.linestyle": ":",
    "axes.axisbelow": True, "xtick.bottom": False, "ytick.left": False
}

style_vintage = {
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "font.family": "serif",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "grid.color": "#d5ceb8", "grid.linewidth": 0.7, "grid.linestyle": "-.",
    "axes.axisbelow": True, "axes.facecolor": "#fdf6e3", "figure.facecolor": "#fdf6e3",
    "text.color": "#332c23", "axes.labelcolor": "#332c23",
    "xtick.color": "#332c23", "ytick.color": "#332c23"
}

style_blueprint = {
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "font.family": "monospace",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.linewidth": 2.0, "axes.edgecolor": "#1a1a1a",
    "axes.grid": True, "grid.color": "#1a1a1a", "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.facecolor": "#f4f4f8", "figure.facecolor": "#f4f4f8",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a"
}

def plot_sample_chart(title):
    """Generates a sample bar chart incorporating all four custom colors."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Mock data
    groups = ['Q1', 'Q2', 'Q3', 'Q4']
    approved = [65, 78, 72, 85]
    denied = [35, 22, 28, 15]
    x = np.arange(len(groups))
    width = 0.35
    
    # Plot the C_FAIR band (e.g., an acceptable historical variance zone)
    ax.axhspan(40, 60, color=C_FAIR, alpha=0.5, label='Fair Zone')
    
    # Plot the C_LINE threshold
    ax.axhline(75, color=C_LINE, linestyle='--', linewidth=1.5, label='Target Threshold')
    
    # Plot the bars using C_GOOD and C_BAD
    ax.bar(x - width/2, approved, width, color=C_GOOD, label='Approved')
    ax.bar(x + width/2, denied, width, color=C_BAD, label='Denied')
    
    # Formatting
    ax.set_ylabel('Applicant Volume')
    ax.set_title(title, pad=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    
    # Legend with a border matching the C_LINE for cohesion
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=1, edgecolor=C_LINE)
    
    plt.tight_layout()
    plt.show()

# 3. Generate the plots using rc_context to avoid permanently altering global settings
with plt.rc_context(style_floating):
    plot_sample_chart('Style 1: Floating Minimalist')

with plt.rc_context(style_vintage):
    plot_sample_chart('Style 2: Vintage Journal')

with plt.rc_context(style_blueprint):
    plot_sample_chart('Style 3: Blueprint / Brutalist')