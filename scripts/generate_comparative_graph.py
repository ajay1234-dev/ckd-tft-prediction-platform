import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

# Set font styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Define stages and data
stages = [
    "Patient\nRegistration",
    "Routine Lab\nTracking",
    "Progression\nDetection",
    "Trajectory &\nRisk Forecasting",
    "Intervention\nPlanning",
    "Final Outcome\n(Crash Prevention)"
]

x = np.arange(len(stages))

# Existing System (KFRE / Standard Static Care)
# Starts strong at basic registration/labs, drops sharply on progression/velocity detection and actionable short-term intervention
y_existing = [94, 88, 42, 58, 52, 50]

# Proposed System (Explainable Longitudinal TFT Framework)
# Matches on basic intake, significantly outperforms on sequence trajectory, early detection, and timely planning
y_proposed = [94, 88, 86, 91, 84, 88]

# 2. Create Figure
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

# 3. Add background shaded regions
# Left Stage (Existing Focus): x from -0.5 to 2.5
ax.axvspan(-0.5, 2.5, color='#eaf4fc', alpha=0.85, zorder=1)
# Right Stage (Proposed Focus): x from 2.5 to 5.5
ax.axvspan(2.5, 5.5, color='#fff4ea', alpha=0.85, zorder=1)

# Vertical dividing dashed line
ax.axvline(x=2.5, color='#333333', linestyle='--', linewidth=1.2, zorder=2)

# 4. Add Top Region Headers with Arrows
# Left region banner
ax.annotate(
    '', xy=(2.4, 107), xytext=(-0.4, 107),
    arrowprops=dict(arrowstyle='<->', color='#025ca0', lw=1.5)
)
ax.text(
    1.0, 108.5, 'Standard Care & Static Screening Stage\n(Existing System)',
    ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#025ca0'
)

# Right region banner
ax.annotate(
    '', xy=(5.4, 107), xytext=(2.6, 107),
    arrowprops=dict(arrowstyle='<->', color='#d35400', lw=1.5)
)
ax.text(
    4.0, 108.5, 'Prognostic Forecasting & Early Intervention Stage\n(Proposed System)',
    ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#d35400'
)

# 5. Grid lines
ax.grid(True, linestyle='--', alpha=0.45, color='#cbd5e1', zorder=2)
ax.set_axisbelow(True)

# 6. Plot Lines & Markers
# Existing System line (Blue)
line_exist = ax.plot(
    x, y_existing,
    color='#0284c7', linewidth=2.4, marker='o', markersize=8.5,
    label='Existing System (Standard Care & KFRE)', zorder=4
)

# Proposed System line (Orange)
line_prop = ax.plot(
    x, y_proposed,
    color='#ea580c', linewidth=2.4, marker='s', markersize=8.5,
    label='Proposed System (Explainable TFT Framework)', zorder=5
)

# 7. Add Data Labels with intelligent offsets to avoid collision
for i in range(len(x)):
    # Existing System label (Blue)
    if i == 0:
        ey_offset = -7.5
    elif i == 1:
        ey_offset = -7.5
    elif i == 2:
        ey_offset = -8
    elif i == 3:
        ey_offset = -7.5
    elif i == 4:
        ey_offset = -7.5
    else:
        ey_offset = -7.5
        
    ax.annotate(
        f'{y_existing[i]}%',
        xy=(x[i], y_existing[i]),
        xytext=(0, ey_offset),
        textcoords='offset points',
        ha='center', va='center',
        fontsize=9.5, fontweight='bold', color='#0369a1',
        zorder=6
    )

    # Proposed System label (Orange)
    if i == 0:
        py_offset = 7.5
    elif i == 1:
        py_offset = 7.5
    elif i == 2:
        py_offset = 7.5
    elif i == 3:
        py_offset = 7.5
    elif i == 4:
        py_offset = 7.5
    else:
        py_offset = 7.5

    ax.annotate(
        f'{y_proposed[i]}%',
        xy=(x[i], y_proposed[i]),
        xytext=(0, py_offset),
        textcoords='offset points',
        ha='center', va='center',
        fontsize=9.5, fontweight='bold', color='#c2410c',
        zorder=6
    )

# 8. Set Titles and Labels
ax.set_title(
    'Chronic Kidney Disease Care Lifecycle Performance Comparison',
    fontsize=14, fontweight='bold', pad=32, color='#0f172a'
)
ax.set_xlabel('Clinical Process Stage', fontsize=11, fontweight='bold', labelpad=12, color='#1e293b')
ax.set_ylabel('Diagnostic & Care Effectiveness Rate (%)', fontsize=11, fontweight='bold', labelpad=12, color='#1e293b')

# 9. Set Ticks and Limits
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=9.5, color='#334155')
ax.set_yticks(np.arange(0, 111, 20))
ax.set_yticklabels([f'{y}' for y in np.arange(0, 111, 20)], fontsize=9.5, color='#334155')
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0, 118)

# Clean spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#94a3b8')
    ax.spines[spine].set_linewidth(1.0)

# 10. Add Bottom Legend styled like card
legend = ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.18),
    ncol=2, frameon=True, fontsize=10,
    handletextpad=0.8, columnspacing=2.0
)
legend.get_frame().set_edgecolor('#cbd5e1')
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_boxstyle('round,pad=0.5')

plt.tight_layout()

# Save in reports/figures and root
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('5', exist_ok=True)

fig.savefig('reports/figures/comparative-analysis-graph.png', dpi=300, bbox_inches='tight')
fig.savefig('5/comparative-analysis-graph.png', dpi=300, bbox_inches='tight')
fig.savefig('comparative-analysis-graph.png', dpi=300, bbox_inches='tight')

print("Successfully generated comparative-analysis-graph.png in reports/figures/, 5/, and root!")
