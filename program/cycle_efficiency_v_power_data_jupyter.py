#!/usr/bin/env python3
"""

Performance Limits of Low Carbon Power Generation Cycles
DOI: 

Interactive plot:
    -Figure 13
    -Cycle power versus cycle efficiency
    -Shows the Allam cycle with and without Air Separation Unit (ASU)
    -Shows the Intercooled-recuperated (ICR) cycle without post-combustion
    carbon capture, with 90%-, and 99% carbon capture
    -Allows to adjust modeled air separation work and specific reboiler duty
    to inspect the sensitivity to the gas separation penalties
    
    -Shaded regions show the tip clearance variation between 0.25- and 0.5 mm

Author: lsl37
Created: June, 2025
Modified: For ipywidgets compatibility
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Polygon

# Constants
POWER_OUTPUT = np.array([8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 320.0])

#Modeled inputs for gas separation penalties
INIT_ASU_WORK = 1.59  # MJ/kg O2
INIT_REBOILER_DUTY = 4.4  # MJ/kg CO2
INIT_REBOILER_DUTY_90 = 3.9  # MJ/kg CO2

# Color scheme
COLORS = {
    'allam_no_asu': '#08306b',
    'allam_w_asu': '#c6dbef', 
    'icr_no_ccs': '#800026',
    'icr_90_ccs': '#e31a1c',
    'icr_99_ccs': '#feb24c'
}

# Cycle names for visibility control
CYCLE_NAMES = [
    'Allam',
    'Allam with ASU',
    r'ICR 0% CO₂ Capture',
    r'ICR 90% CO₂ Capture', 
    r'ICR 99% CO₂ Capture'
]

def load_data():
    """Load all cycle efficiency data arrays."""
    
    # Allam cycle data
    allam_05_no_asu = np.array([57.24, 59.98, 62.53, 64.32, 66.26, 67.14, 67.46])
    allam_025_no_asu = np.array([62.45, 64.08, 65.61, 66.66, 67.86, 68.66, 68.82])
    allam_05_w_asu = np.array([45.83, 48.68, 51.16, 53.22, 54.68, 55.66, 56.06])
    allam_025_w_asu = np.array([51.0, 52.56, 54.20, 55.25, 56.38, 56.95, 57.19])
    
    # ICR (Integrated gasification combined cycle) data
    icr_05_no_ccs = np.array([53.26, 58.1, 60.87, 62.06, 62.56, 62.26, 62.084])
    icr_025_no_ccs = np.array([58.0, 60.67, 62.16, 62.76, 63.057, 62.56, 62.36])
    icr_05_90_ccs = np.array([50.40, 55.075, 57.77, 58.93, 59.41, 59.12, 58.95])
    icr_025_90_ccs = np.array([54.98, 57.57, 59.02, 59.61, 59.90, 59.41, 59.22])
    icr_05_99_ccs = np.array([47.26, 51.65, 54.19, 55.28, 55.74, 55.46, 55.30])
    icr_025_99_ccs = np.array([51.56, 54.00, 55.37, 55.92, 56.19, 55.74, 55.55])
    
    # Air separation unit sensitivity data (efficiency vs MJ/kg O2 work)
    asu_change_values = np.array([2.0, 1.79, 1.59, 1.39, 1.19, 0.99, 0.79, 0.59, 0.3])
    asu_efficiency_change = np.array([-3.38, -1.65, 0.0, 1.65, 3.3, 4.94, 6.59, 8.24, 10.63])
    
    # Reboiler duty sensitivity data for 99% capture
    reboiler_duty_99 = np.array([3.07, 3.27, 3.47, 3.57, 3.79, 4.39, 4.76])
    efficiency_99 = np.array([57.62, 57.70, 57.77, 57.76, 57.06, 54.2, 52.16])
    
    # Reboiler duty sensitivity data for 90% capture  
    reboiler_duty_90 = np.array([3.4, 3.6, 3.8, 3.9, 4.15, 4.48])
    efficiency_90 = np.array([57.96, 58.04, 58.11, 58.1, 57.2, 55.7])
    
    return {
        'allam_05_no_asu': allam_05_no_asu,
        'allam_025_no_asu': allam_025_no_asu,
        'allam_05_w_asu': allam_05_w_asu,
        'allam_025_w_asu': allam_025_w_asu,
        'icr_05_no_ccs': icr_05_no_ccs,
        'icr_025_no_ccs': icr_025_no_ccs,
        'icr_05_90_ccs': icr_05_90_ccs,
        'icr_025_90_ccs': icr_025_90_ccs,
        'icr_05_99_ccs': icr_05_99_ccs,
        'icr_025_99_ccs': icr_025_99_ccs,
        'asu_change_values': asu_change_values,
        'asu_efficiency_change': asu_efficiency_change,
        'reboiler_duty_99': reboiler_duty_99,
        'efficiency_99': efficiency_99,
        'reboiler_duty_90': reboiler_duty_90,
        'efficiency_90': efficiency_90
    }

def create_interpolation_functions(data):
    """Create interpolation functions for parameter sensitivity analysis."""
    
    # ASU work sensitivity function
    asu_change_func = interp1d(
        data['asu_change_values'], 
        data['asu_efficiency_change'], 
        kind='linear', 
        fill_value='extrapolate'
    )
    
    # Reboiler duty sensitivity functions
    eff_99_func = interp1d(
        data['reboiler_duty_99'], 
        data['efficiency_99'], 
        kind='quadratic', 
        fill_value='extrapolate'
    )
    
    eff_90_func = interp1d(
        data['reboiler_duty_90'], 
        data['efficiency_90'], 
        kind='quadratic', 
        fill_value='extrapolate'
    )
    
    def reboiler_change_99(reb_duty):
        return eff_99_func(reb_duty) - eff_99_func(INIT_REBOILER_DUTY)
    
    def reboiler_change_90(reb_duty):
        return eff_90_func(reb_duty) - eff_90_func(INIT_REBOILER_DUTY_90)
    
    return asu_change_func, reboiler_change_99, reboiler_change_90

def setup_plot():
    """Set up the main plot with proper formatting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right = 0.7)
    
    # Configure axes
    ax.set_xlabel('Power Output (MW)', fontsize=20)
    ax.set_ylabel('Efficiency (%)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='x', which='both', direction='in', top=True)
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    ax.minorticks_on()
    ax.grid(True, zorder=0, linestyle=':')
    
    # Add all spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    return fig, ax

def create_legend(ax):
    """Create a custom legend for the plot."""
    handles = []
    
    for i, (color, label) in enumerate(zip(COLORS.values(), CYCLE_NAMES)):
        handles.append(Polygon([(0,0),(10,0),(0,-10)], facecolor=color,
                              label=label, edgecolor='black', linewidth=1.5))
    
    leg = plt.legend(handles=handles, fontsize=12, ncol=1, 
                    loc='upper left', bbox_to_anchor=(1, 1))
    leg.set_draggable(state=False)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(2.0)
    
    return leg

def run_e_v_power_plot():
    """Main function to create the interactive plot."""
    # Load data
    data = load_data()
    
    # Create interpolation functions
    asu_change_func, reboiler_change_99, reboiler_change_90 = create_interpolation_functions(data)
    
    # Set up plot
    fig, ax = setup_plot()
    
    # Create storage for lines and fills
    plot_elements = {
        'allam_no_asu': {},
        'allam_w_asu': {},
        'icr_no_ccs': {},
        'icr_90_ccs': {},
        'icr_99_ccs': {}
    }
    
    # Plot all data initially
    # Allam cycle without ASU
    plot_elements['allam_no_asu']['line_05'], = ax.plot(POWER_OUTPUT, data['allam_05_no_asu'], 
            c=COLORS['allam_no_asu'], linewidth=3.0, zorder=1000)
    plot_elements['allam_no_asu']['line_025'], = ax.plot(POWER_OUTPUT, data['allam_025_no_asu'], 
            c=COLORS['allam_no_asu'], linewidth=3.0, zorder=1000)
    plot_elements['allam_no_asu']['fill'] = ax.fill_between(POWER_OUTPUT, data['allam_05_no_asu'], data['allam_025_no_asu'], 
                    where=(data['allam_025_no_asu'] > data['allam_05_no_asu']), 
                    color=COLORS['allam_no_asu'], alpha=0.8, interpolate=True)
    
    # Allam with ASU
    plot_elements['allam_w_asu']['line_05'], = ax.plot(POWER_OUTPUT, data['allam_05_w_asu'], 
                             c=COLORS['allam_w_asu'], linewidth=3.0, zorder=1000)
    plot_elements['allam_w_asu']['line_025'], = ax.plot(POWER_OUTPUT, data['allam_025_w_asu'], 
                              c=COLORS['allam_w_asu'], linewidth=3.0, zorder=1000)
    plot_elements['allam_w_asu']['fill'] = ax.fill_between(POWER_OUTPUT, data['allam_05_w_asu'], data['allam_025_w_asu'], 
                                     where=(data['allam_025_w_asu'] > data['allam_05_w_asu']), 
                                     color=COLORS['allam_w_asu'], alpha=0.8, interpolate=True, zorder=2500)
    
    # ICR without CCS
    plot_elements['icr_no_ccs']['line_05'], = ax.plot(POWER_OUTPUT, data['icr_05_no_ccs'], 
            c=COLORS['icr_no_ccs'], linewidth=3.0, zorder=1000)
    plot_elements['icr_no_ccs']['line_025'], = ax.plot(POWER_OUTPUT, data['icr_025_no_ccs'], 
            c=COLORS['icr_no_ccs'], linewidth=3.0, zorder=1000)
    plot_elements['icr_no_ccs']['fill'] = ax.fill_between(POWER_OUTPUT, data['icr_05_no_ccs'], data['icr_025_no_ccs'], 
                    where=(data['icr_025_no_ccs'] > data['icr_05_no_ccs']), 
                    color=COLORS['icr_no_ccs'], alpha=0.8, interpolate=True, zorder=3000)
    
    # ICR 90% CCS
    plot_elements['icr_90_ccs']['line_05'], = ax.plot(POWER_OUTPUT, data['icr_05_90_ccs'], 
                              c=COLORS['icr_90_ccs'], linewidth=3.0, zorder=1000)
    plot_elements['icr_90_ccs']['line_025'], = ax.plot(POWER_OUTPUT, data['icr_025_90_ccs'], 
                               c=COLORS['icr_90_ccs'], linewidth=3.0, zorder=1000)
    plot_elements['icr_90_ccs']['fill'] = ax.fill_between(POWER_OUTPUT, data['icr_05_90_ccs'], data['icr_025_90_ccs'], 
                                  where=(data['icr_025_90_ccs'] > data['icr_05_90_ccs']), 
                                  color=COLORS['icr_90_ccs'], alpha=0.8, interpolate=True, zorder=3000)
    
    # ICR 99% CCS
    plot_elements['icr_99_ccs']['line_05'], = ax.plot(POWER_OUTPUT, data['icr_05_99_ccs'], 
                              c=COLORS['icr_99_ccs'], linewidth=3.0, zorder=1000)
    plot_elements['icr_99_ccs']['line_025'], = ax.plot(POWER_OUTPUT, data['icr_025_99_ccs'], 
                               c=COLORS['icr_99_ccs'], linewidth=3.0, zorder=1000)
    plot_elements['icr_99_ccs']['fill'] = ax.fill_between(POWER_OUTPUT, data['icr_05_99_ccs'], data['icr_025_99_ccs'], 
                                  where=(data['icr_025_99_ccs'] > data['icr_05_99_ccs']), 
                                  color=COLORS['icr_99_ccs'], alpha=0.8, interpolate=True, zorder=3000)
    
    # Create visibility checkboxes - more compact
    cycle_keys = list(plot_elements.keys())
    visibility_checkboxes = {}
    checkbox_list = []
    for i, cycle_name in enumerate(CYCLE_NAMES):
        checkbox = widgets.Checkbox(
            value=True, description=cycle_name,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        visibility_checkboxes[cycle_name] = checkbox
        checkbox_list.append(checkbox)
    
    # Create sliders - more compact
    asu_slider = widgets.FloatSlider(
        value=INIT_ASU_WORK, min=0.3, max=2.0, step=0.01,
        description='Air separation work (MJ/kg O₂):',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='500px')
    )
    
    reb90_slider = widgets.FloatSlider(
        value=INIT_REBOILER_DUTY_90, min=2.5, max=5.0, step=0.01,
        description='Reboiler duty 90% capture (MJ/kg CO₂):',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='500px')
    )
    
    reb_slider = widgets.FloatSlider(
        value=INIT_REBOILER_DUTY, min=2.5, max=5.0, step=0.01,
        description='Reboiler duty 99% capture (MJ/kg CO₂):',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='500px')
    )
    
    # Reset button
    reset_button = widgets.Button(description='Reset')
    
    # Create legend
    legend = create_legend(ax)
    
    # Store references to current fills for updates
    current_fills = {
        'allam_w_asu': plot_elements['allam_w_asu']['fill'],
        'icr_90_ccs': plot_elements['icr_90_ccs']['fill'],
        'icr_99_ccs': plot_elements['icr_99_ccs']['fill']
    }
    
    def on_visibility_change(change):
        """Handle visibility checkbox changes"""
        cycle_name = change['owner'].description
        index = CYCLE_NAMES.index(cycle_name)
        cycle_key = cycle_keys[index]
        
        if change['new']:
            # Show the cycle
            for key in ['line_05', 'line_025']:
                if key in plot_elements[cycle_key]:
                    plot_elements[cycle_key][key].set_visible(True)
            if 'fill' in plot_elements[cycle_key] and plot_elements[cycle_key]['fill'] in ax.collections:
                plot_elements[cycle_key]['fill'].set_alpha(0.8)
        else:
            # Hide the cycle
            for key in ['line_05', 'line_025']:
                if key in plot_elements[cycle_key]:
                    plot_elements[cycle_key][key].set_visible(False)
            if 'fill' in plot_elements[cycle_key] and plot_elements[cycle_key]['fill'] in ax.collections:
                plot_elements[cycle_key]['fill'].set_alpha(0)
        
        fig.canvas.draw_idle()
    
    def update_asu(change):
        delta = asu_change_func(asu_slider.value)
        plot_elements['allam_w_asu']['line_05'].set_ydata(data['allam_05_w_asu'] + delta)
        plot_elements['allam_w_asu']['line_025'].set_ydata(data['allam_025_w_asu'] + delta)
        
        current_fills['allam_w_asu'].remove()
        current_fills['allam_w_asu'] = ax.fill_between(
            POWER_OUTPUT, 
            data['allam_05_w_asu'] + delta, 
            data['allam_025_w_asu'] + delta, 
            where=(data['allam_025_w_asu'] + delta > data['allam_05_w_asu'] + delta), 
            color=COLORS['allam_w_asu'], 
            alpha=0.8 if visibility_checkboxes[CYCLE_NAMES[1]].value else 0, 
            interpolate=True, zorder=2500
        )
        plot_elements['allam_w_asu']['fill'] = current_fills['allam_w_asu']
        fig.canvas.draw_idle()
    
    def update_reb90(change):
        delta = reboiler_change_90(reb90_slider.value)
        plot_elements['icr_90_ccs']['line_05'].set_ydata(data['icr_05_90_ccs'] + delta)
        plot_elements['icr_90_ccs']['line_025'].set_ydata(data['icr_025_90_ccs'] + delta)
        
        current_fills['icr_90_ccs'].remove()
        current_fills['icr_90_ccs'] = ax.fill_between(
            POWER_OUTPUT, 
            data['icr_05_90_ccs'] + delta, 
            data['icr_025_90_ccs'] + delta, 
            where=(data['icr_025_90_ccs'] + delta > data['icr_05_90_ccs'] + delta), 
            color=COLORS['icr_90_ccs'], 
            alpha=0.8 if visibility_checkboxes[CYCLE_NAMES[3]].value else 0, 
            interpolate=True, zorder=3000
        )
        plot_elements['icr_90_ccs']['fill'] = current_fills['icr_90_ccs']
        fig.canvas.draw_idle()
    
    def update_reb99(change):
        delta = reboiler_change_99(reb_slider.value)
        plot_elements['icr_99_ccs']['line_05'].set_ydata(data['icr_05_99_ccs'] + delta)
        plot_elements['icr_99_ccs']['line_025'].set_ydata(data['icr_025_99_ccs'] + delta)
        
        current_fills['icr_99_ccs'].remove()
        current_fills['icr_99_ccs'] = ax.fill_between(
            POWER_OUTPUT, 
            data['icr_05_99_ccs'] + delta, 
            data['icr_025_99_ccs'] + delta, 
            where=(data['icr_025_99_ccs'] + delta > data['icr_05_99_ccs'] + delta), 
            color=COLORS['icr_99_ccs'], 
            alpha=0.8 if visibility_checkboxes[CYCLE_NAMES[4]].value else 0, 
            interpolate=True, zorder=3000
        )
        plot_elements['icr_99_ccs']['fill'] = current_fills['icr_99_ccs']
        fig.canvas.draw_idle()
    
    def on_reset_click(b):
        asu_slider.value = INIT_ASU_WORK
        reb_slider.value = INIT_REBOILER_DUTY
        reb90_slider.value = INIT_REBOILER_DUTY_90
    
    # Connect callbacks
    for checkbox in visibility_checkboxes.values():
        checkbox.observe(on_visibility_change, names='value')
    
    asu_slider.observe(update_asu, names='value')
    reb_slider.observe(update_reb99, names='value')
    reb90_slider.observe(update_reb90, names='value')
    reset_button.on_click(on_reset_click)
    
    # Create the UI layout - compact and organized at bottom
    # Organize checkboxes in rows
    checkbox_rows = []
    for i in range(0, len(checkbox_list), 3):  # 3 checkboxes per row
        row = widgets.HBox(checkbox_list[i:i+3])
        checkbox_rows.append(row)
    
    visibility_section = widgets.VBox([
        widgets.HTML("<b>Show/Hide Cycles:</b>"),
        widgets.VBox(checkbox_rows)
    ], layout=widgets.Layout(width='400px'))
    
    # Group sliders
    sliders_section = widgets.VBox([
        asu_slider,
        reb90_slider,
        reb_slider,
        reset_button
    ])
    
    # Organize controls in horizontal layout
    controls_row = widgets.HBox([
        visibility_section,
        sliders_section
    ], layout=widgets.Layout(justify_content='space-between'))
    
    # Display plot first, then controls at bottom
    plt.show()
    display(controls_row)
    
    slider_data = {
        'asu_slider': asu_slider,
        'reb_slider': reb_slider,
        'reb90_slider': reb90_slider,
        'reset_button': reset_button
    }
    
    controls_data = {
        'visibility_checkboxes': visibility_checkboxes
    }
    
    return fig, ax, slider_data# , legend

##########################################
#Example run
# fig, ax, slider_data, legend = run_e_v_power_plot()
##########################################





