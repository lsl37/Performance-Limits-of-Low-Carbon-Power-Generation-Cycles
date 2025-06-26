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
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.patches import Polygon


###################### USER INPUT #############################################

#Change plot_settings_plain to plot_settings_latex if the latex environment
# is supported

## import program.plot_settings_latex
## print('Latex environment supported')



##import program.plot_settings_plain
# print('Latex environment not supported')
####################### END USER INPUT ########################################



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
    'Allam w ASU',
    r'ICR 0\% CO$_2$ Capture',
    r'ICR 90\% CO$_2$ Capture', 
    r'ICR 99\% CO$_2$ Capture'
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
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Adjust plot margins similar to other scripts
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.65, top=0.95)
    
    # Configure axes
    ax.set_xlabel('Power Output (MW)', fontsize=24)
    ax.set_ylabel('Efficiency (\%)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
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
    
    leg = plt.legend(handles=handles, fontsize=14, ncol=1, 
                    loc='lower center', bbox_to_anchor=(-2.0, 11.0))
    leg.set_draggable(state=True)
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
    
    # Create visibility checkboxes
    visibility_ax = plt.axes([0.68, 0.68, 0.3, 0.25])
    cycle_keys = list(plot_elements.keys())
    visibility_checks = CheckButtons(visibility_ax, CYCLE_NAMES, [True]*len(CYCLE_NAMES))
    visibility_ax.set_title('Show/Hide Cycles', fontsize=12, weight='bold')
    
    # Create horizontal sliders like in other scripts
    # ASU work slider
    ax_asu = plt.axes([0.3, 0.20, 0.3, 0.03])
    asu_slider = Slider(
        ax_asu,
        r'Air separation work (MJ/kg O$_{2}$)',
        0.3, 2.0, valinit=INIT_ASU_WORK, valstep=0.01,
        facecolor='#4292c6', edgecolor='k'
    )
    asu_slider.label.set_fontsize(14)
    asu_slider.valtext.set_fontsize(14)
    
    # ICR 90% reboiler duty slider
    ax_reb90 = plt.axes([0.3, 0.15, 0.3, 0.03])
    reb90_slider = Slider(
        ax_reb90,
        r"Reboiler duty (MJ/kg CO$_{2}$)",
        2.5, 5.0, valinit=INIT_REBOILER_DUTY_90, valstep=0.01,
        facecolor=COLORS['icr_90_ccs'], edgecolor='k'
    )
    reb90_slider.label.set_fontsize(14)
    reb90_slider.valtext.set_fontsize(14)
    
    # ICR 99% reboiler duty slider
    ax_reb = plt.axes([0.3, 0.10, 0.3, 0.03])
    reb_slider = Slider(
        ax_reb,
        r"Reboiler duty (MJ/kg CO$_{2}$)",
        2.5, 5.0, valinit=INIT_REBOILER_DUTY, valstep=0.01,
        facecolor=COLORS['icr_99_ccs'], edgecolor='k'
    )
    reb_slider.label.set_fontsize(14)
    reb_slider.valtext.set_fontsize(14)
    
    # Reset button
    reset_ax = fig.add_axes([0.92, 0.015, 0.06, 0.04])
    reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
    
    # Create legend
    legend = create_legend(ax)
    
    # Store references to current fills for updates
    current_fills = {
        'allam_w_asu': plot_elements['allam_w_asu']['fill'],
        'icr_90_ccs': plot_elements['icr_90_ccs']['fill'],
        'icr_99_ccs': plot_elements['icr_99_ccs']['fill']
    }
    
    def visibility_func(label):
        """Handle visibility checkbox changes"""
        index = CYCLE_NAMES.index(label)
        cycle_key = cycle_keys[index]
        
        if visibility_checks.get_status()[index]:
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
    
    def update_asu(val):
        delta = asu_change_func(asu_slider.val)
        plot_elements['allam_w_asu']['line_05'].set_ydata(data['allam_05_w_asu'] + delta)
        plot_elements['allam_w_asu']['line_025'].set_ydata(data['allam_025_w_asu'] + delta)
        
        current_fills['allam_w_asu'].remove()
        current_fills['allam_w_asu'] = ax.fill_between(
            POWER_OUTPUT, 
            data['allam_05_w_asu'] + delta, 
            data['allam_025_w_asu'] + delta, 
            where=(data['allam_025_w_asu'] + delta > data['allam_05_w_asu'] + delta), 
            color=COLORS['allam_w_asu'], 
            alpha=0.8 if visibility_checks.get_status()[1] else 0, 
            interpolate=True, zorder=2500
        )
        plot_elements['allam_w_asu']['fill'] = current_fills['allam_w_asu']
        fig.canvas.draw_idle()
    
    def update_reb90(val):
        delta = reboiler_change_90(reb90_slider.val)
        plot_elements['icr_90_ccs']['line_05'].set_ydata(data['icr_05_90_ccs'] + delta)
        plot_elements['icr_90_ccs']['line_025'].set_ydata(data['icr_025_90_ccs'] + delta)
        
        current_fills['icr_90_ccs'].remove()
        current_fills['icr_90_ccs'] = ax.fill_between(
            POWER_OUTPUT, 
            data['icr_05_90_ccs'] + delta, 
            data['icr_025_90_ccs'] + delta, 
            where=(data['icr_025_90_ccs'] + delta > data['icr_05_90_ccs'] + delta), 
            color=COLORS['icr_90_ccs'], 
            alpha=0.8 if visibility_checks.get_status()[3] else 0, 
            interpolate=True, zorder=3000
        )
        plot_elements['icr_90_ccs']['fill'] = current_fills['icr_90_ccs']
        fig.canvas.draw_idle()
    
    def update_reb99(val):
        delta = reboiler_change_99(reb_slider.val)
        plot_elements['icr_99_ccs']['line_05'].set_ydata(data['icr_05_99_ccs'] + delta)
        plot_elements['icr_99_ccs']['line_025'].set_ydata(data['icr_025_99_ccs'] + delta)
        
        current_fills['icr_99_ccs'].remove()
        current_fills['icr_99_ccs'] = ax.fill_between(
            POWER_OUTPUT, 
            data['icr_05_99_ccs'] + delta, 
            data['icr_025_99_ccs'] + delta, 
            where=(data['icr_025_99_ccs'] + delta > data['icr_05_99_ccs'] + delta), 
            color=COLORS['icr_99_ccs'], 
            alpha=0.8 if visibility_checks.get_status()[4] else 0, 
            interpolate=True, zorder=3000
        )
        plot_elements['icr_99_ccs']['fill'] = current_fills['icr_99_ccs']
        fig.canvas.draw_idle()
    
    def reset(event):
        asu_slider.reset()
        reb_slider.reset()
        reb90_slider.reset()
    
    # Connect callbacks
    visibility_checks.on_clicked(visibility_func)
    asu_slider.on_changed(update_asu)
    reb_slider.on_changed(update_reb99)
    reb90_slider.on_changed(update_reb90)
    reset_button.on_clicked(reset)
    
    plt.show()
    
    slider_data = {
        'asu_slider': asu_slider,
        'reb_slider': reb_slider,
        'reb90_slider': reb90_slider,
        'reset_button': reset_button
    }
    
    controls = {
        'visibility_checks': visibility_checks
    }
    
    return fig, ax, slider_data, legend

##########################################
#Example run
# fig, ax, slider_data, legend = run_e_v_power_plot()
##########################################





