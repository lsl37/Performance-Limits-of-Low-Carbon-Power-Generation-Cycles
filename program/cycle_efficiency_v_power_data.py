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
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon


###################### USER INPUT #############################################

#Change plot_settings_plain to plot_settings_latex if the latex environment
# is supported

## import plot_settings_latex
## print('Latex environment supported')



import plot_settings_plain
print('Latex environment not supported')
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
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Adjust plot margins
    plt.subplots_adjust(left=0.07, bottom=0.25, right=0.69, top=0.88)
    
    # Configure axes
    ax.set_xlabel('Power Output (MW)', fontsize=28)
    ax.set_ylabel('Efficiency (\%)', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='x', which='both', direction='in', top=True)
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    ax.minorticks_on()
    ax.grid(True, zorder=0, linestyle=':')
    
    # Add all spines
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    return fig, ax

def plot_base_data(ax, data):
    """Plot the base efficiency curves without interactive elements."""
    
    # Allam cycle without ASU
    ax.plot(POWER_OUTPUT, data['allam_05_no_asu'], 
            c=COLORS['allam_no_asu'], linewidth=3.0, zorder=1000)
    ax.plot(POWER_OUTPUT, data['allam_025_no_asu'], 
            c=COLORS['allam_no_asu'], linewidth=3.0, zorder=1000)
    ax.fill_between(POWER_OUTPUT, data['allam_05_no_asu'], data['allam_025_no_asu'], 
                    where=(data['allam_025_no_asu'] > data['allam_05_no_asu']), 
                    color=COLORS['allam_no_asu'], alpha=0.8, interpolate=True)
    
    # ICR without CCS
    ax.plot(POWER_OUTPUT, data['icr_05_no_ccs'], 
            c=COLORS['icr_no_ccs'], linewidth=3.0, zorder=1000)
    ax.plot(POWER_OUTPUT, data['icr_025_no_ccs'], 
            c=COLORS['icr_no_ccs'], linewidth=3.0, zorder=1000)
    ax.fill_between(POWER_OUTPUT, data['icr_05_no_ccs'], data['icr_025_no_ccs'], 
                    where=(data['icr_025_no_ccs'] > data['icr_05_no_ccs']), 
                    color=COLORS['icr_no_ccs'], alpha=0.8, interpolate=True, zorder=3000)

def create_interactive_elements(fig, ax, data, asu_change_func, reboiler_change_99, reboiler_change_90):
    """Create interactive plot elements with sliders."""
    
    # Plot interactive Allam with ASU lines
    line_allam_05, = ax.plot(POWER_OUTPUT, data['allam_05_w_asu'], 
                             c=COLORS['allam_w_asu'], linewidth=3.0, zorder=1000)
    line_allam_025, = ax.plot(POWER_OUTPUT, data['allam_025_w_asu'], 
                              c=COLORS['allam_w_asu'], linewidth=3.0, zorder=1000)
    fill_allam_asu = ax.fill_between(POWER_OUTPUT, data['allam_05_w_asu'], data['allam_025_w_asu'], 
                                     where=(data['allam_025_w_asu'] > data['allam_05_w_asu']), 
                                     color=COLORS['allam_w_asu'], alpha=0.8, interpolate=True, zorder=2500)
    
    # Plot interactive ICR lines
    line_icr_05_90, = ax.plot(POWER_OUTPUT, data['icr_05_90_ccs'], 
                              c=COLORS['icr_90_ccs'], linewidth=3.0, zorder=1000)
    line_icr_025_90, = ax.plot(POWER_OUTPUT, data['icr_025_90_ccs'], 
                               c=COLORS['icr_90_ccs'], linewidth=3.0, zorder=1000)
    fill_icr_90 = ax.fill_between(POWER_OUTPUT, data['icr_05_90_ccs'], data['icr_025_90_ccs'], 
                                  where=(data['icr_025_90_ccs'] > data['icr_05_90_ccs']), 
                                  color=COLORS['icr_90_ccs'], alpha=0.8, interpolate=True, zorder=3000)
    
    line_icr_05_99, = ax.plot(POWER_OUTPUT, data['icr_05_99_ccs'], 
                              c=COLORS['icr_99_ccs'], linewidth=3.0, zorder=1000)
    line_icr_025_99, = ax.plot(POWER_OUTPUT, data['icr_025_99_ccs'], 
                               c=COLORS['icr_99_ccs'], linewidth=3.0, zorder=1000)
    fill_icr_99 = ax.fill_between(POWER_OUTPUT, data['icr_05_99_ccs'], data['icr_025_99_ccs'], 
                                  where=(data['icr_025_99_ccs'] > data['icr_05_99_ccs']), 
                                  color=COLORS['icr_99_ccs'], alpha=0.8, interpolate=True, zorder=3000)
    
    # Create sliders
    slider_data = create_sliders(fig, data, line_allam_05, line_allam_025, fill_allam_asu,
                                line_icr_05_90, line_icr_025_90, fill_icr_90,
                                line_icr_05_99, line_icr_025_99, fill_icr_99,
                                asu_change_func, reboiler_change_99, reboiler_change_90, ax)
    
    return slider_data

def create_sliders(fig, data, line_allam_05, line_allam_025, fill_allam_asu,
                  line_icr_05_90, line_icr_025_90, fill_icr_90,
                  line_icr_05_99, line_icr_025_99, fill_icr_99,
                  asu_change_func, reboiler_change_99, reboiler_change_90, ax):
    """Create parameter adjustment sliders."""
    
    # ASU work slider
    ax_asu = fig.add_axes([0.72, 0.25, 0.0225, 0.63])
    asu_slider = Slider(
        ax=ax_asu,
        label=r"Air separation work (MJ/kg O$_{2}$)",
        valmin=0.3,
        valmax=2.0,
        valinit=INIT_ASU_WORK,
        orientation="vertical",
        edgecolor='k',
        facecolor='#4292c6'
    )
    asu_slider.label.set_rotation(90)
    asu_slider.label.set_ha("right")
    asu_slider.label.set_va("center")
    asu_slider.label.set_position((2.1, 0.5))
    
    # Reboiler duty sliders
    ax_reb = fig.add_axes([0.92, 0.25, 0.0225, 0.63])
    reb_slider = Slider(
        ax=ax_reb,
        label=r"Reboiler duty (MJ/kg CO$_{2}$)",
        valmin=2.5,
        valmax=5.0,
        valinit=INIT_REBOILER_DUTY,
        orientation="vertical",
        edgecolor='k',
        facecolor=COLORS['icr_99_ccs']
    )
    reb_slider.label.set_rotation(90)
    reb_slider.label.set_ha("right")
    reb_slider.label.set_va("center")
    reb_slider.label.set_position((2.1, 0.5))
    
    ax_reb90 = fig.add_axes([0.82, 0.25, 0.0225, 0.63])
    reb90_slider = Slider(
        ax=ax_reb90,
        label=r"Reboiler duty (MJ/kg CO$_{2}$)",
        valmin=2.5,
        valmax=5.0,
        valinit=INIT_REBOILER_DUTY_90,
        orientation="vertical",
        edgecolor='k',
        facecolor=COLORS['icr_90_ccs']
    )
    reb90_slider.label.set_rotation(90)
    reb90_slider.label.set_ha("right")
    reb90_slider.label.set_va("center")
    reb90_slider.label.set_position((2.1, 0.5))
    
    # Create update functions with proper closure
    def create_update_functions():
        # Store reference to current fill objects
        current_fills = {
            'allam_asu': fill_allam_asu,
            'icr_90': fill_icr_90,
            'icr_99': fill_icr_99
        }
        
        def update_asu(val):
            delta = asu_change_func(asu_slider.val)
            line_allam_05.set_ydata(data['allam_05_w_asu'] + delta)
            line_allam_025.set_ydata(data['allam_025_w_asu'] + delta)
            
            current_fills['allam_asu'].remove()
            current_fills['allam_asu'] = ax.fill_between(
                POWER_OUTPUT, 
                data['allam_05_w_asu'] + delta, 
                data['allam_025_w_asu'] + delta, 
                where=(data['allam_025_w_asu'] + delta > data['allam_05_w_asu'] + delta), 
                color=COLORS['allam_w_asu'], alpha=0.8, interpolate=True, zorder=2500
            )
            fig.canvas.draw_idle()
        
        def update_reb99(val):
            delta = reboiler_change_99(reb_slider.val)
            line_icr_05_99.set_ydata(data['icr_05_99_ccs'] + delta)
            line_icr_025_99.set_ydata(data['icr_025_99_ccs'] + delta)
            
            current_fills['icr_99'].remove()
            current_fills['icr_99'] = ax.fill_between(
                POWER_OUTPUT, 
                data['icr_05_99_ccs'] + delta, 
                data['icr_025_99_ccs'] + delta, 
                where=(data['icr_025_99_ccs'] + delta > data['icr_05_99_ccs'] + delta), 
                color=COLORS['icr_99_ccs'], alpha=0.8, interpolate=True, zorder=3000
            )
            fig.canvas.draw_idle()
        
        def update_reb90(val):
            delta = reboiler_change_90(reb90_slider.val)
            line_icr_05_90.set_ydata(data['icr_05_90_ccs'] + delta)
            line_icr_025_90.set_ydata(data['icr_025_90_ccs'] + delta)
            
            current_fills['icr_90'].remove()
            current_fills['icr_90'] = ax.fill_between(
                POWER_OUTPUT, 
                data['icr_05_90_ccs'] + delta, 
                data['icr_025_90_ccs'] + delta, 
                where=(data['icr_025_90_ccs'] + delta > data['icr_05_90_ccs'] + delta), 
                color=COLORS['icr_90_ccs'], alpha=0.8, interpolate=True, zorder=3000
            )
            fig.canvas.draw_idle()
        
        return update_asu, update_reb99, update_reb90
    
    update_asu, update_reb99, update_reb90 = create_update_functions()
    
    # Connect sliders to update functions
    asu_slider.on_changed(update_asu)
    reb_slider.on_changed(update_reb99)
    reb90_slider.on_changed(update_reb90)
    
    # Reset button
    reset_ax = fig.add_axes([0.88, 0.015, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
    
    def reset(event):
        asu_slider.reset()
        reb_slider.reset()
        reb90_slider.reset()
    
    reset_button.on_clicked(reset)
    
    return {
        'asu_slider': asu_slider,
        'reb_slider': reb_slider,
        'reb90_slider': reb90_slider,
        'reset_button': reset_button
    }

def create_legend(ax):
    """Create a custom legend for the plot."""
    handles = []
    labels = [
        'Allam',
        'Allam w ASU',
        r'ICR 0\% CO$_2$ Capture',
        r'ICR 90\% CO$_2$ Capture', 
        r'ICR 99\% CO$_2$ Capture'
    ]
    
    for i, (color, label) in enumerate(zip(COLORS.values(), labels)):
        handles.append(Polygon([(0,0),(10,0),(0,-10)], facecolor=color,
                              label=label, edgecolor='black', linewidth=1.5))
    
    position =  position={'bbox_to_anchor': (0.5, -0.2), 'loc': 'upper center'}
    legend = ax.legend(handles=handles, fontsize=18, ncol=3, **position)
    legend.set_draggable(state=True)
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_linewidth(2.0)
    
    return legend

# def main():
    # """Main function to create the interactive plot."""
    # Load data
    
def run_e_v_power_plot():   
    data = load_data()
    
    # Create interpolation functions
    asu_change_func, reboiler_change_99, reboiler_change_90 = create_interpolation_functions(data)
    
    # Set up plot
    fig, ax = setup_plot()
    
    # Plot base data
    plot_base_data(ax, data)
    
    # Create interactive elements
    slider_data = create_interactive_elements(
        fig, ax, data, asu_change_func, reboiler_change_99, reboiler_change_90
    )
    
    # Add legend (optional - uncomment if desired)
    legend = create_legend(ax)
    
    plt.show()
    
    return fig, ax, slider_data, legend 



##########################################
#Example run
# fig, ax, slider_data, legend = run_plot()
##########################################





