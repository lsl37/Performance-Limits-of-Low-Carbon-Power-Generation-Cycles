#!/usr/bin/env python3
"""
Performance Limits of Low Carbon Power Generation Cycles
DOI: 

Interactive plot:
    -Figure 4
    -GELM2500 integrated gas-turbine cycles

Author: lsl37
Created: June, 2025
Modified: For ipywidgets compatibility 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import LogLocator, ScalarFormatter
import ipywidgets as widgets
from IPython.display import display
import matplotlib.patches as mpatches

def plot_polygon(ax, corners, facecolor, edgecolor='black', linewidth=1.5, alpha=0.9, zorder=1000, hatch=None):
    """Helper function to plot polygons"""
    polygon = Polygon(corners, facecolor=facecolor, edgecolor=edgecolor, 
                     linewidth=linewidth, alpha=alpha, zorder=zorder, hatch=hatch)
    return ax.add_patch(polygon)

def interpolate_uncertainty_points(center_points, boundary_points, uncertainty_fractionx_heatttransfer, 
                                  uncertainty_fractiony_turb, uncertainty_fractiony_comp, change_combustor, change_cooling_efficiency, data):
    """
    Interpolate between center points and boundary points based on uncertainty fractions.
    """
    interpolated_points = []
    
    # Compressor efficiency deltas ±2%
    comp_delta_positive = data[0][1]
    comp_delta_negative = data[1][1]
    
    #Combustor efficiency deltas (1-5 %)
    combustor_positive = data[0][0][0]
    combustor_negative = data[1][0][0]
    combustor_positive_size = data[0][0][1]
    combustor_negative_size = data[1][0][1]
    
    #Cooling efficiency deltas (0.054-0.094)
    cooling_positive = data[0][-1][0]
    cooling_negative = data[1][-1][0]
    cooling_positive_size = data[0][-1][1]
    cooling_negative_size = data[1][-1][1]
    
    for i, boundary_point in enumerate(boundary_points):
        
        if i == 2 or i == 3 or i == 4:
            nearest_center = center_points[0]
        else:
            nearest_center = center_points[1]
        
        # Interpolate x coordinate (heat transfer)
        interpolated_x = nearest_center[0] + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        
        # Interpolate y coordinate (turbine + compressor effects)
        turbine_delta_y = uncertainty_fractiony_turb * (boundary_point[1] - nearest_center[1])
        
        # Determine compressor contribution
        if boundary_point[1] > nearest_center[1]:
            comp_contribution = comp_delta_positive * uncertainty_fractiony_comp
        else:
            comp_contribution = comp_delta_negative * uncertainty_fractiony_comp
        
        interpolated_y = nearest_center[1] + turbine_delta_y + comp_contribution
        
        if change_combustor > 0:
            interpolated_y += abs(change_combustor)*combustor_negative
            interpolated_x = nearest_center[0]*(abs(change_combustor)*(combustor_negative_size-1.0) +1.0) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        elif change_combustor < 0:
            interpolated_y += abs(change_combustor)*combustor_positive
            interpolated_x = nearest_center[0]*(abs(change_combustor)*(combustor_positive_size-1.0) +1.0) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        
        if change_cooling_efficiency > 0:
            interpolated_y += abs(change_cooling_efficiency)*cooling_negative
            interpolated_x = nearest_center[0]*(abs(change_cooling_efficiency)*(cooling_negative_size-1.0) +1.0) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        elif change_cooling_efficiency < 0:
            interpolated_y += abs(change_cooling_efficiency)*cooling_positive
            interpolated_x = nearest_center[0]*(abs(change_cooling_efficiency)*(cooling_positive_size-1.0) +1.0) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        
        interpolated_points.append([interpolated_x, interpolated_y])
    
    return np.array(interpolated_points)

# Define all data as module-level constants
###############################################################################

# Original volume
V_org = 50.611
V_org_H2 = 48.90

# Simple cycle data (baseline)
simple_cycle_eff_CH4 = [1.0, 37.7]
simple_cycle_eff = [(V_org_H2/V_org),38.77]

# Define all cycle data in a structured way
cycle_data = {
    'CA': {
        'points': np.array([
                           [ 1.57174406, 42.03      ],
                           [ 2.12721506, 42.03      ],
                           [ 2.2320476 , 42.53      ],
                           [ 2.2320476 , 45.78      ],
                           [ 1.66708335, 45.78      ],
                           [ 1.57174406, 45.28      ]]),
        'center': np.array([[1.9777834858034815, 44.17],
                            [1.8745253008239318, 43.67]
                           ]),
        'color': '#ffffd4',
        'zorder': 1000
    },
    'IC': {
        'points': np.array([
                           [ 2.4846856 , 42.5       ],
                           [ 3.68972897, 42.5       ],
                           [ 3.86520487, 44.45      ],
                           [ 3.86520487, 47.54      ],
                           [ 2.59051585, 47.54      ],
                           [ 2.4846856 , 45.72      ]]),
        'center': np.array([[ 3.2429629922348893 , 46.01      ],
                            [3.1030724546047304, 44.13      ]
                           ]),
        'color': '#fee391',
        'zorder': 1000
    },
    'REC': {
        'points': np.array([[ 1.04527747, 46.68      ],
                           [ 1.63626008, 46.68      ],
                           [ 3.9105449 , 47.95      ],
                           [ 3.9105449 , 51.32      ],
                           [ 1.9058753 , 51.32      ],
                           [ 1.04527747, 50.01      ]]),
        'center': np.array([[2.703048744344115, 49.644],   
                           [1.2726087214241963, 48.36]]),
        'color': '#fe9929',
        'zorder': 1000
    },
    'ICR': {
        'points': np.array([
                           [ 2.2292857 , 52.42      ],
                           [ 3.30688227, 52.42      ],
                           [ 6.25713807, 54.5       ],
                           [ 6.25713807, 57.8       ],
                           [ 2.96144179, 57.8       ],
                           [ 2.2292857 , 55.7       ]]),
        'center': np.array([[4.401612297721839, 56.04],
                            [2.8695342909644146, 54.0]
                           ]),
        'color': '#cc4c02',
        'zorder': 3000
    },
    'STIG': {
        'points': np.array([[ 2.02437217, 47.87      ],
                           [ 3.14077303, 47.87      ],
                           [ 3.26091444, 48.37      ],
                           [ 3.26091444, 51.68      ],
                           [ 2.08593054, 51.68      ],
                           [ 2.02437217, 51.18      ]]),
        'center': np.array([[2.7026, 50.0],
                           [2.61075, 49.5]]),
        'color': '#deebf7',
        'zorder': 2000
    },
    'WAC-HAT': {
        'points': np.array([[ 0.87289093, 53.3       ],
                           [ 1.00456463, 53.3       ],
                           [ 3.31894623, 54.37      ],
                           [ 3.31894623, 57.65      ],
                           [ 1.64697853, 57.65      ],
                           [ 0.87289093, 56.65      ]]),
        'center': np.array([[2.515461065776215, 56.02],
                           [0.9545355752701982, 55.33]]),
        'color': '#9ecae1',
        'zorder': 1000
    },
    'HAT': {
        'points': np.array([[ 1.88100302, 54.87      ],
                           [ 3.20851994, 54.87      ],
                           [ 4.46179324, 55.91      ],
                           [ 4.46179324, 59.15      ],
                           [ 2.61737515, 59.15      ],
                           [ 1.88100302, 58.1       ]]),
        'center': np.array([[3.5371658335144534, 57.54],
                           [2.5684139811503433, 56.5]]),
        'color': '#2171b5',
        'zorder': 3000
    },
    'RWI': {
        'points': np.array([[ 2.20055708, 54.82      ],
                           [ 5.26244133, 54.82      ],
                           [ 7.8286943 , 57.02      ],
                           [ 7.8286943 , 60.33      ],
                           [ 2.78172131, 60.33      ],
                           [ 2.20055708, 58.15      ]]),
        'center': np.array([[ 4.48125, 58.69],
                           [3.5, 56.49]]),
        'color': '#081d58',
        'zorder': 1000
    },
    'CC 1P': {
        'points': np.array([[ 5.63520147, 51.94      ],
                           [ 9.20678441, 51.94      ],
                           [ 9.68902257, 53.02      ],
                           [ 9.68902257, 53.86      ],
                           [ 5.92034917, 53.86      ],
                           [ 5.63520147, 52.84      ]]),
        'center': np.array([[ 7.9398, 53.46      ],
                           [ 7.5495, 52.89      ]]),
        'color': '#e87a87',
        'zorder': 2000
    },
    'CC 3P RH': {
        'points': np.array([[ 7.98950321, 55.95      ],
                           [12.68911142, 55.95      ],
                           [13.30861272, 56.57      ],
                           [13.30861272, 58.06      ],
                           [ 8.15715343, 58.06      ],
                           [ 7.98950321, 57.47      ]]),
        'center': np.array([[ 10.79313, 57.32      ],
                           [ 10.397, 56.71      ]]),
        'color': '#d53e4f',
        'zorder': 2000
    },
    'ORC': {
        'points': np.array([[ 5.32009471, 46.89      ],
                           [ 8.95458366, 46.89      ],
                           [ 9.74845173, 47.99      ],
                           [ 9.74845173, 50.337     ],
                           [ 5.61116541, 50.337     ],
                           [ 5.32009471, 49.27      ]]),
        'center': np.array([[  7.8017, 49.22      ],
                           [ 7.25077045, 48.13      ]]),
        'color': '#004529',
        'zorder': 1000
    },
    'sCO2': {
        'points': np.array([[ 4.14991905, 44.14      ],
                           [ 5.34218153, 44.14      ],
                           [ 7.27759461, 45.11      ],
                           [ 7.27759461, 49.08      ],
                           [ 4.51802893, 49.08      ],
                           [ 4.14991905, 48.05      ]]),
        'center': np.array([[  6.0192, 47.09      ],
                           [ 4.8425, 46.1      ]]),
        'color': '#31a354',
        'zorder': 1000
    }
}

# Data for sensitivity analysis
data = {
    'CA': ([[0.33, 0.9925], 0.97, 1.46, [1.96, 0.957]], [[-0.34, 1.008 ], -1.05, -1.46, [-1.9,  1.045]]),
    'IC': ([[0.31,0.99], 0.90, 0.0, [1.37,0.97]], [[-0.33,1.01], -0.98, -0.0, [-1.17,1.03]]),
    'REC': ([[0.58,0.99], 0.69, 0.94, [1.06,0.98]], [[-0.59,1.01], -0.74, -0.94, [-0.73,1.015]]),
    'ICR': ([[0.47, 0.991],0.59,1.09,[1.24, 0.978]], [[-0.49, 1.009],-0.63,-1.1,[-1.47, 1.028]]),
    'STIG': ([[0.44, 0.991],0.79,0.0,[2.266, 0.96]], [[-0.49, 1.01],-0.63,-0.0,[-1.57, 1.032]]),
    'WAC-HAT': ([[0.50, 0.991],0.60,1.16,[1.35, 0.976]], [[-0.45, 1.008],-0.63,-1.16,[-1.11, 1.02]]),
    'HAT': ([[0.46, 0.992],0.67,1.23,[2.19, 0.96]], [[-0.46, 1.008],-0.74,-1.24,[-1.48,1.026]]),
    'RWI': ([[0.5, 0.991],0.56,1.02,[2.10, 0.965]], [[-0.52, 1.009],-0.6,-1.03,[-2.28, 1.04]]),
    'CC 1P': ([[0.17, 0.997],0.15,0.0,[2.93, 0.948]], [[-0.18, 1.003],-0.17,0.0,[-2.65, 1.052]]),
    'CC 3P RH': ([[0.31,0.995 ],0.28,0.0,[1.85, 0.969]], [[-0.31, 1.005],-0.30,0.0,[-1.58, 1.028]]),
    'ORC': ([[0.36, 0.993],0.56,0.0,[1.67, 0.967]], [[-0.38, 1.008],-0.61,0.0,[-1.63,1.034]]),
    'sCO2': ([[0.53, 0.989],1.22,0.0,[1.02, 0.979]], [[-0.53, 1.011],-1.28,0.0,[-1.34, 1.029]])
}

def run_hydrogen_cycles_plot():
    """Main function to create the interactive plot."""
    
    # Create the plot with smaller size to fit in cell
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.7)

    # Plot baseline
    ax.scatter(simple_cycle_eff[0],simple_cycle_eff[1], color='black', marker='o', s=100,zorder=1000)
    ax.scatter(simple_cycle_eff_CH4[0],simple_cycle_eff_CH4[1], color='gray', marker='o', s=100,zorder=1000)

    # Initialize cycle polygons dictionary
    cycle_polygons = {}
    cycle_names = list(cycle_data.keys())

    # Plot all cycles initially
    for cycle_name, cycle_info in cycle_data.items():
        polygon = plot_polygon(ax, cycle_info['points'], 
                              facecolor=cycle_info['color'], 
                              zorder=cycle_info['zorder'])
        cycle_polygons[cycle_name] = polygon

    # Labels and formatting
    plt.text(1.04,38.4,r'GELM2500 H$_{2}$',fontsize=16)
    plt.text(1.06,36.9,r'GELM2500 CH$_{4}$',fontsize=16)
    ax.set_xlabel(r'Size (-)', fontsize=20)
    ax.set_ylabel('Efficiency (%)', fontsize=20)
    ax.set_xscale('log')
    ax.set_ylim(35.0, 65.0)
    ax.set_xlim(0.7, 20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='x', which='both', direction='in', top=True)
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    plt.minorticks_on()
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, zorder=0, linestyle=':')

    # Add legend
    handles = []
    for cycle_name, cycle_info in cycle_data.items():
        handles.append(Polygon([(0,0),(10,0),(0,-10)], 
                              facecolor=cycle_info['color'],
                              label=cycle_name, 
                              edgecolor='black',
                              linewidth=1.5))
    leg = plt.legend(handles=handles, fontsize=12, ncol=1, 
                    loc='upper left', bbox_to_anchor=(1, 1))
    leg.set_draggable(state=False)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(2.0)

    # Store original points for each cycle
    original_points = {name: info['points'].copy() for name, info in cycle_data.items()}

    # Global variable to track update mode
    current_update_mode = 'All'

    # Create ipywidgets - more compact
    heat_transfer_slider = widgets.FloatSlider(
        value=20, min=0, max=20, step=1,
        description='Heat Transfer Uncertainty (%):',
        style={'description_width': '300px'},
        layout=widgets.Layout(width='600px')
    )
    
    turb_slider = widgets.FloatSlider(
        value=2, min=0, max=2, step=0.1,
        description='Turb. ηp Uncertainty (%):',
        style={'description_width': '300px'},
        layout=widgets.Layout(width='600px')
    )
    
    comp_slider = widgets.FloatSlider(
        value=0, min=0, max=2, step=0.1,
        description='Comp. ηp Uncertainty (%):',
        style={'description_width': '300px'},
        layout=widgets.Layout(width='600px')
    )
    
    comb_slider = widgets.FloatSlider(
        value=3.0, min=1.0, max=5.0, step=0.1,
        description='Combustor Pressure Loss (%):',
        style={'description_width': '300px'},
        layout=widgets.Layout(width='600px')
    )
    
    cooling_slider = widgets.FloatSlider(
        value=0.074, min=0.054, max=0.094, step=0.001,
        description='Cooling parameter b (-):',
        style={'description_width': '300px'},
        layout=widgets.Layout(width='600px')
    )

    # Create visibility checkboxes - more compact
    visibility_checkboxes = {}
    checkbox_list = []
    for cycle_name in cycle_names:
        checkbox = widgets.Checkbox(
            value=True, description=cycle_name,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='120px')
        )
        visibility_checkboxes[cycle_name] = checkbox
        checkbox_list.append(checkbox)

    # Create horizontal update mode checkboxes (like radio buttons)
    update_options = ['All'] + cycle_names
    update_checkboxes = {}
    update_checkbox_list = []
    
    for option in update_options:
        checkbox = widgets.Checkbox(
            value=(option == 'All'),  # Only 'All' checked initially
            description=option,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='120px')
        )
        update_checkboxes[option] = checkbox
        update_checkbox_list.append(checkbox)

    # Reset button
    reset_button = widgets.Button(description='Reset')

    def get_interpolated_points(cycle_name):
        """Get interpolated points for a specific cycle based on current slider values"""
        heat_transfer_uncertainty = heat_transfer_slider.value
        turbine_uncertainty = turb_slider.value
        compressor_uncertainty = comp_slider.value
        combustor_pressure_loss = comb_slider.value    
        cooling_efficiency = cooling_slider.value
        
        uncertainty_fractionx = heat_transfer_uncertainty / 20.0
        uncertainty_fractiony_turb = turbine_uncertainty / 2.0
        uncertainty_fractiony_comp = compressor_uncertainty / 2.0
        
        nom_combustor_loss = 3.0
        change_combustor = (combustor_pressure_loss - nom_combustor_loss)/2.0
        
        nom_cooling_efficiency = 0.074
        change_cooling_efficiency = (cooling_efficiency - nom_cooling_efficiency)/0.02
        
        return interpolate_uncertainty_points(
            cycle_data[cycle_name]['center'], 
            cycle_data[cycle_name]['points'], 
            uncertainty_fractionx,
            uncertainty_fractiony_turb,
            uncertainty_fractiony_comp,
            change_combustor,
            change_cooling_efficiency,
            data[cycle_name]
        )

    def update_plot():
        """Update the plot based on current widget values"""
        nonlocal cycle_polygons, current_update_mode
        
        # Determine which cycles to update
        if current_update_mode == 'All':
            cycles_to_update = cycle_names
        else:
            cycles_to_update = [current_update_mode]
        
        # Update selected cycles
        for cycle_name in cycles_to_update:
            # Only update if cycle is visible
            if visibility_checkboxes[cycle_name].value:
                new_points = get_interpolated_points(cycle_name)
                
                # Remove old polygon
                if cycle_name in cycle_polygons and cycle_polygons[cycle_name] in ax.patches:
                    cycle_polygons[cycle_name].remove()
                
                # Create new polygon
                polygon = plot_polygon(ax, new_points, 
                                     facecolor=cycle_data[cycle_name]['color'], 
                                     zorder=cycle_data[cycle_name]['zorder'])
                cycle_polygons[cycle_name] = polygon
        
        fig.canvas.draw_idle()

    def on_visibility_change(change):
        """Handle visibility checkbox changes"""
        nonlocal cycle_polygons
        cycle_name = change['owner'].description
        
        if change['new']:
            # Show the cycle
            if cycle_name not in cycle_polygons or cycle_polygons[cycle_name] not in ax.patches:
                # Get current slider values to determine polygon shape
                if current_update_mode == 'All' or current_update_mode == cycle_name:
                    points = get_interpolated_points(cycle_name)
                else:
                    points = original_points[cycle_name]
                
                polygon = plot_polygon(ax, points, 
                                     facecolor=cycle_data[cycle_name]['color'], 
                                     zorder=cycle_data[cycle_name]['zorder'])
                cycle_polygons[cycle_name] = polygon
        else:
            # Hide the cycle
            if cycle_name in cycle_polygons and cycle_polygons[cycle_name] in ax.patches:
                cycle_polygons[cycle_name].remove()
        
        fig.canvas.draw_idle()

    def on_update_mode_change(change):
        """Handle update mode checkbox changes (radio button behavior)"""
        nonlocal current_update_mode
        
        if change['new']:  # If checkbox was checked
            # Uncheck all others (radio button behavior)
            for option, cb in update_checkboxes.items():
                if cb != change['owner']:
                    cb.value = False
            current_update_mode = change['owner'].description

    def on_reset_click(b):
        """Reset all sliders and update all cycles"""
        heat_transfer_slider.value = 20
        turb_slider.value = 2
        comp_slider.value = 0
        comb_slider.value = 3.0
        cooling_slider.value = 0.074
        
        # Reset update mode to 'All'
        for option, cb in update_checkboxes.items():
            cb.value = (option == 'All')
        
        # Reset all cycles to original if in individual mode
        if current_update_mode != 'All':
            for cycle_name in cycle_names:
                if visibility_checkboxes[cycle_name].value:
                    if cycle_name in cycle_polygons and cycle_polygons[cycle_name] in ax.patches:
                        cycle_polygons[cycle_name].remove()
                    
                    polygon = plot_polygon(ax, original_points[cycle_name], 
                                         facecolor=cycle_data[cycle_name]['color'], 
                                         zorder=cycle_data[cycle_name]['zorder'])
                    cycle_polygons[cycle_name] = polygon
        
        fig.canvas.draw_idle()

    # Connect callbacks
    for checkbox in visibility_checkboxes.values():
        checkbox.observe(on_visibility_change, names='value')
    
    # Connect update mode callbacks
    for checkbox in update_checkboxes.values():
        checkbox.observe(on_update_mode_change, names='value')
    
    reset_button.on_click(on_reset_click)
    
    # Observe slider changes
    heat_transfer_slider.observe(lambda change: update_plot(), names='value')
    turb_slider.observe(lambda change: update_plot(), names='value')
    comp_slider.observe(lambda change: update_plot(), names='value')
    comb_slider.observe(lambda change: update_plot(), names='value')
    cooling_slider.observe(lambda change: update_plot(), names='value')

    # Create the UI layout - compact and organized at bottom
    # Organize visibility checkboxes in a compact grid
    checkbox_rows = []
    for i in range(0, len(checkbox_list), 4):  # 4 checkboxes per row
        row = widgets.HBox(checkbox_list[i:i+4])
        checkbox_rows.append(row)
    
    # Organize update checkboxes in rows  
    update_rows = []
    for i in range(0, len(update_checkbox_list), 4):  # 4 checkboxes per row
        row = widgets.HBox(update_checkbox_list[i:i+4])
        update_rows.append(row)
    
    # Create sections
    visibility_section = widgets.VBox([
        widgets.HTML("<b>Show/Hide Cycles:</b>"),
        widgets.VBox(checkbox_rows)
    ], layout=widgets.Layout(width='360px'))
    
    # Update section with proper checkboxes
    update_section = widgets.VBox([
        widgets.HTML("<b>Update Mode:</b>"),
        widgets.VBox(update_rows)
    ], layout=widgets.Layout(width='360px'))
    
    # Stack visibility and update sections vertically
    selection_section = widgets.VBox([
        visibility_section,
        update_section
    ])
    
    # Group sliders in columns
    sliders_col1 = widgets.VBox([
        heat_transfer_slider,
        turb_slider,
        comp_slider,
        comb_slider,
        cooling_slider,
        reset_button
    ])
    
    # sliders_col2 = widgets.VBox([
    #     comb_slider,
    #     cooling_slider,
    #     reset_button
    # ])
    
    # Organize controls in horizontal layout
    controls_row = widgets.HBox([
        selection_section,
        sliders_col1
    ], layout=widgets.Layout(justify_content='space-between'))
    
    # Display plot first, then controls at bottom
    plt.show()
    display(controls_row)
    
    # Return the interactive elements
    slider_data = {
        'heat_transfer_slider': heat_transfer_slider,
        'turb_slider': turb_slider,
        'comp_slider': comp_slider,
        'comb_slider': comb_slider,
        'cooling_slider': cooling_slider,
        'reset_button': reset_button
    }
    
    controls_data = {
        'visibility_checkboxes': visibility_checkboxes,
        'update_checkboxes': update_checkboxes  
    }
    
    return fig, ax, slider_data, controls_data

# Test the function
if __name__ == "__main__":
    fig, ax, slider_data, controls = run_hydrogen_cycles_plot()
##########################################
# Example run
# fig, ax, slider_data, controls, legend = run_hydrogen_cycles_plot()
##########################################
