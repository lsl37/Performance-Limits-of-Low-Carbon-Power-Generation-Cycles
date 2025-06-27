#!/usr/bin/env python3
"""
Performance Limits of Low Carbon Power Generation Cycles
DOI: 

Interactive plot:
    -Figure 6/17
    -Uses Capex and Opex as horizontal and vertical axes

Author: lsl37
Created: June, 2025
Modified: For ipywidgets compatibility
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter
import ipywidgets as widgets
from IPython.display import display
from scipy.interpolate import interp1d

def plot_polygon(ax, corners, facecolor, edgecolor='black', linewidth=1.5, alpha=0.9, zorder=1000, hatch=None):
    """Helper function to plot polygons"""
    polygon = Polygon(corners, facecolor=facecolor, edgecolor=edgecolor, 
                     linewidth=linewidth, alpha=alpha, zorder=zorder, hatch=hatch)
    return ax.add_patch(polygon)

def update_axis_limits(ax, cycle_polygons, padding=0.1):
    """
    Update axis limits to ensure all visible polygons are within view.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to update
    cycle_polygons : dict
        Dictionary of cycle name -> polygon patch
    padding : float
        Fraction of range to add as padding (default 0.1 = 10%)
    """
    # Get all visible polygon vertices
    all_x = []
    all_y = []
    
    for cycle_name, polygon in cycle_polygons.items():
        if polygon in ax.patches:  # Check if polygon is visible
            vertices = polygon.get_xy()
            all_x.extend(vertices[:, 0])
            all_y.extend(vertices[:, 1])
    
    if all_x and all_y:  # Only update if there are visible polygons
        # Calculate ranges
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_padding = x_range * padding
        y_padding = y_range * padding
        
        # Get current limits
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Calculate new limits - only expand, never contract
        new_xlim = (
            min(current_xlim[0], x_min - x_padding),
            max(current_xlim[1], x_max + x_padding)
        )
        new_ylim = (
            min(current_ylim[0], y_min - y_padding),
            max(current_ylim[1], y_max + y_padding)
        )
        
        # Update limits if they changed
        if new_xlim != current_xlim:
            ax.set_xlim(new_xlim)
        if new_ylim != current_ylim:
            ax.set_ylim(new_ylim)

def interpolate_uncertainty_points(center_points, boundary_points, uncertainty_fractionx_heatttransfer, 
                                  uncertainty_fractiony_turb, uncertainty_fractiony_comp, uncertainty_fractiony_fuel, uncertainty_fractiony_carbon, data,change_ASU, change_ICR_reboiler, change_CCGT_reboiler, cycle_name):
    """
    Interpolate between center points and boundary points based on uncertainty fractions.
    """
    interpolated_points = []
    
    # Compressor efficiency deltas ±2%
    comp_delta_positive = data[0][0]
    comp_delta_negative = data[1][0]
    
    fuel_delta_positive = data[0][1]
    fuel_delta_negative = data[1][1]
    
    carbon_delta_positive = data[0][2]
    carbon_delta_negative = data[1][2]
    
    for i, boundary_point in enumerate(boundary_points):
        
        if cycle_name != 'CCGT CCS':
            if i == 2 or i == 3 or i == 4:
                nearest_center = center_points[0]
            else:
                nearest_center = center_points[1]
        else:
            if i == 0 or i == 1 or i == 2:
                nearest_center = center_points[1]
            else:
                nearest_center = center_points[0]
        
        # Interpolate x coordinate (heat transfer)
        interpolated_x = nearest_center[0] + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        
        # Interpolate y coordinate (turbine + compressor effects)
        turbine_delta_y = uncertainty_fractiony_turb * (boundary_point[1] - nearest_center[1])
        
        # Determine compressor contribution
        if boundary_point[1] > nearest_center[1]:
            comp_contribution = comp_delta_positive * uncertainty_fractiony_comp
        else:
            comp_contribution = comp_delta_negative * uncertainty_fractiony_comp
            
        if uncertainty_fractiony_fuel >= 0:
            fuel_contribution = fuel_delta_positive * abs(uncertainty_fractiony_fuel)
        else:
            fuel_contribution = fuel_delta_negative * abs(uncertainty_fractiony_fuel)
            
        
        if uncertainty_fractiony_carbon >= 0:
            carbon_contribution = carbon_delta_positive * abs(uncertainty_fractiony_carbon)
        else:
            carbon_contribution = carbon_delta_negative * abs(uncertainty_fractiony_carbon)
        
        interpolated_y = nearest_center[1] + turbine_delta_y + comp_contribution + fuel_contribution + carbon_contribution
        
        if cycle_name == 'Allam w ASU':
            interpolated_y += change_ASU[0] 
            interpolated_x = nearest_center[0]*(53.23/(change_ASU[1]  + 53.23)) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
            pass
        elif cycle_name == 'ICR CCS':
            interpolated_y += change_ICR_reboiler[0]
            interpolated_x = nearest_center[0]*(56.48/(change_ICR_reboiler[1] + 56.48)) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        elif cycle_name == 'CCGT CCS':
            interpolated_y += change_CCGT_reboiler[0]
            interpolated_x = nearest_center[0]*(55.07/(change_CCGT_reboiler[1] + 55.07)) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        else:
            pass
        
        interpolated_points.append([interpolated_x, interpolated_y])
    
    return np.array(interpolated_points)

# Define all cycle data with hardcoded values
cycle_data = {
    'CCGT': {
        'points': np.array([[ 5.66431791, 125.97109313      ],
       [7.83154389, 125.97109313      ],
       [7.93672881, 124.69990901      ],
       [7.93672881, 120.17129089     ],
       [ 5.74039497, 120.17129089      ],
       [ 5.66431791, 121.34797463      ]]),
        'center': np.array([[7.0,122.391],[6.90722948,123.61377052]]),
        'color': '#d53e4f',
        'zorder': 2000,
        'hatch': None,
        'has_ccs': False
    },
    'CCGT CCS': {
        'points': np.array([[15.92134292, 62.7750209      ],
               [24.49754373, 62.7750209      ],
               [24.49754373, 60.69527555      ],
               [24.04807046 , 59.06149348      ],
               [15.6292231, 59.06149348      ],
               [15.6292231, 60.69525511      ],
               ]),
        'center': np.array([[20.0, 59.86137],[20.37381234, 61.89018766]]),
        'color': '#d53e4f',
        'zorder': 800,
        'hatch': '//',
        'has_ccs': True
    },
    'ICR CCS': {
        'points': np.array([[ 13.835, 59.88500401      ],
               [20.83, 59.88500401      ],
               [21.11985984, 58.05982421      ],
               [21.11985984, 55.44136526    ],
               [13.96011852, 55.44136526      ],
               [ 13.835, 57.11502413      ]]),
        'center': np.array([[17.78, 56.70],[17.54, 58.46]]),
        'color': '#cc4c02',
        'zorder': 100,
        'hatch': '//',
        'has_ccs': True
    },
    'Allam w ASU': {
        'points': np.array([[ 22.55, 54.48814998      ],
               [25.8483402, 54.48814998      ],
               [26.15267155, 54.24632272      ],
               [26.15267155, 50.63746388     ],
               [22.73181856, 50.63746388     ],
               [22.55, 50.87929114      ]]),
        'center': np.array([[24.44,  52.37775494],[24.2, 53.23]]),
        'color': '#c2a5cf',
        'zorder': 2200,
        'hatch': None,
        'has_ccs': True
    }
}

# Data for sensitivity analysis
data = {
    'CCGT':    ([0.87, 14.915, 47.92795],[-0.801, -13.05,-77.37]),
    'CCGT CCS': ([0.44, 17.0895, 5.49],[-0.40, -14.95, -8.865]),
    'ICR CCS': ([0.5356, 16.66, 5.354],[-0.48, -14.58, -8.40]),
    'Allam w ASU': ([0.44, 17.68, 0.0],[- 0.40758, -15.47, 0.0])
}

#Modeled inputs for gas separation penalties
INIT_ASU_WORK = 1.59  # MJ/kg O2
INIT_REBOILER_DUTY_CCGT = 3.8  # MJ/kg CO2
INIT_REBOILER_DUTY_ICR = 3.9  # MJ/kg CO2

INIT_fuel_price = 68.0 #p/therm
INIT_carbon_price = 247.0 #£/tCO2

MIN_fuel_price = 47.0 #p/therm
MAX_fuel_price = 93.0 #p/therm

MIN_carbon_price = 0.0 #£/tCO2
MAX_carbon_price = 400.0 #£/tCO2

def load_gas_separation_data():
    """Load changes due to sensitivity to gas separation penalties."""
    
    #Modeled inputs for gas separation penalties
    INIT_ASU_WORK = 1.59  # MJ/kg O2
    INIT_REBOILER_DUTY_CCGT = 3.8  # MJ/kg CO2
    INIT_REBOILER_DUTY_ICR = 3.9  # MJ/kg CO2
    
    # Air separation unit sensitivity data (efficiency vs MJ/kg O2 work)
    asu_change_values = np.array([2.0, 1.79, 1.59, 1.39, 1.19, 0.99, 0.79, 0.59, 0.3])
    asu_efficiency_change = np.array([-3.38, -1.65, 0.0, 1.65, 3.3, 4.94, 6.59, 8.24, 10.63])
    
    # Reboiler duty sensitivity data for 90% capture  
    reboiler_duty_ICR = np.array([3.4, 3.6, 3.8, 3.9, 4.15, 4.48])
    efficiency_ICR = np.array([57.96, 58.04, 58.11, 58.1, 57.2, 55.7])
    
    reboiler_duty_CC3PRH = np.array([2.5 , 3.0,  4.0,  5.0 ])
    eff_CC3PRH = np.array([ 57.1  ,  56.3 , 54.78, 53.34])
    
    return {
        'asu_change_values': asu_change_values,
        'asu_efficiency_change': asu_efficiency_change,
        'reboiler_duty_ICR': reboiler_duty_ICR,
        'efficiency_ICR': efficiency_ICR,
        'reboiler_duty_CCGT': reboiler_duty_CC3PRH,
        'efficiency_CCGT': eff_CC3PRH
    }

def create_interpolation_functions_gas_separation(data):
    """Create interpolation functions for parameter sensitivity analysis."""
    
    # ASU work sensitivity function
    asu_change_func = interp1d(
        data['asu_change_values'], 
        data['asu_efficiency_change'], 
        kind='linear', 
        fill_value='extrapolate'
    )
    
    eff_ICR_func = interp1d(
        data['reboiler_duty_ICR'], 
        data['efficiency_ICR'], 
        kind='quadratic', 
        fill_value='extrapolate'
    )
    
    eff_CCGT_func = interp1d(
        data['reboiler_duty_CCGT'], 
        data['efficiency_CCGT'], 
        kind='quadratic', 
        fill_value='extrapolate'
    )
    
    #Modeled inputs for gas separation penalties
    INIT_ASU_WORK = 1.59  # MJ/kg O2
    INIT_REBOILER_DUTY_CCGT = 3.8  # MJ/kg CO2
    INIT_REBOILER_DUTY_ICR = 3.9  # MJ/kg CO2
    
    def reboiler_change_ICR(reb_duty):
        deff = eff_ICR_func(reb_duty) - eff_ICR_func(INIT_REBOILER_DUTY_ICR)
        dopex =  52.3 * ((58.82/ (56.48 + deff)) - (58.82/ 56.48))
        return [dopex, deff]
    
    def reboiler_change_CCGT(reb_duty):
        deff = eff_CCGT_func(reb_duty) - eff_CCGT_func(INIT_REBOILER_DUTY_CCGT)
        dopex =  52.3 * ((58.82/ (55.07 + deff)) - (58.82/ 55.07))
        return [dopex, deff]
    
    def asu_change_opex_func(asu_work):
        deff = asu_change_func(asu_work)
        dopex = 44.0 * ((58.82/ (53.23 + deff)) - (58.82/ 53.23))
        return [dopex, deff]
    
    return asu_change_opex_func, reboiler_change_ICR, reboiler_change_CCGT

# Load gas separation data and functions as module-level
gas_separation_data = load_gas_separation_data()
asu_change_func, reboiler_change_ICR, reboiler_change_CCGT = create_interpolation_functions_gas_separation(gas_separation_data)

def run_ccs_costs_plot():
    """Main function to create the interactive CCS costs plot."""
    
    # Create the plot with smaller size
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.7)
    
    
    # Initialize cycle polygons dictionary
    cycle_polygons = {}
    cycle_names = list(cycle_data.keys())

    # Plot all cycles initially
    for cycle_name, cycle_info in cycle_data.items():
        polygon = plot_polygon(ax, cycle_info['points'], 
                              facecolor=cycle_info['color'], 
                              zorder=cycle_info['zorder'],
                              hatch=cycle_info['hatch'])
        cycle_polygons[cycle_name] = polygon
        
    # Add table with specifications
    table_text = """Power Output - 640 MW
Coolant fraction - 10%
Pol. Eff. Compressor - 91%
Pol. Eff. Turbine - 87%
Max. Blade Temp. - 1000 C
Comb. Pressure Loss - 1%"""

    ax.text(1.022, 0.52, table_text, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='white', 
                     edgecolor='black',
                     alpha=0.9),
            zorder=10000)

    ax.set_xlabel(r'CapEx (£/MWh)',fontsize=20)
    ax.set_ylabel(r'OpEx (£/MWh)',fontsize=20)
    ax.set_xlim(-0.1, 30.0)
    ax.set_ylim(40.0, 140.0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='x', which='both', direction='in', top=True)
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    plt.minorticks_on()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.grid(True, zorder=0, linestyle=':')

    # Add legend
    handles = []
    for cycle_name, cycle_info in cycle_data.items():
        handles.append(Polygon([(0,0),(10,0),(0,-10)], 
                              facecolor=cycle_info['color'],
                              label=cycle_name.replace(' ', ' '), 
                              edgecolor='black',
                              linewidth=1.5,
                              hatch=cycle_info['hatch']))

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
        description='Heat/Mass Transfer Uncertainty (%):',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px')
    )
    
    turb_slider = widgets.FloatSlider(
        value=2, min=0, max=2, step=0.1,
        description='Turb. ηp Uncertainty (%):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )
    
    comp_slider = widgets.FloatSlider(
        value=0, min=0, max=2, step=0.1,
        description='Comp. ηp Uncertainty (%):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )
    
    asu_slider = widgets.FloatSlider(
        value=INIT_ASU_WORK, min=0.3, max=2.0, step=0.01,
        description='Air separation work (MJ/kg O₂):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )
    
    icr_reboiler_slider = widgets.FloatSlider(
        value=INIT_REBOILER_DUTY_ICR, min=2.5, max=5.0, step=0.01,
        description='ICR Reboiler duty (MJ/kg CO₂):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )
    
    ccgt_reboiler_slider = widgets.FloatSlider(
        value=INIT_REBOILER_DUTY_CCGT, min=2.5, max=5.0, step=0.01,
        description='CCGT Reboiler duty (MJ/kg CO₂):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )
    
    fuel_price_slider = widgets.FloatSlider(
        value=INIT_fuel_price, min=MIN_fuel_price, max=MAX_fuel_price, step=1.0,
        description='Fuel Cost (p/therm):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )
    
    carbon_price_slider = widgets.FloatSlider(
        value=INIT_carbon_price, min=MIN_carbon_price, max=MAX_carbon_price, step=1.0,
        description='Carbon Price (£/tCO₂):',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='400px')
    )

    # Create visibility checkboxes - more compact
    visibility_checkboxes = {}
    checkbox_list = []
    for cycle_name in cycle_names:
        checkbox = widgets.Checkbox(
            value=True, description=cycle_name,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='240px')
        )
        visibility_checkboxes[cycle_name] = checkbox
        checkbox_list.append(checkbox)

    # Create update mode radio buttons - compact
    # update_radio = widgets.RadioButtons(
    #     options=['All'] + cycle_names,
    #     value='All',
    #     description='Update:',
    #     style={'description_width': '60px'},
    #     layout=widgets.Layout(width='120px')
    # )
    
    # Create horizontal update mode checkboxes (like radio buttons)
    update_options = ['All'] + cycle_names
    update_checkboxes = {}
    update_checkbox_list = []
    
    for option in update_options:
        checkbox = widgets.Checkbox(
            value=(option == 'All'),  # Only 'All' checked initially
            description=option,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='240px')
        )
        update_checkboxes[option] = checkbox
        update_checkbox_list.append(checkbox)

    # Reset button
    reset_button = widgets.Button(description='Reset')

    def get_modified_points(cycle_name):
        """Get modified points for a specific cycle based on current slider values"""
        heat_transfer_uncertainty = heat_transfer_slider.value
        turbine_uncertainty = turb_slider.value
        compressor_uncertainty = comp_slider.value
        
        ASU_uncertainty = asu_slider.value
        ICR_reboiler_uncertainty = icr_reboiler_slider.value
        CCGT_reboiler_uncertainty = ccgt_reboiler_slider.value
        
        change_ASU = asu_change_func(ASU_uncertainty)
        change_ICR_reboiler = reboiler_change_ICR(ICR_reboiler_uncertainty)
        change_CCGT_reboiler = reboiler_change_CCGT(CCGT_reboiler_uncertainty)
        
        uncertainty_fractionx = heat_transfer_uncertainty / 20.0
        uncertainty_fractiony_turb = turbine_uncertainty / 2.0
        uncertainty_fractiony_comp = compressor_uncertainty / 2.0
        
        fuel_cost = fuel_price_slider.value
        carbon_price = carbon_price_slider.value
        
        if fuel_cost >= INIT_fuel_price:
            uncertainty_fractiony_fuel = (fuel_cost - INIT_fuel_price)/(MAX_fuel_price - INIT_fuel_price)
        else:
            uncertainty_fractiony_fuel = (fuel_cost - INIT_fuel_price)/(INIT_fuel_price - MIN_fuel_price)
            
        if carbon_price >= INIT_carbon_price:
            uncertainty_fractiony_carbon = (carbon_price - INIT_carbon_price)/(MAX_carbon_price - INIT_carbon_price)
        else:
            uncertainty_fractiony_carbon = (carbon_price - INIT_carbon_price)/(INIT_carbon_price - MIN_carbon_price)
        
        return interpolate_uncertainty_points(
            cycle_data[cycle_name]['center'], 
            cycle_data[cycle_name]['points'], 
            uncertainty_fractionx,
            uncertainty_fractiony_turb,
            uncertainty_fractiony_comp,
            uncertainty_fractiony_fuel,
            uncertainty_fractiony_carbon,
            data[cycle_name],
            change_ASU,
            change_ICR_reboiler,
            change_CCGT_reboiler,
            cycle_name
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
                new_points = get_modified_points(cycle_name)
                
                # Remove old polygon
                if cycle_name in cycle_polygons and cycle_polygons[cycle_name] in ax.patches:
                    cycle_polygons[cycle_name].remove()
                
                # Create new polygon
                polygon = plot_polygon(ax, new_points, 
                                     facecolor=cycle_data[cycle_name]['color'], 
                                     zorder=cycle_data[cycle_name]['zorder'],
                                     hatch=cycle_data[cycle_name]['hatch'])
                cycle_polygons[cycle_name] = polygon
                
        # Update axis limits to ensure all polygons are visible
        update_axis_limits(ax, cycle_polygons)
        
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
                    points = get_modified_points(cycle_name)
                else:
                    points = original_points[cycle_name]
                
                polygon = plot_polygon(ax, points, 
                                     facecolor=cycle_data[cycle_name]['color'], 
                                     zorder=cycle_data[cycle_name]['zorder'],
                                     hatch=cycle_data[cycle_name]['hatch'])
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
        asu_slider.value = INIT_ASU_WORK
        icr_reboiler_slider.value = INIT_REBOILER_DUTY_ICR
        ccgt_reboiler_slider.value = INIT_REBOILER_DUTY_CCGT
        fuel_price_slider.value = INIT_fuel_price
        carbon_price_slider.value = INIT_carbon_price
        
        # Reset all cycles to original if in individual mode
        if current_update_mode != 'All':
            for cycle_name in cycle_names:
                if visibility_checkboxes[cycle_name].value:
                    if cycle_name in cycle_polygons and cycle_polygons[cycle_name] in ax.patches:
                        cycle_polygons[cycle_name].remove()
                    
                    polygon = plot_polygon(ax, original_points[cycle_name], 
                                         facecolor=cycle_data[cycle_name]['color'], 
                                         zorder=cycle_data[cycle_name]['zorder'],
                                         hatch=cycle_data[cycle_name]['hatch'])
                    cycle_polygons[cycle_name] = polygon
            
        ax.set_xlim(-0.1, 30.0)
        ax.set_ylim(40.0, 140.0)
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
    asu_slider.observe(lambda change: update_plot(), names='value')
    icr_reboiler_slider.observe(lambda change: update_plot(), names='value')
    ccgt_reboiler_slider.observe(lambda change: update_plot(), names='value')
    fuel_price_slider.observe(lambda change: update_plot(), names='value')
    carbon_price_slider.observe(lambda change: update_plot(), names='value')

    # Create the UI layout - compact and organized at bottom
    # Organize checkboxes in rows
    checkbox_rows = []
    for i in range(0, len(checkbox_list), 2):  # 2 checkboxes per row
        row = widgets.HBox(checkbox_list[i:i+2])
        checkbox_rows.append(row)
        
    # Organize update checkboxes in rows  
    update_rows = []
    for i in range(0, len(update_checkbox_list), 3):  # 3 checkboxes per row
        row = widgets.HBox(update_checkbox_list[i:i+3])
        update_rows.append(row)
    
    visibility_section = widgets.VBox([
        widgets.HTML("<b>Show/Hide:</b>"),
        widgets.VBox(checkbox_rows)
    ], layout=widgets.Layout(width='440px'))
    
    # Update section with proper checkboxes
    update_section = widgets.VBox([
        widgets.HTML("<b>Update Mode:</b>"),
        widgets.VBox(update_rows)
    ], layout=widgets.Layout(width='440px'))
    
    # Stack visibility and update sections vertically
    selection_section = widgets.HBox([
        visibility_section,
        update_section
    ])
    
    # Group sliders in columns
    sliders_col1 = widgets.VBox([
        heat_transfer_slider,
        turb_slider,
        comp_slider,
        asu_slider
    ])
    
    sliders_col2 = widgets.VBox([
        icr_reboiler_slider,
        ccgt_reboiler_slider,
        fuel_price_slider,
        carbon_price_slider
    ])
    
    
    
    # Organize controls in horizontal layout
    sliders_row = widgets.HBox([
        sliders_col1,
        sliders_col2,
        reset_button
    ])
    
    # Organize controls in horizontal layout
    controls_row = widgets.VBox([
        selection_section,
        sliders_row
    ], layout=widgets.Layout(justify_content='space-between'))
    
    
    # Display plot first, then controls at bottom
    plt.show()
    display(controls_row)
    
    # Return the interactive elements
    slider_data = {
        'heat_transfer_slider': heat_transfer_slider,
        'turb_slider': turb_slider,
        'comp_slider': comp_slider,
        'asu_slider': asu_slider,
        'icr_reboiler_slider': icr_reboiler_slider,
        'ccgt_reboiler_slider': ccgt_reboiler_slider,
        'fuel_price_slider': fuel_price_slider,
        'carbon_price_slider': carbon_price_slider,
        'reset_button': reset_button
    }
    
    controls = {
        'visibility_checkboxes': visibility_checkboxes,
        'update_checkboxes': update_checkboxes
    }
    
    return fig, ax, slider_data, controls#, leg

# Test the function
# if __name__ == "__main__":
#     fig, ax, slider_data, controls, legend = run_ccs_costs_plot()



