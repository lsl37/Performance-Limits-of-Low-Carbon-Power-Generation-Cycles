#!/usr/bin/env python3
"""
Performance Limits of Low Carbon Power Generation Cycles
DOI: 

Interactive plot:
    -Figure 6
    -GELM2500 option as well

Author: lsl37
Created: June, 2025
Modified: For ipywidgets compatibility
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
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
                                  uncertainty_fractiony_turb, uncertainty_fractiony_comp, data,change_ASU, change_ICR_reboiler, change_CCGT_reboiler, cycle_name):
    """
    Interpolate between center points and boundary points based on uncertainty fractions.
    """
    interpolated_points = []
    
    # Compressor efficiency deltas ±2%
    comp_delta_positive = data[0][0]
    comp_delta_negative = data[1][0]
    
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
        
        interpolated_y = nearest_center[1] + turbine_delta_y + comp_contribution
        
        if cycle_name == 'Allam w ASU':
            interpolated_y += change_ASU 
            interpolated_x = nearest_center[0]*(nearest_center[1]/(change_ASU + nearest_center[1])) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        elif cycle_name == 'ICR CCS':
            interpolated_y += change_ICR_reboiler
            interpolated_x = nearest_center[0]*(nearest_center[1]/(change_ICR_reboiler + nearest_center[1])) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        elif cycle_name == 'CCGT CCS':
            interpolated_y += change_CCGT_reboiler
            interpolated_x = nearest_center[0]*(nearest_center[1]/(change_CCGT_reboiler + nearest_center[1])) + uncertainty_fractionx_heatttransfer * (boundary_point[0] - nearest_center[0])
        else:
            pass
        
        interpolated_points.append([interpolated_x, interpolated_y])
    
    return np.array(interpolated_points)

# Baseline data
simple_cycle_GELM2500 = np.array([1.0, 37.7])  # GELM2500
simple_cycle_SGT6 = np.array([0.33366803120414, 43.67])  # SGT6-9000HL

# Define cycle data for both turbines
cycle_data_SGT6 = {
    'GT CCS': {
        'points': np.array([[17.75048703, 37.61      ],
       [28.79046807, 37.61      ],
       [29.6367252 , 38.27      ],
       [29.6367252 , 40.78      ],
       [18.31552151, 40.78      ],
       [17.75048703, 40.11      ]]),
        'center': np.array([[23.976,39.525],[23.27047755,38.86]]),
        'color': 'lightgray',
        'zorder': 1000,
        'hatch': '//',
        'has_ccs': True
    },
    'CCGT': {
        'points': np.array([[ 7.44731778, 61.89      ],
       [12.60815622, 61.89      ],
       [12.77749549, 62.57      ],
       [12.77749549, 64.28      ],
       [ 7.55930462, 64.28      ],
       [ 7.44731778, 63.67      ]]),
        'center': np.array([[10.168,63.425],[10.0277,62.78]]),
        'color': '#d53e4f',
        'zorder': 2000,
        'hatch': None,
        'has_ccs': False
    },
    'CCGT Blue H2': {
        'points': np.array([[10.81718977, 40.755     ],
               [18.01380273, 40.755     ],
               [18.20302828, 41.4       ],
               [18.20302828, 42.63      ],
               [10.94232773, 42.63      ],
               [10.81718977, 41.985     ]]),
        'center': np.array([[14.572678, 42.015],[14.41549, 41.37]]),
        'color': '#74a9cf',
        'zorder': 2000,
        'hatch': '//',
        'has_ccs': True
    },
    'CCGT CCS': {
        'points': np.array([[16.33190757, 52.34      ],
               [26.76901013, 52.34      ],
               [26.76901013, 53.93      ],
               [26.2181102 , 55.87      ],
               [16.09216608, 55.87      ],
               [16.09216608, 54.26      ],
               ]),
        'center': np.array([[21.155, 54.9],[21.5504, 53.3]]),
        'color': '#d53e4f',
        'zorder': 800,
        'hatch': '//',
        'has_ccs': True
    },
    'ICR': {
        'points': np.array([[1.4721908083450321, 57.43],
         [2.4696076926202473, 57.43],
         [4.694825598940583, 59.15],
         [4.694825598940583, 62.08],
         [2.023324765471997, 62.08],
         [1.4721908083450321, 60.38]]),
        'center': np.array([[3.9417985, 60.615], [3.359075, 58.905]]),
        'color': '#cc4c02',
        'zorder': 1000,
        'hatch': None,
        'has_ccs': False
    },
    'ICR CCS': {
        'points': np.array([[ 8.79077815, 55.04      ],
               [14.40547141, 55.04      ],
               [16.46942063, 56.65      ],
               [16.46942063, 59.52      ],
               [10.23626116, 59.52      ],
               [ 8.79077815, 57.91      ]]),
        'center': np.array([[13.3525, 58.085],[11.59812478, 56.47]]),
        'color': '#cc4c02',
        'zorder': 100,
        'hatch': '//',
        'has_ccs': True
    },
    'Allam w/o ASU': {
        'points': np.array([[ 5.34167375, 62.56      ],
               [ 9.02336192, 62.56      ],
               [11.35487088, 63.21      ],
               [11.35487088, 66.42      ],
               [ 6.31675821, 66.42      ],
               [ 5.34167375, 65.75      ]]),
        'center': np.array([[8.835814545, 64.815],[7.182517835, 64.155]]),
        'color': '#762a83',
        'zorder': 1000,
        'hatch': None,
        'has_ccs': True
    },
    'Allam w ASU': {
        'points': np.array([[ 9.29730296, 51.26      ],
               [15.26288751, 51.26      ],
               [16.93801743, 51.92      ],
               [16.93801743, 55.206     ],
               [10.09464346, 55.206     ],
               [ 9.29730296, 54.52      ]]),
        'center': np.array([[13.516,  53.563],[12.28, 52.89]]),
        'color': '#c2a5cf',
        'zorder': 2200,
        'hatch': None,
        'has_ccs': True
    }
}

cycle_data_GELM2500 = {
    'GT CCS': {
        'points': np.array([[23.08747328, 31.3       ],
               [37.61454249, 31.3       ],
               [41.12263164, 32.05      ],
               [41.12263164, 35.1       ],
               [26.133977  , 35.1       ],
               [23.08747328, 34.35      ]]),
        'center': np.array([[33.62830432,33.575],[30.351007885,32.825]]),
        'color': 'lightgray',
        'zorder': 1000,
        'hatch': '//',
        'has_ccs': True
    },
    'CCGT': {
        'points': np.array([[ 8.03435656, 55.76      ],
               [12.76034854, 55.76      ],
               [13.38332774, 56.22      ],
               [13.38332774, 57.61      ],
               [ 8.20294798, 57.61      ],
               [ 8.03435656, 57.08      ]]),
        'center': np.array([[10.79313786,56.915],[10.39735255,56.42]]),
        'color': '#d53e4f',
        'zorder': 2000,
        'hatch': None,
        'has_ccs': False
    },
    'CCGT Blue H2': {
        'points': np.array([[12.94909139, 35.605     ],
               [21.04343003, 35.605     ],
               [21.25598993, 36.25      ],
               [21.25598993, 37.25      ],
               [13.07989029, 37.25      ],
               [12.94909139, 36.605     ]]),
        'center': np.array([[17.16794011, 36.75      ],
               [16.99626071, 36.105     ]]),
        'color': '#74a9cf',
        'zorder': 2000,
        'hatch': '//',
        'has_ccs': False
    },
    'CCGT CCS': {
        'points': np.array([[15.67900456, 46.61      ],
               [25.64007824, 46.61      ],
               [25.64007824, 47.94      ],
               [24.99942186, 49.78      ],
               [15.5494023 , 49.78      ],
               [15.5494023 , 48.23      ]]),
        'center': np.array([[20.27441208, 48.86],[20.6595, 47.42]]),
        'color': '#d53e4f',
        'zorder': 800,
        'hatch': '//',
        'has_ccs': True
    },
    'ICR': {
        'points': np.array([
         [2.2602596273537374, 51.63],
         [3.3528284365058982, 51.63],
         [6.344075398628757, 53.73],
         [6.344075398628757, 56.97],
         [3.0025883701171683, 56.97],
         [2.2602596273537374, 54.88]]),
        'center': np.array([[ 4.67333188, 55.35      ],
               [ 2.80654403, 53.255     ]]),
        'color': '#cc4c02',
        'zorder': 1000,
        'hatch': None,
        'has_ccs': False
    },
    'ICR CCS': {
        'points': np.array([[11.78889451, 48.394     ],
               [18.72326727, 48.394     ],
               [21.5219802 , 50.444     ],
               [21.5219802 , 53.684     ],
               [12.5348039 , 53.684     ],
               [11.78889451, 51.614     ]]),
        'center': np.array([[17.02839205, 52.064     ],
               [15.25608089, 50.004     ]]),
        'color': '#cc4c02',
        'zorder': 100,
        'hatch': '//',
        'has_ccs': True
    },
    'Allam w/o ASU': {
        'points': np.array([[ 5.72504001, 59.948     ],
               [ 9.72219478, 59.948     ],
               [12.0304914 , 60.488     ],
               [12.0304914 , 63.898     ],
               [ 6.71267116, 63.898     ],
               [ 5.72504001, 63.338     ]]),
        'center': np.array([[ 9.37158128, 62.193     ],
               [ 7.7236174 , 61.643     ]]),
        'color': '#762a83',
        'zorder': 1000,
        'hatch': None,
        'has_ccs': True
    },
    'Allam w ASU': {
        'points': np.array([[ 8.97091304, 47.726     ],
               [14.4157199 , 47.726     ],
               [16.09103495, 48.266     ],
               [16.09103495, 51.676     ],
               [ 9.78558021, 51.676     ],
               [ 8.97091304, 51.116     ]]),
        'center': np.array([[12.93830758, 49.971     ],
               [11.69331647, 49.421     ]]),
        'color': '#c2a5cf',
        'zorder': 2200,
        'hatch': None,
        'has_ccs': True
    }
}

# Data for sensitivity analysis
data = {
    'GT CCS': ([0.62],[-0.67]),
    'CCGT':    ([0.43],[-0.46]),
    'CCGT Blue H2':  ([0.3],[-0.32]),
    'CCGT CCS': ([0.4],[-0.43]),
    'ICR': ([0.5],[-0.53]),
    'ICR CCS': ([0.5],[-0.55]),
    'Allam w/o ASU': ([0.45],[-0.48]),
    'Allam w ASU': ([0.45],[-0.48])
}

# Table specifications
table_specs = {
    'SGT6': {
        'text': """Power Output - 640 MW
Coolant fraction - 10%
Pol. Eff. Compressor - 91%
Pol. Eff. Turbine - 87%
Max. Blade Temp. - 1000 C
Comb. Pressure Loss - 1%""",
        'position': (1.02, 0.75)
    },
    'GELM2500': {
        'text': """Power Output - 32 MW
Coolant fraction - 10%
Pol. Eff. Compressor - 91%
Pol. Eff. Turbine - 81.2%
Max. Blade Temp. - 860 C
Comb. Pressure Loss - 3%""",
        'position': (1.02, 0.25)
    }
}

#Modeled inputs for gas separation penalties
INIT_ASU_WORK = 1.59  # MJ/kg O2
INIT_REBOILER_DUTY_CCGT = 3.8  # MJ/kg CO2
INIT_REBOILER_DUTY_ICR = 3.9  # MJ/kg CO2

def load_gas_separation_data():
    """Load changes due to sensitivity to gas separation penalties."""
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
    
    def reboiler_change_ICR(reb_duty):
        return eff_ICR_func(reb_duty) - eff_ICR_func(INIT_REBOILER_DUTY_ICR)
    
    def reboiler_change_CCGT(reb_duty):
        return eff_CCGT_func(reb_duty) - eff_CCGT_func(INIT_REBOILER_DUTY_CCGT)
    
    return asu_change_func, reboiler_change_ICR, reboiler_change_CCGT

# Load gas separation data and functions as module-level
gas_separation_data = load_gas_separation_data()
asu_change_func, reboiler_change_ICR, reboiler_change_CCGT = create_interpolation_functions_gas_separation(gas_separation_data)

def run_ccs_cycles_plot():
    """Main function to create the interactive CCS plots."""
    
    # Create the plot with smaller size
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.7)

    # Plot baseline points
    ax.scatter(simple_cycle_GELM2500[0], simple_cycle_GELM2500[1], 
              color='black', marker='o', s=100, zorder=1000)
    ax.scatter(simple_cycle_SGT6[0], simple_cycle_SGT6[1], 
              color='black', marker='o', s=100, zorder=1000)

    # Initialize with SGT6 data
    current_turbine = 'High'
    cycle_data = cycle_data_SGT6.copy()
    cycle_names = list(cycle_data.keys())

    # Initialize cycle polygons dictionary
    cycle_polygons = {}

    # Plot all cycles initially
    for cycle_name, cycle_info in cycle_data.items():
        polygon = plot_polygon(ax, cycle_info['points'], 
                              facecolor=cycle_info['color'], 
                              zorder=cycle_info['zorder'],
                              hatch=cycle_info['hatch'])
        cycle_polygons[cycle_name] = polygon

    # Add table specifications
    
    table_text_SGT6 = """Power Output - 640 MW
Coolant fraction - 10%
Pol. Eff. Compressor - 91%
Pol. Eff. Turbine - 87%
Max. Blade Temp. - 1000 C
Comb. Pressure Loss - 1%"""

    ax.text(1.02, 0.34, table_text_SGT6, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      edgecolor='black',
                      alpha=0.9),
            zorder=10000)
    
    table_text_GELM2500 = """Power Output - 32 MW
Coolant fraction - 10%
Pol. Eff. Compressor - 91%
Pol. Eff. Turbine - 81.2%
Max. Blade Temp. - 860 C
Comb. Pressure Loss - 3%"""
    
    ax.text(1.02, 0.02, table_text_GELM2500, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      edgecolor='black',
                      alpha=0.9),
            zorder=10000)
        
    
    table_text_allam = """Pol. Eff. Compressor Allam - 83%
Pol. Eff. Pump Allam - 78%"""

    ax.text(1.02, -0.1, table_text_allam, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      edgecolor='black',
                      alpha=0.9),
            zorder=10000)

    ax.text(1.08,0.57,'High Baseline',transform=ax.transAxes,fontsize=12, zorder = 20000)
    ax.text(1.08,0.25,'Low Baseline',transform=ax.transAxes,fontsize=12, zorder = 20000)

    # Labels and text
    plt.text(0.95,35.7,'GELM2500',fontsize=12)
    plt.text(0.90,42.0,'SGT6-9000HL',fontsize=12)

    ax.set_xlabel(r'Size (-)',fontsize=20)
    ax.set_ylabel('Efficiency (%)',fontsize=20)
    ax.set_xlim(-0.1, 32.0)
    ax.set_ylim(35.0, 68.0)
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
                    loc='upper left', bbox_to_anchor=(1, 1.1))
    leg.set_draggable(state=False)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(2.0)

    # Store original points for each cycle
    original_points = {name: info['points'].copy() for name, info in cycle_data.items()}

    # Global variable to track update mode
    current_update_mode = 'All'

    # Create ipywidgets - more compact
    # Turbine selector buttons
    turbine_selector = widgets.ToggleButtons(
        options=['High', 'Low'],
        value='High',
        description='',
        button_style='info',
        style={'description_width': '60px'},
        layout=widgets.Layout(width='200px')
    )

    # Visibility checkboxes - more compact
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

    # Update mode radio buttons - compact
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
            layout=widgets.Layout(width='120px')
        )
        update_checkboxes[option] = checkbox
        update_checkbox_list.append(checkbox)

    # Sliders - more compact
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

    # Reset button
    reset_button = widgets.Button(description='Reset')

    def switch_turbine(turbine_type):
        """Switch between turbine datasets"""
        nonlocal current_turbine, cycle_data, cycle_polygons, original_points, visibility_checkboxes
        
        if current_turbine == turbine_type:
            return  # No change needed
        
        current_turbine = turbine_type
        
        # Remove all existing polygons
        for polygon in cycle_polygons.values():
            if polygon in ax.patches:
                polygon.remove()
        cycle_polygons.clear()
        
        # Switch data
        if turbine_type == 'High':
            cycle_data = cycle_data_SGT6.copy()
        else:
            cycle_data = cycle_data_GELM2500.copy()
        
        # Update original points
        original_points = {name: info['points'].copy() for name, info in cycle_data.items()}
        
        # Plot new polygons
        for cycle_name, cycle_info in cycle_data.items():
            if visibility_checkboxes[cycle_name].value:
                polygon = plot_polygon(ax, cycle_info['points'], 
                                     facecolor=cycle_info['color'], 
                                     zorder=cycle_info['zorder'],
                                     hatch=cycle_info['hatch'])
                cycle_polygons[cycle_name] = polygon
        
        # Update plot title
        if turbine_type == 'High':
            ax.set_title('High Baseline', fontsize=12, pad=10)
        else:
            ax.set_title('Low Baseline', fontsize=12, pad=10)
        
        fig.canvas.draw_idle()

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
        
        return interpolate_uncertainty_points(
            cycle_data[cycle_name]['center'], 
            cycle_data[cycle_name]['points'], 
            uncertainty_fractionx,
            uncertainty_fractiony_turb,
            uncertainty_fractiony_comp,
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

    def on_turbine_change(change):
        """Handle turbine selector change"""
        switch_turbine(change['new'])

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
            
        ax.set_xlim(-0.1, 32.0)
        ax.set_ylim(35.0, 68.0)
        fig.canvas.draw_idle()

    # Connect callbacks
    turbine_selector.observe(on_turbine_change, names='value')
    
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

    # Create the UI layout - compact and organized at bottom
    # Organize checkboxes in rows
    checkbox_rows = []
    for i in range(0, len(checkbox_list), 3):  # 3 checkboxes per row
        row = widgets.HBox(checkbox_list[i:i+3])
        checkbox_rows.append(row)
    
    # Organize update checkboxes in rows  
    update_rows = []
    for i in range(0, len(update_checkbox_list), 3):  # 3 checkboxes per row
        row = widgets.HBox(update_checkbox_list[i:i+3])
        update_rows.append(row)
    
    visibility_section = widgets.VBox([
        widgets.HTML("<b>Show/Hide:</b>"),
        widgets.VBox(checkbox_rows)
    ], layout=widgets.Layout(width='360px'))
    
    # Update section with proper checkboxes
    update_section = widgets.VBox([
        widgets.HTML("<b>Update Mode:</b>"),
        widgets.VBox(update_rows)
    ], layout=widgets.Layout(width='360px'))
    
    #Baseline section
    baseline_section = widgets.VBox([
        widgets.HTML("<b>Baseline:</b>"),
        turbine_selector
    ], layout=widgets.Layout(width='160px'))
    
    # Group sliders in columns
    sliders_col1 = widgets.VBox([
        heat_transfer_slider,
        turb_slider,
        comp_slider
    ])
    
    sliders_col2 = widgets.VBox([
        asu_slider,
        icr_reboiler_slider,
        ccgt_reboiler_slider
    ])
    
    # Top row with turbine selector and update mode
    top_controls = widgets.HBox([
        baseline_section,
        visibility_section,
        update_section
    ])
    
    # Bottom row with visibility and sliders
    bottom_controls = widgets.HBox([
        sliders_col1,
        sliders_col2,
        reset_button
    ], layout=widgets.Layout(justify_content='space-between'))
    
    # Organize all controls
    all_controls = widgets.VBox([
        top_controls,
        bottom_controls
    ])
    
    # Display plot first, then controls at bottom
    plt.show()
    display(all_controls)
    
    # Update plot title
    if current_turbine == 'High':
        ax.set_title('High Baseline', fontsize=12, pad=10)
    else:
        ax.set_title('Low Baseline', fontsize=12, pad=10)

    # Return the interactive elements
    slider_data = {
        'heat_transfer_slider': heat_transfer_slider,
        'turb_slider': turb_slider,
        'comp_slider': comp_slider,
        'asu_slider': asu_slider,
        'icr_reboiler_slider': icr_reboiler_slider,
        'ccgt_reboiler_slider': ccgt_reboiler_slider,
        'reset_button': reset_button
    }
    
    controls = {
        'visibility_checkboxes': visibility_checkboxes,
        'update_checkboxes': update_checkboxes, 
        'turbine_selector': turbine_selector
    }
    
    return fig, ax, slider_data, controls#, leg

##########################################
# Example run
# fig, ax, slider_data, controls, legend = run_ccs_cycles_plot()
##########################################