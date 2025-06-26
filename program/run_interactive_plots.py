#!/usr/bin/env python3
"""
Power Generation Cycles Sensitivity Analysis Tool
Main program to run interactive plots

Author: lsl37
Created: June 2025
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

# Import plot modules
from cycle_efficiency_v_power_data import run_e_v_power_plot
from hydrogen_cycles import run_hydrogen_cycles_plot
from CCS_cycles import run_ccs_cycles_plot
from CCS_costs import run_ccs_costs_plot

# Import matplotlib to ensure proper backend
import matplotlib
import matplotlib.pyplot as plt

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗                                                            ║
║              Performance Limits of Low Carbon Power Generation        ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def print_menu():
    """Print the main menu options."""
    print("\nAvailable Visualizations:")
    print("─" * 50)
    print("1. Cycle Efficiency vs Power Output")
    
    print("2. Hydrogen gas-turbine cycles")
    
    print("3. Carbon capture cycles")
    
    print("4. Carbon capture cost analysis")
    
    print("0. Exit")
    print("─" * 50)

def run_visualization(choice):
    """Run the selected visualization."""
    try:
        if choice == '1':
            print("\nLoading Cycle Efficiency vs Power plot...")
            print("Use sliders to adjust:")
            print("- Air separation work")
            print("- Reboiler duty for different capture rates")
            fig, ax, slider_data, legend = run_e_v_power_plot()
            return fig, ax, slider_data, legend
            
        elif choice == '2':
            print("\nLoading Hydrogen gas-turbine cycles...")
            print("Features:")
            print("- Show/hide individual cycles")
            print("- Adjust component uncertainties")
            print("- Update all cycles or individual ones")
            fig, ax, slider_data, controls, legend = run_hydrogen_cycles_plot()
            return fig, ax, slider_data, controls, legend
            
        elif choice == '3':
            print("\nLoading Carbon Capture Cycles...")
            print("Click on turbine specifications tables to switch between:")
            print("- GELM2500 (Low Baseline)")
            print("- SGT6-9000HL (High Baseline)")
            fig, ax, slider_data, controls, legend = run_ccs_cycles_plot()
            return fig, ax, slider_data, controls, legend
            
        elif choice == '4':
            print("\nLoading Carbon Capture Costs...")
            print("Adjust economic parameters:")
            print("- Fuel costs (p/therm)")
            print("- Carbon price (£/tCO2)")
            fig, ax, slider_data, controls, legend = run_ccs_costs_plot()
            return fig, ax, slider_data, controls, legend
            
        else:
            print("\nInvalid choice. Please try again.")
            return None
            
    except Exception as e:
        print(f"\nError running visualization: {str(e)}")
        print("Please check that all dependencies are installed correctly.")
        return None

def check_latex_support():
    """Check if LaTeX is available and configure matplotlib accordingly."""
    try:
        import subprocess
        result = subprocess.run(['latex', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("LaTeX support detected and enabled.")
            return True
    except:
        pass
    
    print("LaTeX support not available. Using standard text rendering.")
    return False

def main():
    """Main entry point for the application."""
    print(BANNER)
    
    # Check for LaTeX support
    has_latex = check_latex_support()
    
    # Configure matplotlib backend for better interactive performance
    try:
        matplotlib.use('Qt5Agg')
        print("Using Qt5 backend for better performance.")
    except:
        print("Qt5 backend not available. Using default backend.")
    
    # Handle command line arguments
    if len(sys.argv) == 2 and sys.argv[1] in ['1', '2', '3', '4']:
        choice = sys.argv[1]
        print(f"\nRunning visualization {choice} from command line...")
        result = run_visualization(choice)
        if result:
            plt.show()
        return
    
    # Interactive menu loop
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                print("Exit choice menu")
                break
                
            result = run_visualization(choice)
            
            if result:
                return result
               
                    
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or press Ctrl+C to exit.")

if __name__ == "__main__":
    result = main()