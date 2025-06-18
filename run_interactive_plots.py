#!/usr/bin/env python3
"""
Performance Limits of Low Carbon Power Generation Cycles
DOI: 
    
    -Run all interactive plots
    
Author: lsl37
Created: June, 2025
"""
import sys
from cycle_efficiency_v_power_data import run_e_v_power_plot

def function_1():
    print("Loading cycle efficiency vs power plot...")
    fig, ax, slider_data, legend = run_e_v_power_plot()  # Added parentheses to actually call the function
    return fig, ax, slider_data, legend

def function_2():
    print("Running function 2!")

def function_3():
    print("Running function 3!")

def main():
    functions = {
        '1': function_1,
        '2': function_2,
        '3': function_3
    }
    
    # Try command line argument first
    if len(sys.argv) == 2:
        choice = sys.argv[1]
    else:
        # Fall back to interactive input
        print("Choose an option:")
        print("1 - Cycle efficiency vs power (Allam and ICR)")
        print("2 - Run function 2") 
        print("3 - Run function 3")
        choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice in functions:
        result = functions[choice]()
        # Keep references alive for interactive plots
        if choice == '1' and result:
            return result
    else:
        print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":  # Fixed the double asterisks
    interactive_elements = main()

