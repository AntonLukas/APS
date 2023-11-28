__author__ = 'Anton Lukas'
########################################################################################################################
################################ FEFLOW Automatic Model Production Script: Version 0.8 #################################
########################################################################################################################

# FAMPS requires a basemodel with:
# - Bottom of model elevation at 0m.
# - All DFEs have an associated discrete feature selection saved as "DFn" (n is Index number of the DFE).
# - Only one abstraction borehole, which should have an observer node of the same name.
# - All boreholes are required to be fully penetrating multilayer wells.
# ONLY FEFLOW 7.5 IS SUPPORTED

# Import only the packages necessary to check if FEFLOW and ifm_contrib is present
import sys
import os

# Check the current operating system
if sys.platform == 'win32':
    sys.path.append('C:\\Program Files\\DHI\\2022\\FEFLOW 7.5\\python')
    os.environ['FEFLOW75_ROOT'] = 'C:\\Program Files\\DHI\\2022\\FEFLOW 7.5\\'
    os.environ['FEFLOW_KERNEL_VERSION'] = '75'
elif sys.platform == 'linux':
    sys.path.append('/opt/feflow/7.5/python/')
    os.environ['FEFLOW75_ROOT'] = '/opt/feflow/7.5/'
    os.environ['FEFLOW_KERNEL_VERSION'] = '75'
else:
    sys.exit("Unsupported operating system.")
# Try to import the ifm_contrib package
try:
    import ifm_contrib as ifm
except ModuleNotFoundError:
    sys.exit("ifm_contrib could not be imported.")
# Check which version of FEFLOW is being used
if ifm.getKernelVersion() < 7500:
    sys.exit("Unsupported version of FEFLOW. FEFLOW 7.5 is required.")

# Import the rest of the required packages
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# FAMPS CONSTANTS (SEE CONFIGURATION FILE):
INPUT_FEM = str                 # FEFLOW base model location
OUTPUT_FILE_DIRECTORY = str     # Output folder location
PUMP_BH_NAME = str              # Name of the one (and only) pumping borehole in the base model.
IS_CONFINED = bool              # True: Confined, False: Unconfined
TRANSIENT_DURATION = float      # Days - Full duration of transient simulation, default: (2 x PUMP_DURATION).
PUMP_DURATION = float           # Days - Duration of transient pumping
PUMP_RATES = dict               # m3/d - Different pumping rates available for cycling.
TIME_STEPS_DAYS = np.ndarray    # Days - Time steps in days
STATIC_WL = float               # Meters - Static water level elevation
NUMBER_LAYERS = int             # Number of layers in the base model
VERTICAL_DFE = int              # Number of vertical discrete feature elements in the base model.
HORIZONTAL_DFE = int            # Number of horizontal discrete feature elements in the base model.
STANDARD_BASEMODEL = bool       # Is the standard FAMPS base model being used.

# Config file location
CONFIG_FILE = str(os.path.join(os.path.abspath(os.getcwd()), os.path.join("data", "FAMPSconfig.json")))

# Global variables
gv_current_model = str
gv_current_pump_rate_id = int
gv_current_layer_matrix = int
gv_current_generator_layer_count = int
gv_current_pump_borehole_id = int
gv_current_pump_borehole_depth = int
gv_current_model_pass = bool
gv_current_rate_change = int
gv_current_rate_high = int
gv_current_rate_low = int
gv_df_model_head = pd.DataFrame()
gv_is_pump_off = bool
gv_time_pump_off = float
gv_rate_memory = []


# Model Generation function
def generate_model(hydraulic_conductivity_xyz, storage_param, discrete_features_list):
    # Define global variables
    global gv_current_pump_borehole_id
    global gv_current_pump_borehole_depth
    global gv_current_model_pass
    global gv_current_rate_change
    global gv_current_rate_high
    global gv_current_rate_low
    global gv_time_pump_off
    global gv_is_pump_off
    global gv_rate_memory

    # Get the start time
    runtime_var_start_time = datetime.now()
    # Print model description
    print(f"{gv_current_model} - Current Time: {runtime_var_start_time}")
    print(f"Layer configuration: {gv_current_layer_matrix}")
    print(f"Pumping Rate: {PUMP_RATES.get(gv_current_pump_rate_id)} m3/d")

    # Assign model parameter distribution
    doc, df_parameter_output = model_parameterization(hydraulic_conductivity_xyz=hydraulic_conductivity_xyz,
                                                      storage_param=storage_param)

    # Initialize discrete feature elements
    doc, dfe_dictionary = dfe_initializer(doc, discrete_features_list)

    # Get the information for the pumping well
    df_well_info = pd.DataFrame(doc.c.mesh.mlw())
    gv_current_pump_borehole_id = int(df_well_info[df_well_info['name'] == PUMP_BH_NAME].index.values)
    borehole_top = float(df_well_info[df_well_info['name'] == PUMP_BH_NAME].top_z.values)
    borehole_bot = float(df_well_info[df_well_info['name'] == PUMP_BH_NAME].bottom_z.values)
    gv_current_pump_borehole_depth = borehole_top - borehole_bot

    # Set initial variable values
    gv_current_model_pass = False
    gv_current_rate_high = max(PUMP_RATES.keys())
    gv_current_rate_low = min(PUMP_RATES.keys())
    gv_rate_memory = [gv_current_pump_rate_id]
    runtime_var_stop_time = 0
    runtime_var_elapsed_time = 0
    pumping_phase_data = pd.DataFrame()

    # While the current groundwater model does not pass
    while not gv_current_model_pass:
        gv_current_rate_change = 0
        gv_is_pump_off = False
        gv_time_pump_off = 0
        # Start the simulation
        doc = run_simulation(doc)
        # If the pump was stopped prematurely
        if gv_current_rate_change == 1:  # Too much drawdown
            # Get the drawdown history prior to pump deactivation
            pumping_phase_data = doc.c.hist.df.HEAD
            # Stop the pump
            doc.stopSimulator()
            # Deactivate the pump time series
            doc.setMultiLayerWellAttrTSID(gv_current_pump_borehole_id, 0, -1)
            # Shorten the model length to the recovery period
            doc.setFinalSimulationTime(TRANSIENT_DURATION - PUMP_DURATION)
            # Print pump deactivation message
            print(f"{gv_current_model}: Pump deactivated, starting transient recovery.")
            # Resume the simulation
            doc.startSimulator()

        # Check if the pump was stopped prematurely, if so, attach the pumping phase data
        if gv_is_pump_off:
            # Attach the pumping phase data to the save function
            save_results(doc, df_parameter_output, dfe_dictionary=dfe_dictionary, pumping_phase_data=pumping_phase_data)
        else:
            # Save the simulation results as is
            save_results(doc, df_parameter_output, dfe_dictionary)
        
        # Stop the Simulation
        doc.stopSimulator()
        # Get the simulation runtime
        runtime_var_stop_time = datetime.now()
        runtime_var_elapsed_time = runtime_var_stop_time - runtime_var_start_time
        # By using the drawdown, determine if we need a higher or lower pumping rate
        if gv_current_rate_change == 1:  # Too much drawdown
            if gv_current_pump_rate_id == gv_current_rate_low:
                gv_current_model_pass = True
            else:
                pumping_rate_selector(True)

                if gv_current_pump_rate_id in gv_rate_memory:
                    gv_current_model_pass = True
                else:
                    gv_rate_memory.append(gv_current_pump_rate_id)
                    print(f"Simulation Completed. Elapsed Time: {runtime_var_elapsed_time}")
                    print(f"New Pumping Rate: {PUMP_RATES.get(gv_current_pump_rate_id)} m3/d.")
                    print(f"Restarting simulation...")
        elif gv_current_rate_change == 2:  # Too little drawdown
            if gv_current_pump_rate_id == gv_current_rate_high:
                gv_current_model_pass = True
            else:
                pumping_rate_selector(False)

                if gv_current_pump_rate_id in gv_rate_memory:
                    gv_current_model_pass = True
                else:
                    gv_rate_memory.append(gv_current_pump_rate_id)
                    print(f"New Pumping Rate: {PUMP_RATES.get(gv_current_pump_rate_id)} m3/d. Restarting simulation...")
        else:
            gv_current_model_pass = True

    # Print completion message
    print(f"{gv_current_model}: Transient simulation completed.")
    print(f"Current Time: {runtime_var_stop_time}, Elapsed Time: {runtime_var_elapsed_time}.")
    # Close the FEFLOW document
    doc.closeDocument()


# Function that assigns hydraulic parameters to the different hydrostratigraphic units
def model_parameterization(hydraulic_conductivity_xyz, storage_param):
    # File load variables
    loaded = False
    reconnection_attempts = 0
    # Try to load the FEFLOW base model
    while not loaded:
        try:
            # Load the FEFLOW base model
            doc = ifm.loadDocument(INPUT_FEM)
            # Set the boolean to true for success
            loaded = True
        except ConnectionError:
            # Print an error message
            print("Failed to establish a connection to the FEFLOW license server.")
            # Check how many reconnection attempts have been made
            if reconnection_attempts > 28:
                sys.exit("Failed to establish a connection to the FEFLOW license server.")
            else:
                # Wait and retry
                time.sleep(300)
                reconnection_attempts += 1
        except FileNotFoundError:
            # Exit the program
            sys.exit("Failed to find the specified FEFLOW base-model.")

    # Get the current parameter configuration (CPC) for the model
    df_kx = doc.c.mesh.df.elements(par={"CONDX": ifm.Enum.P_CONDX})
    df_ky = doc.c.mesh.df.elements(par={"CONDY": ifm.Enum.P_CONDY})
    df_kz = doc.c.mesh.df.elements(par={"CONDZ": ifm.Enum.P_CONDZ})
    df_s = doc.c.mesh.df.elements(par={"STOR": ifm.Enum.P_COMP})

    # List to record parameter assignment for output
    hydraulic_conductivity_parameters = []
    storativity_parameters = []
    current_model_layer = 0

    # Apply the PCMs according to the LCM for each layer in the CPC
    for ModelLayer in range(gv_current_generator_layer_count):
        # Record parameter assignment for output
        hydraulic_conductivity_parameters.append(hydraulic_conductivity_xyz[ModelLayer])
        storativity_parameters.append(storage_param[ModelLayer])
        # For each base model layer
        for _ in range(gv_current_layer_matrix[ModelLayer]):
            current_model_layer += 1
            df_kx.loc[df_kx.LAYER == (current_model_layer*3)-3, 'CONDX'] = hydraulic_conductivity_xyz[ModelLayer]
            df_kx.loc[df_kx.LAYER == (current_model_layer*3)-2, 'CONDX'] = hydraulic_conductivity_xyz[ModelLayer]
            df_kx.loc[df_kx.LAYER == (current_model_layer*3)-1, 'CONDX'] = hydraulic_conductivity_xyz[ModelLayer]
            df_ky.loc[df_ky.LAYER == (current_model_layer*3)-3, 'CONDY'] = hydraulic_conductivity_xyz[ModelLayer]
            df_ky.loc[df_ky.LAYER == (current_model_layer*3)-2, 'CONDY'] = hydraulic_conductivity_xyz[ModelLayer]
            df_ky.loc[df_ky.LAYER == (current_model_layer*3)-1, 'CONDY'] = hydraulic_conductivity_xyz[ModelLayer]
            df_kz.loc[df_kz.LAYER == (current_model_layer*3)-3, 'CONDZ'] = hydraulic_conductivity_xyz[ModelLayer]
            df_kz.loc[df_kz.LAYER == (current_model_layer*3)-2, 'CONDZ'] = hydraulic_conductivity_xyz[ModelLayer]
            df_kz.loc[df_kz.LAYER == (current_model_layer*3)-1, 'CONDZ'] = hydraulic_conductivity_xyz[ModelLayer]
            df_s.loc[df_s.LAYER == (current_model_layer*3)-3, 'STOR'] = storage_param[ModelLayer]
            df_s.loc[df_s.LAYER == (current_model_layer*3)-2, 'STOR'] = storage_param[ModelLayer]
            df_s.loc[df_s.LAYER == (current_model_layer*3)-1, 'STOR'] = storage_param[ModelLayer]
            
    # Update the model with the new values
    print(f"{gv_current_model}: Updating parameter distribution.")
    doc.setParamValues(ifm.Enum.P_CONDX, list(df_kx.CONDX))
    doc.setParamValues(ifm.Enum.P_CONDY, list(df_ky.CONDY))
    doc.setParamValues(ifm.Enum.P_CONDZ, list(df_kz.CONDZ))

    # Combining the parameter assignment lists into one dictionary...
    dic_parameters = {"Layers": gv_current_layer_matrix, "Kxyz": hydraulic_conductivity_parameters}

    # If the model is confined: update specific storativity, if unconfined: update specific yield (porosity)
    if IS_CONFINED:
        doc.setParamValues(ifm.Enum.P_COMP, list(df_s.STOR))
        dic_parameters["SpecificStorativity"] = storativity_parameters
        # ...which is subsequently converted into a DataFrame for easy export to excel.
        df_parameter_output = pd.DataFrame(dic_parameters, columns=["Layers", "Kxyz", "SpecificStorativity"])
    else:
        doc.setParamValues(ifm.Enum.P_UPORO, list(df_s.STOR))
        dic_parameters["Porosity"] = storativity_parameters
        # ...which is subsequently converted into a DataFrame for easy export to excel.
        df_parameter_output = pd.DataFrame(dic_parameters, columns=["Layers", "Kxyz", "Porosity"])

    return doc, df_parameter_output


# Function that activates discrete and elemental features
def dfe_initializer(doc, discrete_features_list):
    # Create export lists
    list_dfe_index = []
    list_dfe_thickness = []
    list_dfe_conductivity = []

    # Activate discrete feature elements
    for discrete_feature in discrete_features_list:
        # Split the Feature string into its components
        attr_split = str(discrete_feature).split("a")
        # Assign the components to variables
        selection_mode = int(attr_split[0])  # 0: Horizontal, 1: Vertical, 2: Arbitrary
        flow = int(attr_split[1])  # 0: Darcy, 1: HP, 2: MS
        aperture = int(attr_split[2])  # mm
        dfe_index = int(attr_split[3])  # Index of the DFE in the base model
        # 1 = Fracture - Hagen-Poiseuille
        if flow == 1:
            # Get the selection for the feature
            feat_selection = doc.getSelectionItems(ifm.Enum.SEL_FRACS, f"Discrete Feature Selection #{dfe_index}")
            # Loop through a range on DFE indexes, assigning values
            for ffFeature in feat_selection:
                # Assign discrete feature flow law and aperture
                doc.setFracLaw(ffFeature, -1, -1, flow)
                doc.c.dfe.setFracFlowConductivity(ffFeature, (aperture/250))
                # Assign area parameter to the discrete feature
                doc.c.dfe.setFracArea(ffFeature, (aperture/25))
            # Export information
            list_dfe_thickness.append((aperture/25))
        # 0 = Impermeable fault/dyke - Darcy
        elif flow == 0:
            # Get the elemental selection
            current_nodal_selection = doc.getSelectionItems(ifm.Enum.SEL_ELEMENTAL, f"Element Selection #{dfe_index}")
            # Set the selection as an impermeable barrier - Darcy (Hydraulic Conductivity set to 0 m3/d)
            for selected_node in current_nodal_selection:
                doc.setParamValue(ifm.Enum.P_CONDX, selected_node, 0.00)
                doc.setParamValue(ifm.Enum.P_CONDY, selected_node, 0.00)
                doc.setParamValue(ifm.Enum.P_CONDZ, selected_node, 0.00)
            # Export information
            list_dfe_thickness.append("Closed")
        else:
            print(f"Error in Feature Creation: Invalid Flow mode = {flow}")
        # Append the resulting values to the output lists
        list_dfe_index.append(dfe_index)
        list_dfe_conductivity.append((aperture/250))
    # Combining the discrete features lists into one dictionary...
    dfe_dictionary = {"Index": list_dfe_index, "Thickness": list_dfe_thickness, "Aperture": list_dfe_conductivity}
    # Return the model file and the dictionary of added DFEs
    return doc, dfe_dictionary


# Function that initiates and stops the simulation as necessary
def run_simulation(doc):
    # Configure model for steady state simulation
    STEADY_DURATION = 10000000  # Workaround, there does not seem to be a way to deactivate the FST/ITI/CT once set.
    # Deactivate the pumping well
    doc.setMultiLayerWellAttrTSID(gv_current_pump_borehole_id, 0, -1)
    # Deactivate final simulation time
    doc.setFinalSimulationTime(STEADY_DURATION)  # days
    # Deactivate the initial time step increment
    doc.setInitialTimeIncrement(STEADY_DURATION)
    # Deactivate custom times
    doc.setCustomTimes(np.array([0, STEADY_DURATION]).tolist())
    # Set the solver to direct (steady-state)
    doc.setEquationSolvingType(ifm.Enum.EQSOLV_DIRECT)

    # Perform steady-state simulation
    print(f"{gv_current_model}: Starting steady state simulation.")
    doc.startSimulator()
    doc.stopSimulator()
    print(f"{gv_current_model}: Steady state simulation completed.")

    # Configure model for transient simulation
    # Set the solver to iterative (transient)
    doc.setEquationSolvingType(ifm.Enum.EQSOLV_ITERAT)
    # Set final simulation time
    doc.setFinalSimulationTime(TRANSIENT_DURATION)  # days
    # Set the initial time step increment
    doc.setInitialTimeIncrement((1 / 60) / 24)  # days
    # Set the array of custom times which will be navigated to by the adaptive time stepping control.
    doc.setCustomTimes(np.array(TIME_STEPS_DAYS).tolist())
    # Activate the pumping well by selecting the pumping schedule
    doc.setMultiLayerWellAttrTSID(gv_current_pump_borehole_id, 0, gv_current_pump_rate_id)

    print(f"{gv_current_model}: Starting transient simulation.")
    # Create output file strings
    # OUTPUT_NAME = f"TemptOut"
    # doc.setOutput(f"{OUTPUT_NAME}.dac", ifm.Enum.F_ASCII, TIME_STEPS_DAYS)

    # Perform transient simulation
    doc.startSimulator()
    # Return the document
    return doc


# Function to plot a particular parameter in slice view
def model_figure_plotter(doc, parameter, filename, is_confined):
    # Create a new matplotlib figure.
    fig, ax = plt.subplots(1, figsize=(20, 12))
    # Set equal x and y axis.
    plt.axis("equal")

    # Add the mesh
    doc.c.plot.faces()
    doc.c.plot.edges(alpha=0.1)

    # Plot the data
    if parameter == "Head":
        doc.c.plot.continuous(par=ifm.Enum.P_HEAD,
                              cmap="feflow_classic")
        # Add color bar
        cbar = plt.colorbar()
        cbar.set_label('Hydraulic Head [mamsl]', rotation=270)
        # Add title
        plt.title("Model Head Distribution at end of Pumping.")
    elif parameter == "Cond":
        doc.c.plot.fringes(par=ifm.Enum.P_CONDX,
                           cmap="feflow_classic",
                           slice=1)
        # Add color bar
        cbar = plt.colorbar()
        cbar.set_label('Hydraulic Conductivity [m/d]', rotation=270)
        # Add title
        plt.title("Model Hydraulic Conductivity Parameterization.")
    elif parameter == "Stor":
        # Check the type of the system
        if is_confined:
            doc.c.plot.fringes(par=ifm.Enum.P_COMP,
                               levels=[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                               cmap="feflow_classic",
                               slice=1)
            # Add color bar
            cbar = plt.colorbar()
            cbar.set_label('Specific Storativity', rotation=270)
        else:
            doc.c.plot.fringes(par=ifm.Enum.P_UPORO,
                               levels=[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                               cmap="feflow_classic",
                               slice=1)
            # Add color bar
            cbar = plt.colorbar()
            cbar.set_label('Specific Yield', rotation=270)
        # Add title
        plt.title("Model Storativity Parameterization.")

    # Add the observation points
    doc.c.plot.obs_markers()
    # doc.c.plot.obs_labels()

    # Add axis labels
    plt.xlabel("Xcoord (meters)")
    plt.ylabel("Ycoord (meters)")

    # Save and clear
    plt.savefig(f"{filename}.png", transparent=False)  # png image

    if parameter == "Head":
        # Second plot with reduced extent in view
        plt.xlim(4500, 5500)
        plt.ylim(4500, 5500)
        # Save and clear
        plt.savefig(f"{filename}_Zoom1.png", transparent=False)  # png image
        # Third plot with extreme reduced extent in view
        plt.xlim(4750, 5250)
        plt.ylim(4750, 5250)
        # Save and clear
        plt.savefig(f"{filename}_Zoom2.png", transparent=False)  # png image
        # Fourth plot with very localised extent in view
        plt.xlim(5100, 5200)
        plt.ylim(4950, 5050)
        # Save and clear
        plt.savefig(f"{filename}_Zoom3.png", transparent=False)  # png image

    plt.clf()
    plt.close()


# Function to create and export the results of the active simulation
def save_results(doc, df_parameter_output, dfe_dictionary, pumping_phase_data=pd.DataFrame()):
    if gv_is_pump_off:
        # Get the recovery data
        df_recovery_results = doc.c.hist.df.HEAD
        # Add the final pumping time to the recovery time
        df_recovery_results.index += pumping_phase_data.index.max()
        # Add the pumping phase data to the recovery data
        df_model_results = pd.concat([pumping_phase_data, df_recovery_results], axis=0)
    else:
        # Get observation data as is
        df_model_results = doc.c.hist.df.HEAD

    # Create DataFrame with model setup information
    df_well_info = pd.DataFrame(doc.c.mesh.mlw())
    df_pump_rate = pd.DataFrame(doc.c.ts.points(gv_current_pump_rate_id), columns=['Simulation Time', 'Rate'])
    df_discrete_features = pd.DataFrame(dfe_dictionary, columns=['Index', 'Thickness', 'Aperture'])

    # Check for premature pump deactivation
    if gv_is_pump_off:
        df_pump_rate.at[1, "Simulation Time"] = gv_time_pump_off

    # Create output file strings
    output_name = f"{gv_current_model}R{gv_current_pump_rate_id}"
    output_dir = f"{OUTPUT_FILE_DIRECTORY}{output_name}"

    # Create output directories
    if not os.path.exists(f"{output_dir}"):
        os.mkdir(f"{output_dir}")
    head_distribution_directory = os.path.join(output_dir, "HeadDistribution")
    if not os.path.exists(head_distribution_directory):
        os.mkdir(head_distribution_directory)
    images_directory = os.path.join(output_dir, "Images")
    if not os.path.exists(images_directory):
        os.mkdir(images_directory)

    # Plot data
    head_image_filename = os.path.join(images_directory, f"{output_name}_Head")
    model_figure_plotter(doc=doc, parameter="Head", filename=head_image_filename, is_confined=IS_CONFINED)
    
    # Get the different slice elevations
    elevation_list = sorted(gv_df_model_head['Z'].unique())

    # For each slice in the model...
    for elev_value in elevation_list:
        # Get the head data at end of pumping for the current slice.
        df_slice_data = gv_df_model_head[gv_df_model_head['Z'] == elev_value]
        # Make the filename and path
        head_filename = os.path.join(head_distribution_directory, f"{output_name}_{int(elev_value + 0.5)}_Head.csv")
        # Save the data to a csv file
        df_slice_data.to_csv(head_filename)

    # Create the Excel file path and name
    excel_out_path = os.path.join(output_dir, f"{output_name}_Data.xlsx")
    # Write the general data to a Microsoft Excel file
    with pd.ExcelWriter(excel_out_path, engine="openpyxl") as writer:
        df_well_info.to_excel(writer, sheet_name="BoreholeInfo")
        df_model_results.to_excel(writer, sheet_name="ModelResults")
        df_parameter_output.to_excel(writer, sheet_name="ModelParameters")
        df_discrete_features.to_excel(writer, sheet_name="DiscreteFeatures")
        df_pump_rate.to_excel(writer, sheet_name="PumpRate")


# Function that either increases or decreases the pumping rate
def pumping_rate_selector(b_decrease: bool):
    """
        Take an array, take the middle value, run the model.
        Find out if > or <, adjust range.
        Repeat.
    """
    global gv_current_pump_rate_id
    global gv_current_rate_low
    global gv_current_rate_high
    global gv_current_rate_change

    if b_decrease:
        gv_current_rate_high = gv_current_pump_rate_id
        gv_current_pump_rate_id = int((gv_current_rate_low + gv_current_rate_high) / 2)
    else:
        gv_current_rate_low = gv_current_pump_rate_id
        gv_current_pump_rate_id = int(((gv_current_rate_low + gv_current_rate_high) / 2) + 0.5)


# FEFLOW Callback function, occurs at the end of every time step
def postTimeStep(doc):
    if doc.getMultiLayerWellAttrTSID(gv_current_pump_borehole_id, 0) == -1:
        return

    global gv_current_model_pass
    global gv_current_rate_change
    global gv_df_model_head
    global gv_is_pump_off
    global gv_time_pump_off

    # Get the drawdown data for the pumping well at current time step
    f_pbh_drawdown = float(doc.getFlowValueOfObsIdAtCurrentTime(doc.findObsByLabel(PUMP_BH_NAME)))

    if f_pbh_drawdown <= 0.5:
        print(f"{gv_current_model}: Available drawdown exceeded.")
        print(f"Time: {doc.getAbsoluteSimulationTime() * 24 * 60} min.")
        print(f"Drawdown: {STATIC_WL - f_pbh_drawdown} m, Available Drawdown: {STATIC_WL} m.")
        # Provide a model fail result and suggest a reduction in pumping rate
        gv_current_model_pass = False
        gv_current_rate_change = 1
        # Export the time step for the pump deactivation
        gv_is_pump_off = True
        gv_time_pump_off = doc.getAbsoluteSimulationTime()

        # Export the head at end of pumping
        dic_head_out = {}
        for node in range(0, doc.getNumberOfNodes()):
            # Round the model head data as it is assigned for each node
            dic_head_out[node] = [round(doc.getX(node), 2),
                                  round(doc.getY(node), 2),
                                  round(doc.getZ(node), 2),
                                  round(doc.getResultsFlowHeadValue(node), 3)]
        df_model_head_data = pd.DataFrame.from_dict(dic_head_out, orient='index', columns=['X', 'Y', 'Z', 'Head'])
        # Compress the model head data
        for column in df_model_head_data:
            # Downcast the float values
            if df_model_head_data[column].dtype == 'float64':
                df_model_head_data[column]=pd.to_numeric(df_model_head_data[column], downcast='float')
        # Assign the data to the global dataframe
        gv_df_model_head = df_model_head_data
        # Stop the pump
        doc.pauseSimulator()
    
    if float(doc.getAbsoluteSimulationTime()) == PUMP_DURATION:
        if gv_current_rate_change != 1:
            # Export the head at end of pumping
            dic_head_out = {}
            for node in range(0, doc.getNumberOfNodes()):
                # Round the model head data as it is assigned for each node
                dic_head_out[node] = [round(doc.getX(node), 2),
                                      round(doc.getY(node), 2),
                                      round(doc.getZ(node), 2),
                                      round(doc.getResultsFlowHeadValue(node), 3)]
            df_model_head_data = pd.DataFrame.from_dict(dic_head_out, orient='index', columns=['X', 'Y', 'Z', 'Head'])
            # Compress the model head data
            for column in df_model_head_data:
                # Downcast the float values
                if df_model_head_data[column].dtype == 'float64':
                    df_model_head_data[column]=pd.to_numeric(df_model_head_data[column], downcast='float')
            # Assign the data to the global dataframe
            gv_df_model_head = df_model_head_data

        if (STATIC_WL-f_pbh_drawdown) < (STATIC_WL/2) and gv_current_rate_change != 1:
            if gv_current_rate_high == gv_current_pump_rate_id:
                print(f"Drawdown: {STATIC_WL-f_pbh_drawdown}m, Available Drawdown: {STATIC_WL}m. Max rate achieved.")
                gv_current_model_pass = True
                gv_current_rate_change = 0
            else:
                print(f"Drawdown: {STATIC_WL-f_pbh_drawdown}m, Available Drawdown: {STATIC_WL}m. Next pumping rate.")
                gv_current_model_pass = False
                gv_current_rate_change = 2


# Load the configuration file
def load_famps_config(configfile):
    """
        Load the FAMPS configuration file.
    """
    # Prepare global variables for assignment
    global INPUT_FEM
    global OUTPUT_FILE_DIRECTORY
    global PUMP_BH_NAME
    global IS_CONFINED
    global PUMP_DURATION
    global PUMP_RATES
    global STATIC_WL
    global TIME_STEPS_DAYS
    global TRANSIENT_DURATION
    global NUMBER_LAYERS
    global VERTICAL_DFE
    global HORIZONTAL_DFE
    global STANDARD_BASEMODEL
    # Assign the paths for OS preprocessing
    input_fem_path = configfile["INPUT_FEM"]
    output_dir_path = configfile["OUTPUT_FILE_DIRECTORY"]
    # Check the current operating system
    if sys.platform == 'win32':
        # If the directories have a linux filestructure
        if "/" in input_fem_path:
            # replace each instance of a forward slash with a backslash
            input_fem_path.replace("/", "\\")
        if "/" in output_dir_path:
            # replace each instance of a forward slash with a backslash
            output_dir_path.replace("/", "\\")
    else:
        # If the directories have a Windows filestructure
        if "\\" in input_fem_path:
            # replace each instance of a forward slash with a backslash
            input_fem_path.replace("\\", "/")
        if "\\" in output_dir_path:
            # replace each instance of a forward slash with a backslash
            output_dir_path.replace("\\", "/")
    # Assign the FAMPS configuration file variables
    INPUT_FEM = configfile["INPUT_FEM"]
    OUTPUT_FILE_DIRECTORY = configfile["OUTPUT_FILE_DIRECTORY"]
    PUMP_BH_NAME = configfile["PUMP_BH_NAME"]
    IS_CONFINED = configfile["IS_CONFINED"]
    PUMP_DURATION = configfile["PUMP_DURATION"]
    PUMP_RATES = eval(configfile["PUMP_RATES"])
    STATIC_WL = configfile["STATIC_WL"]
    TIME_STEPS_DAYS = (np.array(configfile["MEASUREMENT_TIME_STEPS"]) / 60) / 24  # Days
    TRANSIENT_DURATION = 2 * PUMP_DURATION  # Days
    NUMBER_LAYERS = configfile["NUMBER_LAYERS"]
    VERTICAL_DFE = configfile["VERTICAL_DFE"]
    HORIZONTAL_DFE = configfile["HORIZONTAL_DFE"]
    STANDARD_BASEMODEL = configfile["STANDARD_BASEMODEL"]


# Main function
def main():
    # Linking global variables
    global gv_current_layer_matrix
    global gv_current_generator_layer_count
    global gv_current_model
    global gv_current_pump_rate_id

    # Load the script configuration file
    with open(CONFIG_FILE, "r") as fConfig:
        dic_config = json.load(fConfig)
    # Assign the global configuration variables
    load_famps_config(dic_config)
    # STARTING POINT CONTROL, MAKE VALUES ZERO IN CONFIG FILE TO DISABLE
    SP_MODEL_NUMBER = dic_config["SP_MODEL_NUMBER"]          # Model template number
    SP_MODEL_ITERATION = dic_config["SP_MODEL_ITERATION"]    # Model iteration number
    SP_MODEL_DFE = dic_config["SP_MODEL_DFE"]                # Model DFE iteration number

    # Create output directory
    if not os.path.exists(f"{OUTPUT_FILE_DIRECTORY}"):
        os.mkdir(f"{OUTPUT_FILE_DIRECTORY}")    

    # Load the TemplateGenerator .xlsx file
    df_template_generator = pd.read_excel(str(os.path.abspath(os.getcwd())) + dic_config["MODEL_MATRICES_FILE"])

    # For each iteration in the TemplateGenerator file
    for model_iteration in range(df_template_generator.shape[0]):
        # Layer configuration (LC) from TemplateGenerator file
        gv_current_layer_matrix = eval(df_template_generator.LayerConfig.iloc[model_iteration])
        # Isotropic Hydraulic Conductivity Parameter Configuration (Kxyz-PC) from TemplateGenerator file
        k_matrix = eval(df_template_generator.KValues.iloc[model_iteration])
        # Storativity Parameter Configuration (S-PC) from TemplateGenerator file
        s_matrix = eval(df_template_generator.SValues.iloc[model_iteration])
        # Discrete Features Configuration (FC-C) from TemplateGenerator file
        discrete_features = eval(df_template_generator.DFeatures.iloc[model_iteration])

        # Number of layers in the LC
        gv_current_generator_layer_count = len(gv_current_layer_matrix)
        # Number of parameter iterations in the PC
        parameter_iterations = len(k_matrix)

        for param_model in range(parameter_iterations):
            # Create a model for each discrete feature iteration
            for discrete_feature_iteration, discrete_feature_dict in enumerate(discrete_features.items()):
                # Create the model ID
                gv_current_model = f'{dic_config["MODEL_DESCRIPTION"]}M{model_iteration + 1}I{param_model + 1}F{discrete_feature_iteration + 1}'
                gv_current_pump_rate_id = int((len(PUMP_RATES.keys()) / 2) + 0.5)
                # Get the required parameterization
                hydraulic_conductivity_xyz = k_matrix[str(param_model)]
                storage_param = s_matrix[str(param_model)]
                # Starting point control
                if (model_iteration+1) == SP_MODEL_NUMBER and (param_model+1) <= SP_MODEL_ITERATION:
                    if (param_model+1) == SP_MODEL_ITERATION and (discrete_feature_iteration+1) >= SP_MODEL_DFE:
                        # Create a new model and run it
                        generate_model(hydraulic_conductivity_xyz, storage_param, discrete_feature_dict[1])
                    else:
                        # Skip the files.
                        print(f"Skipping {gv_current_model}: Overwrite by starting point control.")
                elif (model_iteration+1) < SP_MODEL_NUMBER:
                    # Skip the files.
                    print(f"Skipping {gv_current_model}: Overwrite by starting point control.")
                else:
                    # Create a new model and run it
                    generate_model(hydraulic_conductivity_xyz, storage_param, discrete_feature_dict[1])


# This is a script that is meant to be run, not imported
if __name__ == '__main__':
    main()
