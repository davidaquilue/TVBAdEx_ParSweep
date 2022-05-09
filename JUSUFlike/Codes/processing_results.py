""" This file will contain the functions that will process and obtain images of the HPC parameter sweep results."""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


dict_params = {'a': 0, 'b_e': 1, 'E_L_i': 2, 'E_L_e': 3, 'T': 4}

ranges_params = {'a': (0, 0.5), 'b_e': (0, 120), 'E_L_e': (-80, -60), 'E_L_i': (-80, -60), 'T': (5, 40)}

dict_metrics = {'mean_FR_e': 5, 'mean_FR_i': 23, 'std_FR_e': 6, 'std_FR_i': 24,
                'mean_FC_e': 7, 'mean_FC_i': 25, 'mean_PLI_e': 8, 'mean_PLI_i': 26,
                'mean_up_e': 9, 'mean_up_i': 27, 'mean_down_e': 10, 'mean_down_i': 28,
                'max_FR_e': 11, 'max_FR_i': 29, 'fmax_amp_e': 12, 'pmax_amp_e': 13,
                'fmax_amp_i': 30, 'pmax_amp_i': 31, 'fmax_prom_e': 14, 'pmax_prom_e': 15,
                'fmax_prom_i': 32, 'pmax_prom_i': 33, 'slope_PSD_e': 16, 'score_PSD_e': 17,
                'slope_PSD_i': 34, 'score_PSD_i': 35, 'delta_rel_p_e': 18, 'theta_rel_p_e': 19,
                'alpha_rel_p_e': 20, 'beta_rel_p_e': 21, 'gamma_rel_p_e': 22, 'delta_rel_p_i': 36,
                'theta_rel_p_i': 37, 'alpha_rel_p_i': 38, 'beta_rel_p_i': 39, 'gamma_rel_p_i': 40,
                'ratio_frmean_dmn_exc': 41, 'ratio_zscore_dmn_exc': 42, 'ratio_frmean_dmn_inh': 43,
                'ratio_zscore_dmn_inh': 44, 'ratio_AI_exc': 45, 'corr_FC_SC_e': 46, 'corr_FC_SC_i': 47,
                'corr_FC_tract_e': 48, 'corr_FC_tract_i': 49,'coeff_var_e': 50, 'coeff_var_i':51, 'std_of_means_e': 52, 'std_of_means_i': 53, 'means_of_std_e':54, 'means_of_std_i': 55}


def batch_files(results_folder, batches_folder, batch_size=100, n_cols=56, name_batch='0'):
    """Function that will merge the multiple .npy files in the results' folder into less, larger, .npy files.

    Parameters
    ----------
    results_folder: str
        Path to the folder containing the results

    batches_folder: str
        Path to the folder where the new .npy files will be stored

    batch_size: int
        Number of files that will be merged into the new file

    n_cols: int
        Number of elements/metrics of each .npy file.

    n_chunk: str
        In the case that only one batch is created per function call, we can assign a special name to it with this variable.
    """
    files_in_folder = os.listdir(results_folder)  # Obtain a list with all the files. This might be very large in RAM!
    files = []
    for file_name in files_in_folder:
        if '.npy' in file_name:
            files.append(file_name)
    N_batches = len(files) // batch_size
    Rem = len(files) % batch_size  # Remainder will be useful for a last batch with less files in it.
    if Rem != 0:
        N_batches += 1

    # We will check if batches_folder contains other batches and add new batches to it.
    # CAREFUL WHEN WE WANT TO RE-OBTAIN THE BATCHES, THOUGH.
    batches_in_folder = len(os.listdir(batches_folder))
    for batch in range(N_batches):
        if batch != N_batches - 1:  # Not last batch -> size of batch = batch_size
            file_batch = np.empty((batch_size, n_cols))
            for ii, file in enumerate(files[batch * batch_size: batch_size * (batch + 1)]):
                file_batch[ii, :] = np.load(results_folder + file)
        else:  # Last batch -> size of batch = Rem
            if Rem == 0:  # If exact division, last batch same size as others
                file_batch = np.empty((batch_size, n_cols))
            else:  # If not exact division, last batch size is the Rem
                file_batch = np.empty((Rem, n_cols))
            for ii, file in enumerate(files[batch * batch_size: -1]):
                file_batch[ii, :] = np.load(results_folder + file)
        if N_batches != 1:
            name_batch = batches_folder + '/batch' + str(batches_in_folder + batch) + '.npy'
        else:
            name_batch = batches_folder + '/batch' + name_batch + '.npy'
        np.save(name_batch, file_batch)


def load_whole_sweep(results_folder, steps):
    """Loads the whole parameter sweep to memory."""
    warnings.warn("ATTENTION: This function will be too expensive with the whole parameter sweep")
    
    files = os.listdir(results_folder)

    parameter_sweep = np.empty((steps ** 5, len(dict_params) + len(dict_metrics)))

    running_idx = 0
    for file in files:
        if '.npy' in file:
            aux_file = np.load(results_folder + file)
            rows_aux = aux_file.shape[0]

            parameter_sweep[running_idx: running_idx + rows_aux, :] = aux_file
            
            running_idx += rows_aux
    
    return parameter_sweep


def load_metric_sweeps(name_metric, results_folder, steps=15):
    """Returns the values of one or multiple metrics for all the parameter space exploration.

    Parameters
    ----------
    name_metric: str or list
        String or list of strings representing the metric/metrics we want to obtain. Names can be found in dict_metrics
        in processing_results.py

    results_folder: str
        Path to the folder containing the results.

    steps: int
        Number of steps taken for each parameter in the parameter sweep.

    Returns
    -------
    pars_metric: ndarray
        A (steps**5, 5 + len(name_metric)) array containing the value of the metric for all the combinations in the
        parameter sweep. Columns 0 to 4 correspond to the parameters. Column 5 and onwards contain the values of the
        metrics. Same order as the name_metric list in case that more than one metric is retrieved.
    """

    files = os.listdir(results_folder)
    if type(name_metric) is str:
        metric_idx = dict_metrics[name_metric]
        pars_metric = np.empty((steps ** 5, 6))  # steps**5 rows, 5 params + 1 metric cols

    elif type(name_metric) is list:
        metric_idx = []
        for name in name_metric:
            metric_idx.append(dict_metrics[name])
        pars_metric = np.empty((steps ** 5, 5 + len(name_metric)))  # steps**5 rows, 5 params + N metric cols
    else:
        raise ValueError("Invalid name_metric type")

    running_idx = 0
    for file in files:
        if '.npy' in file:
            aux_file = np.load(results_folder + file)
            rows_aux = aux_file.shape[0]

            pars_metric[running_idx: running_idx + rows_aux, 0:5] = aux_file[:, 0:5]
            if type(name_metric) is str:
                pars_metric[running_idx: running_idx + rows_aux, 5] = aux_file[:, metric_idx]
            else:
                pars_metric[running_idx: running_idx + rows_aux, 5: 5 + len(name_metric)] = aux_file[:, metric_idx]
            running_idx += rows_aux

    return pars_metric


def find_closest_val(param_name, desired_value, steps):
    """Returns the closest value of param_name to desired_value in the parameter sweep."""
    range_min, range_max = ranges_params[param_name]
    sweep = np.round(np.linspace(range_min, range_max, steps), 3)
    idx = (np.abs(sweep - desired_value)).argmin()
    return sweep[idx]


def metric_for_pairs(params_metric_array, name_metric, sweep_params, fixed_params,
                     steps=2, do_plot=False, fig=None, ax=None, imshow_range=None):
    """Obtains the values of one metric when varying two of the five parameters while fixing the other three.
    One can select to also obtain an imshow of the matrix.
    The fixed parameters can be set to trivial values and the function will find the closest parameter values
    that have been used in the parameter sweep to obtain the plot.

    Parameters
    ----------
    params_metric_array: ndarray (steps**5, 6)
        Array containing all the values of the parameter sweep of the 5 parameters + 1 metric in the last column.
    
    name_metric: str
        Name of the metric from which we want to obtain the matrix. It has to be included in dict_metrics

    sweep_params: list, tuple
        Contains the two strings of the parameters for which we will observe the change in the metric value
    
    fixed_params: dict
        Dictionary with strings and desired fixed values of other 3 parameters.

    steps: int
        Number of different values obtained for each parameter in the original Parameter Sweep in HPC.

    do_plot: bool
        Whether return an ax with the imshow plotted.
    
    fig: matplotlib.pyplot.figure object
        Figure in which we want to plot the imshow if do_plot
    
    ax: matplotlib.pyplot.axis object
        Axis object where we want to plot the imshow if do_plot

    imshow_range: tuple
        Containing (vmin, vmax) for the imshow. If none, set to (None, None) and the min and max of the matrix will be
        used as the limits of the color range.

    Returns
    ----------
    mat: ndarray 
        Array of shape (steps, steps) containing values of desired metric when changing parameters in sweep_params.

     ax: matplotlib.pyplot.axis object
        Axis object where with the imshow plotted if do_plot
    """
    # Handling errors in the inputs
    if len(fixed_params.keys()) != 3:
        raise ValueError("Three parameters must be fixed.")
    if len(sweep_params) != 2:
        raise ValueError("Sweep can only be performed over two parameters.")
    if len([i for i in fixed_params.keys() if i in sweep_params]) != 0:
        raise ValueError("Sweep and Fixed parameters cannot coincide.")
    
    closest_to_desired = {}  # A dictionary containing {fixed_parameter_key: closest_value in sweep}
    for fixed_param in fixed_params:
        desired_value = fixed_params[fixed_param]
        closest_to_desired[fixed_param] = find_closest_val(fixed_param, desired_value, steps)
    
    for i, fixed_param in enumerate(fixed_params):  # Obtain the indexes of those rows that contain the fixed params.
        fix_par_idx_in_arr = dict_params[fixed_param]
        if i == 0:
            bool_idxs = params_metric_array[:, fix_par_idx_in_arr] == closest_to_desired[fixed_param]
        else:
            bool_idxs = (params_metric_array[:, fix_par_idx_in_arr] == closest_to_desired[fixed_param]) & bool_idxs

    # New array containing only those rows where the sweep_params change.
    new_array = params_metric_array[bool_idxs, :]

    # Building the matrix that will be used for the imshow. Increasing values of parameters in x and y dir.
    par_x = sweep_params[0]
    par_y = sweep_params[1]
    sweep_par_x = np.round(np.linspace(ranges_params[par_x][0], ranges_params[par_x][1], steps), 3)
    sweep_par_y = np.round(np.linspace(ranges_params[par_y][0], ranges_params[par_y][1], steps), 3)

    plot_matrix = np.empty((steps, steps))

    par_x_idx_in_arr = dict_params[par_x]
    par_y_idx_in_arr = dict_params[par_y]
    for id_x, val_x in enumerate(sweep_par_x):  # Assign to each element the corresponding metric value
        for id_y, val_y in enumerate(sweep_par_y):
            idx_row_array = (new_array[:, par_x_idx_in_arr] == val_x) & (new_array[:, par_y_idx_in_arr] == val_y)
            plot_matrix[- (id_y + 1), id_x] = new_array[idx_row_array, -1][0]

    if do_plot:
        if type(ax) is None or type(fig) is None:
            raise ValueError("Please provide an ax AND figure object for the plot")
        if imshow_range is None:
            imshow_range = (None, None)
        else:
            # Make a nice title
            list_fix = [(fixed_param + ' = ' + str(closest_to_desired[fixed_param])) for fixed_param in fixed_params]
            title = name_metric + ' for ' + ', '.join(list_fix)
            if len(title) > 40:
                title = name_metric + ' for ' + ', \n'.join(list_fix)

            # plot the image and manage the axis
            im = ax.imshow(plot_matrix, vmin=imshow_range[0], vmax=imshow_range[1])
            # We don't want too many ticks in our plot. See for definitive results if it works well.
            if steps > 8:
                ax.set_xticks(range(0, steps, 2))
                ax.set_yticks(range(0, steps, 2))
                ax.set_xticklabels(map(str, sweep_par_x[::2]))
                ax.set_yticklabels(map(str, np.flip(sweep_par_y[::2])))
            else:
                ax.set_xticks(range(steps))
                ax.set_yticks(range(steps))
                # Shorten lenghts in some cases
                xticklabels = [str_tick[0:6] for str_tick in map(str, sweep_par_x)]
                yticklabels = [str_tick[0:6] for str_tick in map(str, np.flip(sweep_par_y))]
                ax.set_xticklabels(xticklabels)
                ax.set_yticklabels(yticklabels)

            # Colorbar
            ax.set(xlabel=par_x, ylabel=par_y, title=title)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        return plot_matrix, ax
    else:
        return plot_matrix


def plot_multiple_metrics(metrics, results_folder, sweep_params, fixed_params, steps, imshow_ranges=None):
    """Returns a figure with the different imshows of the selected metrics.

    Parameters
    ----------
    metrics: list, tuple
        Contains the names of all the metrics for which we want to obtain an imshow panel.

    results_folder: str
        Directory to where the function has to go look for the results of the parameter sweep.

    sweep_params: list, tuple
        Contains the two strings of the parameters for which we will observe the change in the metric value

    fixed_params: dict
        Dictionary with strings and desired fixed values of other 3 parameters.

    steps: int
        Number of different values obtained for each parameter in the original Parameter Sweep in HPC.

    imshow_ranges: list of tuples
        Contains the range (vmin, vmax) of the imshow for each metric. If we want to let the range free for one metric,
        we set (None, None) in its corresponding index in the list.

    Returns
    ----------
    fig: matplotlib.pyplot.figure object
        Figure in which the len(metrics) panels have been plotted
    """
    subplot_width = 6  # matplotlib units (I think inches)
    subplot_height = 6
    n_plots = len(metrics)

    if imshow_ranges is None:
        imshow_ranges = []
        for i in range(n_plots):
            imshow_ranges.append((None, None))

    if n_plots < 3:
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * subplot_width, subplot_height))
        axes = axes.flatten()
    else:
        n_rows = n_plots // 3
        fig = plt.figure(figsize=(3 * subplot_width, subplot_height * n_rows))
        n_cols_last = n_plots % 3
        if n_cols_last != 0:
            n_rows += 1
        axes = []
        for n in range(n_rows):
            sub2grid_shape = (n_rows, 2 * 3)  # Each plot occupies two rows
            if n == n_rows - 1 and n_cols_last != 0:
                if n_cols_last == 1:
                    axes.append(plt.subplot2grid(shape=sub2grid_shape, loc=(n, 2), fig=fig, colspan=2))
                elif n_cols_last == 2:
                    axes.append(plt.subplot2grid(shape=sub2grid_shape, loc=(n, 1), fig=fig, colspan=2))
                    axes.append(plt.subplot2grid(shape=sub2grid_shape, loc=(n, 3), fig=fig, colspan=2))
            else:  # For every row that is not the last
                for id_plot in range(3):  # 3 plots per row
                    axes.append(plt.subplot2grid(shape=sub2grid_shape, loc=(n, 2 * id_plot), fig=fig, colspan=2))

    for plot, metric in enumerate(metrics):
        params_metric = load_metric_sweeps(metric, results_folder, steps=steps)
        mat, axes[plot] = metric_for_pairs(params_metric, metric, sweep_params, fixed_params, steps=steps,
                                           do_plot=True, fig=fig, ax=axes[plot], imshow_range=imshow_ranges[plot])

    fig.tight_layout()
    return fig


def params_of_max_metric(metric, results_folder, steps, avoid_bp=True, verbose=True):
    """ Returns those combinations of parameters that share the maximum value of one metric. Also returns the maximum
    value of the metric and the row indexes of said combinations.

    Parameters
    ----------
    metric: str
        Code name of the metric for which we search the maximum value and the parameters that result in this value.

    results_folder: str
        Path to the directory containing all the results of the parameter sweep.

    steps: int
        Number of different values obtained for each parameter in the original Parameter Sweep in HPC.

    avoid_bp: bool
        If True, the search of parameters that result in maximum value of the metric will be done on the subspace of the
        parameter space where the broken point has not been reached.

    verbose: bool
        Whether we want to print the parameter values that result in the maximum metric value.

    Returns
    ----------
    max_value_metric: float
        The maximum value of the selected metric

    pars_where_max: numpy.ndarray
        A (K, 5) array containing the K different combinations of parameters that result in max_value_metric

    idxes: ndarray
        A (K, ) array containing the row index of the parameter values resulting in max_value_metric
    """
    pars_metric = load_metric_sweeps(metric, results_folder, steps)

    if avoid_bp:
        max_FRs = load_metric_sweeps('max_FR_e', results_folder, steps)
        no_bps_idxes = max_FRs[:, -1] < 180
        max_value_metric = np.nanmax(pars_metric[no_bps_idxes, -1])
        idxes = np.logical_and(pars_metric[:, -1] == max_value_metric, no_bps_idxes)
    else:
        max_value_metric = np.nanmax(pars_metric[:, -1])
        idxes = np.where(pars_metric[:, -1] == max_value_metric)[0]
    pars_where_max = pars_metric[idxes, 0:5]

    if verbose:
        print(f"Max {metric} = {max_value_metric}")
        if pars_where_max.shape[0] > 50:
            print("Print would be too long")
        else:
            print(f"Sets of values for maximum {metric}")
            for i in range(pars_where_max.shape[0]):
                print_str = ''
                for j, param in enumerate(dict_params):
                    print_str += param + '= ' + str(pars_where_max[i, j]) + ', '
                print(print_str)

    return max_value_metric, pars_where_max, idxes


def check_clean_chunk(indicators_chunk_folder, results_chunk_folder, parsw_chunk, rem_chunk_folder, verbose=True):
    """Function that will check which parameter combinations in parsw_chunk have not been completely simulated (using
    info in indicators_chunk_folder) and will store a rem_chunk array containing those parameter combinations that have
    to be simulated again.

    CAUTION: This function might take a long time to run (~20'). Maybe trying to parallelize it would be more effective.
    However, since it consists of file managing, I don't know how smart it would be to parallelize.

    Parameters
    ----------
    indicators_chunk_folder: str
        Path to the Indicators of the corresponding chunk of parameter space. Information about completed simulations is
        stored in that folder.

    results_chunk_folder: str
        Path to the Results directory of the corresponding chunk of parameter space in the temporary partition of the
        HPC component.

    parsw_chunk: str
        Path to the .npy file containing (N, 5) array, a subset of the whole parameter space, containing N parameter
        combinations. It will also give us the chunk number

    rem_chunk_folder: str
        Path to the directory where one wants to store the rem_chunk array.

    verbose: bool
        Whether we want to print or not the shape of the rem_chunk array.

    Returns
    ----------

    """
    parsweep_data = np.load(parsw_chunk)
    chunk_n = int(parsw_chunk.split('_')[-1][:-4])  # Since it is a .npy path

    # Check if same number of rows in parsweep_data and indicators in folder.
    n_files = 0
    for file in os.scandir(indicators_chunk_folder):
        n_files += 1

    #if n_files == parsweep_data.shape[0]:  # If yes no need to do anything else
    #    print('All simulations correctly computed')

    #else:  # If not, we need to iterate over all the elements in Indicators folder and delete its row in parsweep_data
    for file in os.scandir(indicators_chunk_folder):  # sweep over all Indicator files
        split_str = file.name.split('_')
        a = float(split_str[2])
        b = float(split_str[4])
        E_L_i = float(split_str[6])
        E_L_e = float(split_str[8])
        T = float(split_str[10])

        #print(parsweep_data[:, 0], a, parsweep_data[:, 0] == a)
        #print(parsweep_data[:, 1], b, parsweep_data[:, 1] == b)
        #print(parsweep_data[:, 2], E_L_i, parsweep_data[:, 2] == E_L_i)
        #print(parsweep_data[:, 3], E_L_e, parsweep_data[:, 3] == E_L_e)
        #print(parsweep_data[:, 4], T, parsweep_data[:, 4] == T)
        
        # Now put strange number in the
        a_idx = parsweep_data[:, 0] == a
        b_idx = parsweep_data[:, 1] == b
        E_L_i_idx = parsweep_data[:, 2] == E_L_i
        E_L_e_idx = parsweep_data[:, 3] == E_L_e
        T_idx = parsweep_data[:, 4] == T
        all_idx = np.logical_and.reduce((a_idx, b_idx, E_L_i_idx, E_L_e_idx, T_idx))
        parsweep_data[all_idx, :] = np.nan
        # Once we have filled with nans. We remove those rows that contain a nan value, leaving us with rem_chunk
        rem_chunk = parsweep_data[~np.isnan(parsweep_data).any(axis=1), :]

    if verbose:
        print(f'Number of rows in rem_chunk: {rem_chunk.shape[0]} rows')

    if rem_chunk.shape[0] > 0:
        np.save(rem_chunk_folder + 'chunk_' + str(chunk_n) + '.npy', rem_chunk)  # save array

        # Now, we will check if there are remaining combinations that also appear in the results folder
        # If it is the case, that means that said combinations have not been properly stored, there has been an error
        # and thus, we will want to delete them
        for file in os.scandir(results_chunk_folder):  # Scan the Results folder
            split_str = file.name.split('_')  # Obtain param values of each file
            a = float(split_str[2])
            b = float(split_str[4])
            E_L_i = float(split_str[6])
            E_L_e = float(split_str[8])
            T = float(split_str[10][:-4])

            a_idx = rem_chunk[:, 0] == a  # Indexing to find if parameter values in rem_chunk
            b_idx = rem_chunk[:, 1] == b
            E_L_i_idx = rem_chunk[:, 2] == E_L_i
            E_L_e_idx = rem_chunk[:, 3] == E_L_e
            T_idx = rem_chunk[:, 4] == T
            all_idx = np.logical_and.reduce((a_idx, b_idx, E_L_i_idx, E_L_e_idx, T_idx))
            in_remaining = np.sum(all_idx)  # If parameter combination in rem_chunk, will be 1. Else, 0

            #if in_remaining == 1:  # If in rem_chunk, delete since it means that something went wrong
            #    os.remove(file.path)


if __name__ == '__main__':
    print('Enter what you want to do: \n1. Post-process a chunk \n2. Batch all results \n3. Plot test figure')
    do_these = input()

    # TODO Change all these folders in JUSUF
    if '1' in do_these:
        print('Enter the chunks you want to post-process separated by a comma:')
        chunks_input = input()
        chunks_to_process = [int(i) for i in chunks_input.split(',')]
        for n_chunk in chunks_to_process:
            indicator_folder = '/p/scratch/icei-hbp-2022-0005/Indicators/chunk_' + str(n_chunk) + '/'
            results_folder = '/p/scratch/icei-hbp-2022-0005/Results/chunk_' + str(n_chunk) + '/'
            parsw_chunk = '../Data/data_chunks/chunk_' + str(n_chunk) + '.npy'
            rem_chunk_folder = '../Data/rem_chunks/'
            check_clean_chunk(indicator_folder, results_folder, parsw_chunk, rem_chunk_folder, verbose=True)
            print(f'Chunk {n_chunk} processed.')
        print('All selected chunks have been processed.')

    if '2' in do_these:
        n_chunks = len(os.listdir('/p/scratch/icei-hbp-2022-0005/Results/'))
        for n_chunk in range(n_chunks):
            results_folder = '/p/scratch/Results/chunk_' + str(n_chunk) + '/'
            batches_folder = '../FinalResults/'
            n_cols = len(dict_params.keys()) + len(dict_metrics.keys())
            batch_files(results_folder, batches_folder, batch_size=5, n_cols=n_cols)

    if '3' in do_these:
        metrics = ['mean_FR_e', 'mean_FR_i', 'slope_PSD_e', 'score_PSD_e', 'slope_PSD_i', 'score_PSD_i',
                   'alpha_rel_p_e', 'beta_rel_p_i']
        batches_folder = '../FinalResults/'
        sweep_params = ('E_L_e', 'E_L_i')
        fixed_params = {'a': 0, 'b_e': 120, 'T': 40}
        steps = 2
        fig = plot_multiple_metrics(metrics, batches_folder, sweep_params, fixed_params, steps)
        plt.show()


# Comments on Cleaning the results, managing errors
# Folder system assumed:
# Main folder 1: In Permanent partition / Data partition
#   Contains 3 directories: Codes, Data (contain Remaining Chunks and Original Chunks), Final Results
# Main folder 2: In Temporary partition / Scratch partition
#   Contains 2 directories: Results, Indicators

# We will divide the parameter space into multiple chunks. Remaining Chunks, Original Chunks, Results and Indicators
# folders will have one folder/chunk inside.

# Pipeline (for one chunk): Simulations -> Store metrics in Results + store a COMPLETED.txt in Indicators if everything
# went well -> Simulations finished then load chunk from Data and iterate over Indicators, if in Indicator, delete row
# -> We are left with a Remaining Chunk that we will store in Data -> Now iterate over Results and delete those .npy
# files that are IN Remaining Chunk and in Results (meaning that they were not completed, but it was stored anyway).

# Printing the shape of the Remaining Chunk might be useful to check how many simulations are left in a chunk
