""" This file will contain the functions that will process and obtain images of the HPC parameter sweep results."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tvb_model_reference.src.nuu_tools_simulation_human as tools

from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
from tvb_model_reference.view.plot_human import multiview_one, prepare_surface_regions_human
from analyses import *


def batch_files(results_folder, batches_folder, batch_size=100, n_cols=45):
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

    for batch in range(N_batches):
        if batch != N_batches - 1:  # Not last batch -> size of batch = batch_size
            file_batch = np.empty((batch_size, n_cols))
            for ii, file in enumerate(files[batch * batch_size: batch_size * (batch + 1)]):
                file_batch[ii, :] = np.load(results_folder + file)
        else:  # Last batch -> size of batch = Rem
            file_batch = np.empty((Rem, n_cols))
            for ii, file in enumerate(files[batch * batch_size: -1]):
                file_batch[ii, :] = np.load(results_folder + file)

        name_batch = batches_folder + '/batch' + str(batch) + '.npy'
        np.save(name_batch, file_batch)


dict_params = {'a': 0, 'b_e': 1, 'E_L_i': 2, 'E_L_e': 3, 'T': 4}

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
                'ratio_zscore_dmn_inh': 44, 'ratio_AI_exc': 45}


def load_metric_sweeps(name_metric, results_folder, steps=15):
    """Function that will merge the multiple .npy files in the results' folder into less, larger, .npy files.

    Parameters
    ----------
    name_metric: str or list
        String or list of strings representing the metric/metrics we want to obtain. Names can be found in dict_metrics
        in processing_results.py

    results_folder: str
        Path to the folder containing the results

    steps: int
        Number of steps taken for each parameter in the parameter sweep.

    Returns
    -------
    pars_metric: ndarray
        A (steps**5, 6) array containing the value of the metric for all the combinations in the parameter sweep.
        Columns 0 to 4 correspond to the parameters. Column 5 and onwards contain the values of the metrics. Same order
        as the name_metric list in case that more than one metric is retrieved.
    """

    files = os.listdir(results_folder)
    if type(name_metric) is str:
        metric_idx = dict_metrics[name_metric]
        pars_metric = np.empty((steps ** 5, 6))  # steps**5 rows, 5 params + 1 metric cols

    elif type(name_metric) is list:
        metric_idx = []
        for name in name_metric:
            metric_idx.append(dict_metrics[name])
        pars_metric = np.empty((steps ** 5, 5 + len(name_metric)))  # steps**5 rows, 5 params + 1 metric cols
    else:
        raise ValueError("Invalid name_metric type")

    running_idx = 0
    ii = 0
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

            ii += 1

    return pars_metric


ranges_params = {'a': (0, 0.5), 'b_e': (0, 120), 'E_L_e': (-80, -60), 'E_L_i': (-80, -60), 'T': (5, 40)}


def find_closest_val(param_name, desired_value, steps):
    """Returns the closest value of param_name to desired_value in the parameter sweep."""
    range_min, range_max = ranges_params[param_name]
    sweep = np.linspace(range_min, range_max, steps)
    idx = (np.abs(sweep - desired_value)).argmin()
    return sweep[idx]


def metric_for_pairs(params_metric_array, name_metric,
                     sweep_params, fixed_params, steps=2, do_plot=False, fig=None, ax=None):
    """Main idea. Fix three parameters using the fixed params dictionary which will contain the three values that
    we wiill not sweep as keys and their desired fixed values as values of dict.

    The value will be fixed to the closest value of the parameter sweep.

    Then I will try to find to extract the values and params of the parameter sweep and build a matrix with that.

    Ideally, matrix is ordered from the ground up.
    """

    # First step, indexing the array to obtain only the values of params_metrics_array that contain the fixed points
    closest_to_desired = {}  # A dictionary containing {fixed_parameter_key: closest_value in sweep}
    for fixed_param in fixed_params:
        desired_value = fixed_params[fixed_param]
        closest_to_desired[fixed_param] = find_closest_val(fixed_param, desired_value, steps)

    # First we obtain the indexes of those rows that contain the fixed params.
    for i, fixed_param in enumerate(fixed_params):
        fix_par_idx_in_arr = dict_params[fixed_param]
        if i == 0:
            bool_idxs = params_metric_array[:, fix_par_idx_in_arr] == closest_to_desired[fixed_param]
        else:
            bool_idxs = (params_metric_array[:, fix_par_idx_in_arr] == closest_to_desired[fixed_param]) & bool_idxs

    # Now we have a new array containing only those rows where the sweep_params change.
    new_array = params_metric_array[bool_idxs, :]

    # We now have to find a way to build the matrix
    # It might not be that hard, build two vectors with linspace, look at the matrix and assign them.
    par_x = sweep_params[0]
    par_y = sweep_params[1]
    sweep_par_x = np.linspace(ranges_params[par_x][0], ranges_params[par_x][1], steps)
    sweep_par_y = np.linspace(ranges_params[par_y][0], ranges_params[par_y][1], steps)

    plot_matrix = np.empty((steps, steps))

    par_x_idx_in_arr = dict_params[par_x]
    par_y_idx_in_arr = dict_params[par_y]
    for id_x, val_x in enumerate(sweep_par_x):
        for id_y, val_y in enumerate(sweep_par_y):
            idx_row_array = (new_array[:, par_x_idx_in_arr] == val_x) & (new_array[:, par_y_idx_in_arr] == val_y)
            plot_matrix[- (id_y + 1), id_x] = new_array[idx_row_array, -1][0]

    if do_plot:
        if type(ax) is None or type(fig) is None:
            raise ValueError("Please provide an ax AND figure object for the plot")
        else:
            # Make a nice title
            list_fix = [(fixed_param + ' = ' + str(closest_to_desired[fixed_param])) for fixed_param in fixed_params]
            title = name_metric + ' for ' + ', '.join(list_fix)

            # plot the image and manage the axis
            im = ax.imshow(plot_matrix)
            # We don't want too many ticks in our plot. See for definitive results if it works well.
            if steps > 8:
                ax.set_xticks(range(0, steps, 2))
                ax.set_yticks(range(0, steps, 2))
                ax.set_xticklabels(map(str, sweep_par_x[::2]))
                ax.set_yticklabels(map(str, np.flip(sweep_par_y[::2])))
            else:
                ax.set_xticks(range(steps))
                ax.set_yticks(range(steps))
                ax.set_xticklabels(map(str, sweep_par_x))
                ax.set_yticklabels(map(str, np.flip(sweep_par_y)))

            # Colorbar
            ax.set(xlabel=par_x, ylabel=par_y, title=title)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        return plot_matrix, ax
    else:
        return plot_matrix


def plot_multiple_metrics(metrics, results_folder, params_sweep, fixed_params, steps):

    subplot_width = 5
    subplot_height = 5
    n_plots = len(metrics)
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
        mat, axes[plot] = metric_for_pairs(params_metric, metric, params_sweep, fixed_params,
                                           steps=steps, do_plot=True, fig=fig, ax=axes[plot])

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    results_folder = './results/'
    results_folder = '/media/master/Nuevo vol/Internship/Data/hpc_tvbadex/results/'
    batches_folder = '/home/master/Desktop/tests_MPI/results_batches/'
    batches_folder = './results_batches/'
    n_cols = len(dict_params.keys()) + len(dict_metrics.keys())
    batch_files(results_folder, batches_folder, batch_size=11, n_cols=n_cols)
    fixed_params = {'a': 0.25, 'b_e': 65, 'T': 30}
    params_sweep = ('E_L_e', 'E_L_i')
    steps = 2
    metrics = ['mean_FR_e', 'mean_FR_i', 'slope_PSD_e', 'score_PSD_e', 'slope_PSD_i',
               'score_PSD_i', 'ratio_zscore_dmn_inh', 'ratio_AI_exc']
    fig = plot_multiple_metrics(metrics, batches_folder, params_sweep, fixed_params, steps)
    plt.show()
    fig.savefig('test.png')
