import numpy as np
from matplotlib.tri import Triangulation
from pylab import *
import matplotlib.pyplot as plt

from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin import Parameter
from tvb_model_reference.view.plot_human import multiview_one, prepare_surface_regions_human


def multiview_one_mod(cortex, hemisphere_left, hemisphere_right, region, data, fig, suptitle='', title='', figsize=(15, 10),
                  **kwds):
    cs = cortex
    vtx = cs.vertices
    print(len(vtx[0]))
    tri = cs.triangles
    rm = cs.region_mapping
    x, y, z = vtx.T

    lh_tri = tri[np.unique(np.concatenate([np.where(rm[tri] == i)[0] for i in hemisphere_left]))]
    lh_vtx = vtx[np.concatenate([np.where(rm == i)[0] for i in hemisphere_left])]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[np.unique(np.concatenate([np.where(rm[tri] == i)[0] for i in hemisphere_right]))]
    rh_vtx = vtx[np.concatenate([np.where(rm == i)[0] for i in hemisphere_right])]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T

    data = np.zeros_like(data)
    if type(region) == list:
        for r in region:
            data[rm == r] = 10 * np.random.rand()  # This sets the value that will be plotted.
            # Here is where we will want to change the value according to the correlation with the seed region
            # Also add a special color to the seed region like yellow or green. Then red and blue using z score.

    else:
        data[rm == region] = 10.0

    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[argsort(tz)]),
    }

    def plotview(i, j, k, viewkey, z=None, zlim=None, zthresh=None, suptitle='', shaded=True, cmap=plt.cm.OrRd,
                 viewlabel=False):
        v = views[viewkey]
        ax = subplot(i, j, k)
        if z is None:
            z = rand(v.x.shape[0])
        if not viewlabel:
            axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.13}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        if zlim:
            tc.set_clim(vmin=-zlim, vmax=zlim)
        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=24)
        if viewlabel:
            xlabel(viewkey)
        return tc

    plotview(2, 3, 1, 'lh-lateral', data, **kwds)
    plotview(2, 3, 4, 'lh-medial', data, **kwds)
    plotview(2, 3, 3, 'rh-lateral', data, **kwds)
    plotview(2, 3, 6, 'rh-medial', data, **kwds)
    plotview(1, 3, 2, 'both-superior', data, suptitle=suptitle, **kwds)

    subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)

    if title:
        plt.gcf().suptitle(title)

parameters = Parameter()

# Prepare the elements that we will need to find the interesting nodes
cortex, conn, hem_left, hem_right = prepare_surface_regions_human(parameters,
                                                                  conn_filename='Connectivity.zip',
                                                                  zip_filename='Surface_Cortex.zip',
                                                                  region_map_filename='RegionMapping.txt')
the_data = np.zeros((cortex.region_mapping_data.array_data.shape[0],))

DMN_regions = [28, 29, 52, 53,  # mPFC
          50, 51, 20, 21]   # precuneus and posterior cingulate (seems large)

title = 'closest to DMN in parcellation'  # We are clearly missing the angular gyri
multiview_one_mod(cortex, hem_left, hem_right,
              DMN_regions, the_data, plt.figure(), suptitle='',
              title=title, figsize=(8, 8), shaded=False)
plt.show()
