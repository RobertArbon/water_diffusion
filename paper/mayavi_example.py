import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mayavi import mlab
from pyemma.msm import MaximumLikelihoodHMSM

traj_num = 3
idx = 83

# Load the relevant HMM
hmm = MaximumLikelihoodHMSM().load('msm_10ps/output_{0}/{1}_hmm_obj.pyemma'.format(traj_num, idx))
N = hmm.nstates
obs_to_hid = np.argmax(hmm.metastable_memberships, axis=1)
m_assign = hmm.metastable_assignments

# Load the relevant cluster centers
ccs = pickle.load(file=open('traj_{}_cluster_centers.p'.format(traj_num), 'rb'))
ccs_all = np.concatenate(ccs, axis=0)

# Set up colors
cmap_use = 'viridis_r'
cmap = matplotlib.cm.get_cmap(cmap_use, N)
cmap = (cmap(np.linspace(0,1, 256))*255).astype(int)
col = (0.5, 0.5, 0.5)

# Position data
stride=10
traj = ccs_all[::stride, :]
traj_idx = ccs[idx]

# Other globals
scale_fac = 0.05
res = 20
low_op = 0.1
hi_op = 1


def plot(ax=2):
    # Calculate occluding points
    dims = [0, 1, 2]
    dims.remove(ax)

    centroid = np.mean(traj_idx[:, dims], axis=0)
    centroid_dist = np.sqrt(np.sum((traj_idx[:, dims] - centroid)**2, axis=-1))
    dist = np.sqrt(np.sum((traj[:, dims] - centroid)**2, axis=-1))

    r = np.max(centroid_dist)
    occ_idx = np.where(dist < r)[0]
    idx = np.where(dist >= r)[0]

    # Set up figure
    mlab.figure(bgcolor=(1, 1, 1), size=(1024, 768))

    # plot points
    points1 = mlab.points3d(traj_idx[:, 0], traj_idx[:, 1], traj_idx[:, 2],
                            m_assign, scale_factor=scale_fac, resolution=res)
    points1.glyph.scale_mode = 'scale_by_vector'
    points1.mlab_source.dataset.point_data.vectors = np.ones_like(traj_idx)
    points1.module_manager.scalar_lut_manager.lut.table = cmap

    points2 = mlab.points3d(traj[idx, 0], traj[idx, 1], traj[idx, 2],
                            scale_factor=scale_fac, color=col, resolution=res, opacity=hi_op)

    points3 = mlab.points3d(traj[occ_idx, 0], traj[occ_idx, 1], traj[occ_idx, 2],
                            scale_factor=scale_fac, color=col, resolution=res, opacity=low_op)

    # Change camera view
    views = {2: (0, 0), 1: (90, 90), 0: (0, 90)}

    mlab.view(azimuth=views[ax][0], elevation=views[ax][1])

    # Plot bounding box
    extent = (np.min(traj[:, 0]), np.max(traj[:, 0]),
              np.min(traj[:, 1]), np.max(traj[:, 1]),
              np.min(traj[:, 2]), np.max(traj[:, 2]))

    mlab.outline(points3, color=(.7, .7, .7), extent=extent, line_width=5)

plot(ax=1)
mlab.show()

# mlab.savefig('traj_num_{0}-idx_{1}.png'.format(traj_num, idx))




