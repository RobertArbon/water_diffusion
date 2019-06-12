import os
from pyemma.coordinates.clustering import KmeansClustering
import pickle
import numpy as np


from mayavi import mlab

traj_num = 3
root_dir = 'msm_10ps'

data_dir = root_dir+'/output_{}'.format(traj_num)
dtraj_obj_paths = [os.path.join(data_dir, '{}_dtraj_obj.pyemma'.format(i)) for i in range(1000)]

ccs = [KmeansClustering(n_clusters=1).load(x).clustercenters for x in dtraj_obj_paths]
ccs_all = np.concatenate(ccs, axis=0)

pickle.dump(obj=ccs, file = open('traj_{}_cluster_centers.p'.format(traj_num), 'wb'))

