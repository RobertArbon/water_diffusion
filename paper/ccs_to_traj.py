import numpy as np
import mdtraj as md
from pyemma.coordinates.clustering import KmeansClustering
from pyemma.msm import MaximumLikelihoodHMSM
import pandas as pd
import pickle
# Creates a PDB of the cluster centers given a PyEmma HMM object


def create_top(n):
    df = pd.DataFrame({'serial': np.arange(n),
                       'name': np.repeat('O', n),
                       'element': np.repeat('O', n),
                       'resSeq': np.arange(n)+1,
                       'resName': np.repeat('HOH', n),
                       'chainID': np.repeat(0, n),
                       'segmentID': np.repeat('A', n)})

    top = md.Topology.from_dataframe(df)
    return top


traj_num = 3
n_segs = 1000
idxs = [902]
nstates = [3]
xyzs = []
for idx in range(n_segs):

    # Load cluster centers
    cluster = KmeansClustering(n_clusters=1).\
        load('msm_10ps/output_{0}/{1}_dtraj_obj.pyemma'.format(traj_num, idx))
    ccs = cluster.cluster_centers_

    # Get xyz into appropriate form
    xyz = np.expand_dims(ccs, 0)
    xyzs.append(xyz)

    if idx in idxs:
        i = np.where(idx==np.array(idxs))[0][0]
        print(i)
        # Load HMM
#         hmm = MaximumLikelihoodHMSM().load('msm_10ps/output_{0}/{1}_hmm_obj.pyemma'.format(traj_num, idx))
        hmm = pickle.load(open('bhmm_mods/{0}_state/traj_{1}-idx_{2}_bhmm.p'.format(nstates[i], traj_num, idxs[i]), 'rb'))
        obs_set = hmm.observable_set
        m_assign = hmm.metastable_assignments
        N = hmm.nstates
        # Save individual trajectory
        print(xyz.shape)
        xyz = xyz[:,obs_set, :]
        print(xyz.shape)
        top = create_top(xyz.shape[1])
        print(top)
        traj = md.Trajectory(xyz=xyz, topology=top)

        traj.save_pdb('msm_10ps/output_{0}/{1}_cluster.pdb'.format(traj_num, idx),
                      bfactors=m_assign/float(N))


xyz = np.concatenate(xyzs, axis=0)

xyz = xyz.reshape(1, -1, 3)
print(xyz.shape)
top = create_top(xyz.shape[1])

print(top)
traj = md.Trajectory(xyz=xyz, topology=top)
print(traj)
traj.unitcell_angles = np.repeat([[90,90,90]], 1, axis=0)
lengths = xyz.reshape(-1,3).max(axis=0) - xyz.reshape(-1,3).min(axis=0)
traj.unitcell_lengths = np.repeat(lengths[np.newaxis, :], 1, axis=0)


traj.save('msm_10ps/output_{0}/all_clusters.pdb'.format(traj_num))
