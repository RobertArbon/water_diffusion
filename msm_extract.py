import pyemma.coordinates as coor
import pyemma.msm as msm
import mdtraj as md
import numpy as np
import argparse
import os
import pathos.multiprocessing as mp
import functools as ft
import pickle

def get_data(traj_path, top_path):
    print('Loading trajectory ', traj_path)
    traj = md.load(traj_path, top=top_path)
    return traj


def get_com(xyz):
    return md.compute_center_of_mass(xyz)


def split_trajectory(xyz, dt, chunk):
    """
    splits up the trajectory
    :param xyz: the trajectory
    :param dt: the frame timestep in ps
    :param chunk: size of each chunk in ps
    :return: array of arrays
    """
    n_frames = int(chunk/dt)
    n_splits = xyz.shape[0] // n_frames  # number of windows in trajectory
    spli_traj = np.array_split(xyz, n_splits)
    return spli_traj


def create_output_dir(output):
    """
    creates output directory if there isn't already one
    :param output: path to possibly existing director
    """
    if not os.path.isdir(output):
        os.mkdir(output)
        print('Making directory ', output)
    else:
        print('Directory {} already exists'.format(output))
    return


def cluster(traj):
    assert isinstance(traj, np.ndarray), "trying to cluster a {}".format(type(traj))
    obj = coor.cluster_kmeans(traj, k=int(np.sqrt(traj.shape[0])), max_iter=100, n_jobs=1)
    return obj


def get_dtrajs(dtraj_obj):
    dtrajs = dtraj_obj.dtrajs
    len_dtrajs = len(dtrajs)
    assert len_dtrajs == 1, "len of dtrajs is {}".format(len_dtrajs)
    return dtrajs[0]


def fit_msm_lag(traj, lag):
    """
    fits MSM
    :param traj: a numpy trajectory
    :param lag: the lag time in frames
    :return: pyemma.MaximumLikelihoodMSM object
    """
    assert isinstance(traj, np.ndarray), "trying to fit msms using a {}".format(type(traj))
    return msm.estimate_markov_model([traj], lag=lag)

def ts_variables(msm_obj):
    """
    Estimates the number of metastable states, the largest ts ratio
     and the longest timescale from the gap in the timescales.
    :param msm_obj:
    :return:
    """
    ts = msm_obj.timescales()[:5]
    ts_ratio = ts[:-1]/ts[1:]
    max_ratio = np.max(ts_ratio)
    n_states = np.argmax(ts_ratio)+2
    t2 = ts[0]
    return n_states, max_ratio, t2


def process_traj(traj, idx, out_path, msm_lag):
    # Clustering
    dtraj_obj = cluster(traj)
    dtraj_obj.save(os.path.join(out_path, '{}_dtraj_obj.pyemma'.format(idx)), overwrite=True)
    dtraj = dtraj_obj.dtrajs[0]
    np.save(os.path.join(out_path, '{}_dtraj.npy'.format(idx)), arr=dtraj)

    # MSM
    mm = fit_msm_lag(dtraj, msm_lag)
    mm.save(os.path.join(out_path, '{}_msm_obj.pyemma'.format(idx)), overwrite=True)

    # Timescales properties
    n, max_r, t2 = ts_variables(mm)

    # HMM
    if (max_r > 1.5) & (t2 > mm.lag):
        hmm = mm.coarse_grain(n)
        hmm.save(os.path.join(out_path, '{}_hmm_obj.pyemma'.format(idx)), overwrite=True)

    return

def main(args):

    # get arguments
    traj_path = args.traj
    top_path = args.top
    chunk = args.chunk
    output_path = args.output

    # get data
    traj = get_data(traj_path, top_path)
    dt = traj.timestep

    # Make output dir
    create_output_dir(output_path)

    # msm parameters
    msm_lag = int(args.msm_lag/dt)

    # split data
    com_traj = get_com(traj)
    split_trajs = split_trajectory(com_traj, dt, chunk)
    print('Number of chunks : {0}\n'
          'Frames per chunk : {1}\n'
          'Length of chunk  : {2:4.2} ps'.format(len(split_trajs),
                                                 split_trajs[0].shape[0],
                                                 split_trajs[0].shape[0]*dt))
    # parameters:
    n_trajs = len(split_trajs)
    indexes = np.arange(n_trajs)
    output_paths = np.repeat(output_path, n_trajs)
    msm_lags = np.repeat(msm_lag, n_trajs)

    n_workers = mp.cpu_count()
    with mp.ProcessPool(nodes=n_workers) as pool:
        print('Using {} cpus'.format(n_workers))

        pool.map(process_traj, split_trajs,indexes, output_paths, msm_lags)




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Creates MSMs and HMMs from a water trajectory in chunks.')
    parser.add_argument('traj', type=str, help='path to single trajectory.')
    parser.add_argument('top', type=str, help='path to topology.')
    parser.add_argument('chunk', type=int, help='size of chunk in ps.')
    parser.add_argument('output', type=str, help='output directory name.')
    parser.add_argument('msm_lag', type=int, help="The MSM lag time in ps")
    args = parser.parse_args()
    main(args)
