import sys
import time
import h5py
import os

import numpy as np
import numpy.ma as ma

from mpi4py import MPI
from itertools import product

import LVAR_final as lvar  # must be imported for lvarc
import LVAR_calculations_final as lvarc


# dissimilarity and likelihood clustering
def compute_master_theta(models, windows, tseries):
    master_tseries = []
    for window_idx in models:
        window = windows[window_idx]
        t0, tf = window
        ts = tseries[t0:tf]
        master_tseries.append(ts)
        master_tseries.append([np.nan]*ts.shape[1])
    master_tseries = ma.masked_invalid(ma.vstack(master_tseries))
    master_theta, eps = lvarc.get_theta_masked(master_tseries)
    return master_theta


def likelihood_distance(models, windows, tseries, thetas):
    master_theta = compute_master_theta(models, windows, tseries)
    distances = []
    for model_idx in models:
        theta = thetas[model_idx]
        window = windows[model_idx]
        t0, tf = window
        ts = ma.masked_invalid(tseries[t0:tf])
        theta_here, eps = lvarc.get_theta(ts)
        distances.append(lvarc.loglik_mvn_masked(theta, ts) -
                         lvarc.loglik_mvn_masked(master_theta, ts))
    return np.sum(distances)


if __name__ == '__main__':

    # set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    node = MPI.Get_processor_name()

    # start the clock
    t0 = time.time()
    if rank == 0:
        print("starting dissim matrix script")

    # data directories
    DATA_DIR = './data/output/guppy_ts_sigmoid/'
    OUTPUT_DIR = './data/output/guppy_ts_sigmoid/'

    # file names
    TSERIES = 'with_sigmoid_x_new.npy'
    win_seg_file = 'windows.npy'
    thetas_file = 'thetas.npy'

    comm.Barrier()

    buf2_size = None
    buf_len_models = None
    all_models = None
    if rank == 0:
        print('rank 0 starting for model assignment')
        t1 = time.time()
        windows = np.load(''.join([OUTPUT_DIR, win_seg_file]))
        len_models = len(windows)

        buf2_size = np.empty(1, dtype=np.int64)
        buf2_size[:] = int(len_models/size) + 1  # run
        # buf2_size[:] = 32/size  # debug
        print('models per rank: {}, with size: {}'.format(buf2_size[0], size))

        buf_len_models = np.empty(1, dtype=np.int64)
        buf_len_models[:] = len_models  # run
        # buf_len_models[:] = 32  # debug

        t2 = time.time()
        print('setting up models took {}s, total processing time {}s'.format(
              t2-t1, t2-t0))

    else:
        buf2_size = np.empty(1, dtype=np.int64)
        buf_len_models = np.empty(1, dtype=np.int64)
        windows = np.load(''.join([OUTPUT_DIR, win_seg_file]))

    sys.stdout.flush()
    comm.Barrier()

    # cast models
    comm.Bcast(buf_len_models, root=0)
    comm.Bcast(buf2_size, root=0)

    node_output_DIR = ''.join([
        OUTPUT_DIR,
        'matrix_cals/',
        '{0:05d}'.format(rank),
        '/'
        ])

    if not os.path.exists(node_output_DIR):
        try:
            os.makedirs(node_output_DIR)
        except FileExistsError:
            pass

    strt = buf2_size[0] * rank
    if (buf2_size[0] * rank + buf2_size[0]) < buf_len_models[0]:
        stop = buf2_size[0] * rank + buf2_size[0]
    else:
        stop = buf_len_models[0]

    filename = ''.join([
        node_output_DIR,
        str(strt),
        '-',
        str(stop),
        '_dissim.hdf5'
        ])

    print('creating file')
    sys.stdout.flush()
    if stop-strt > 0:
        # first pass start from beginning
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset(
                'data',
                (stop-strt, buf_len_models[0]),
                dtype='f8',
                )
            ITER = product(
                [i for i in np.arange(strt, stop)],
                [j for j in np.arange(0, buf_len_models[0]-1)]
                )

        # second pass start from end
        # with h5py.File(filename, 'a') as f:
        #     dset = f['data']
        #     ITER = product(
        #         [i for i in np.arange(strt, stop)[::-1]],
        #         [j for j in np.arange(0, buf_len_models[0]-1)[::-1]]
        #         )

            # load datasets
            original = ma.masked_array(np.load(''.join([DATA_DIR, TSERIES])))
            eigen_dim = original.shape[1]
            padding = (-len(original)) % size
            eigenmodes = ma.zeros(
                    (original.shape[0]+padding, eigen_dim))
            eigenmodes[:-padding, :] = original
            sample_tseries = eigenmodes
            sample_tseries = ma.masked_invalid(sample_tseries)
            thetas = np.load(''.join([OUTPUT_DIR, thetas_file]))
            windows = np.load(''.join([OUTPUT_DIR, win_seg_file]))
            print('starting iteration')
            sys.stdout.flush()
            while True:
                try:
                    combo = next(ITER)
                    idx1 = combo[0]
                    idx2 = combo[1]
                    models_ = [idx1, idx2]
                    ld = \
                        likelihood_distance(
                            models_, windows,
                            sample_tseries,
                            thetas
                            )
                    dset[idx1-strt, idx2] = ld
                    f.flush()
                except StopIteration:
                    print('iteration finished on node {} rank {}'.format(node, rank))
                    break

    sys.stdout.flush()
    comm.Barrier()

    # filename = ''.join([
    #     OUTPUT_DIR,
    #     'dissim_mat_pairs.hdf5'
    # ])
    # f = h5py.File(
    #     filename,
    #     'w',
    #     driver='mpio',
    #     comm=MPI.COMM_WORLD)

    # dset = f.create_dataset(  # h5py
    #     'test',  # h5py
    #     shape=(buf_len_models[0], buf_len_models[0]),  # h5py
    #     maxshape=(buf_len_models[0], buf_len_models[0]),  # h5py
    #     dtype='f8',  # h5py
    #     chunks=True,
    #     )  # h5py

    # print(buf_len_models[0], buf_len_models[0])
    # f.atomic = False

    # strt = buf2_size[0] * rank
    # if (buf2_size[0] * rank + buf2_size[0]) < buf_len_models[0]:
    #     stop = buf2_size[0] * rank + buf2_size[0]
    # else:
    #     stop = buf_len_models[0]

    # ITER = product(
    #     [i for i in np.arange(strt, stop)],
    #     [j for j in np.arange(0, buf_len_models[0])]
    #     )

    # # load datasets
    # original = ma.masked_array(np.load(''.join([DATA_DIR, TSERIES])))

    # eigen_dim = original.shape[1]
    # padding = int(original.shape[0] / (32-1)) - (original.shape[0] % (32-1))

    # eigenmodes = ma.zeros(
    #         (original.shape[0]+padding, eigen_dim))
    # eigenmodes[:-padding, :] = original

    # sample_tseries = eigenmodes

    # sample_tseries = ma.masked_invalid(sample_tseries)
    # thetas = np.load(''.join([OUTPUT_DIR, thetas_file]))
    # windows = np.load(''.join([OUTPUT_DIR, win_seg_file]))

    # while True:
    #     try:
    #         combo = next(ITER)
    #         idx1 = combo[0]
    #         idx2 = combo[1]
    #         if rank == 0:
    #             models_ = [idx1, idx2]
    #         else:
    #             models_ = [idx1, idx2]
    #         ld = \
    #             likelihood_distance(
    #                 models_, windows,
    #                 sample_tseries,
    #                 thetas
    #                 )
    #         with dset.collective:
    #             dset[idx1, idx2] = ld

    #     except StopIteration:
    #         print(rank, 'stopping iteration')
    #         break
    # sys.stdout.flush()
    # comm.Barrier()

    # f.close()

    # sys.stdout.flush()
    # comm.Barrier()
