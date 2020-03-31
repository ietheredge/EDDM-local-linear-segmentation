# segmentations_script_MPI.py
import time
import sys
import gc
from itertools import product
import numpy as np
import numpy.ma as ma
from mpi4py import MPI

import LVAR_final as lvar
import LVAR_calculations_final as lvarc


def load_data(file):
    f = np.load(file)
    skeleton = np.array(f['midline_points'])[:, 1:-1, :]
    return skeleton


# break segment, eigenvalue and theta decomposition
def get_eigs(theta, frameRate):
    c, A, cov = lvarc.decomposed_theta(theta)
    coef = (A-np.identity(theta.shape[1])) * frameRate
    eigvals = np.linalg.eigvals(coef)
    return eigvals


def get_theta(segment_windows, sample_tseries):
    thetas = []
    for seg_w in segment_windows:
        i_0, i_f = seg_w
        window_bw = sample_tseries[i_0:i_f]
        theta, eps = lvarc.get_theta(window_bw, 1)
        thetas.append(np.vstack(theta))
    return thetas


def reconcile_windows(windows):
    size = windows[-1][-1][-1][-1]
    WINDOWS = windows
    for i in range(len(windows)):
        if i == 0:
            pass
        else:
            for j in range(len(windows[i][0])):
                WINDOWS[i][0][j][0] = windows[i][0][j][0] + (size*i)
                WINDOWS[i][0][j][1] = windows[i][0][j][1] + (size*i)
    WIN = []
    for i in range(len(WINDOWS[:][:][:][:])):
        WIN.extend(WINDOWS[:][:][:][i][0])
    return np.array(WIN)


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


def get_ll_mvn_dist(args):
    ts, theta, master_theta = args
    llmt = lvarc.loglik_mvn_masked(theta, ts)
    llmmt = lvarc.loglik_mvn_masked(master_theta, ts)
    return llmt-llmmt


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


def symmetrize(a):
    return (a + a.T)/2 - np.diag(a.diagonal())


if __name__ == '__main__':
    # start the clock
    t0 = time.time()
    print('starting segmentation and distance calculation script')
    # locate directory with data
    DATA_DIR = './data/output/guppy_ts_sigmoid/'
    OUTPUT_DIR = './data/output/guppy_ts_sigmoid/'
    TSERIES = 'with_sigmoid_x_new.npy'

    # constants

    frameRate = 140
    N = 1000  # number of simulations in the likelihood distribution
    per = 98.75
    w0 = 12
    step_fraction = .1
    i = w0
    w = []
    while i < np.inf:
        w.append(i)
        step = int(i*step_fraction)
        if int(i*step_fraction) > w0:
            break
        if step < 1:
            step = 1
        i += step

    # set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

   # scatter data
    send_buf = None
    if rank == 0:
        print('rank {}'.format(rank))
        original = ma.masked_array(np.load(''.join([DATA_DIR, TSERIES])))
        
        eigen_dim = original.shape[1]
        padding = (-len(original))%size
        # padding = int(original.shape[0] / (size-1)) - (original.shape[0] % (size-1))

        eigenmodes = ma.zeros(
            (original.shape[0]+padding, eigen_dim))
        eigenmodes[:-padding, :] = original

        print(eigenmodes.shape, padding, original.shape)

        send_buf = ma.empty([size, int(eigenmodes.shape[0]/size), eigen_dim],
                            dtype=np.float64)

        send_buf[:, :, :] = ma.reshape(eigenmodes,
                                       [size, int(eigenmodes.shape[0]/size), eigen_dim])

        buf1_size = np.empty(2, dtype=np.int32)
        buf1_size[:] = np.array([int(len(eigenmodes)/size),eigen_dim])
        print('preparing parameters took {}, with size {}'
              .format(time.time()-t0, send_buf.shape))
    else:
        buf1_size = np.empty(2, dtype=np.int32)

    sys.stdout.flush()
    comm.Barrier()
    comm.Bcast(buf1_size, root=0)
    recv_buf = ma.empty([buf1_size[0], buf1_size[1]], dtype=np.float64)

    comm.Scatter(send_buf, recv_buf, root=0)
    sample_time_series = ma.masked_invalid(recv_buf)
    del recv_buf

    # calculate break segments
    breaks_segments = lvar.change_point(w, N, per, sample_time_series, 20)
    windows_segment, segment = breaks_segments

    # calculate thetas
    thetas = []
    for idx, seg in enumerate(segment):
        segment_windows = np.copy(windows_segment[idx])
        segments_windows = list(segment_windows)
        thetas.append(get_theta(segment_windows, sample_time_series))
    thetas = np.concatenate(thetas)
    all_eigs = []
    for theta in thetas:
        all_eigs.append(get_eigs(theta, frameRate))

    # gather data from nodes
    comm.Barrier()
    thetas = comm.gather(thetas, root=0)
    windows_segment = comm.gather(windows_segment, root=0)
    segment = comm.gather(segment, root=0)
    all_eigs = comm.gather(all_eigs, root=0)

    # clean up
    gc.collect()

    send_models = None
    if rank == 0:
        # output segments
        segment = np.concatenate(segment)
        np.save(''.join([OUTPUT_DIR, 'segments']), segment)

        # output windows
        windows_segment = np.array(windows_segment)
        windows = reconcile_windows(windows_segment)
        np.save(''.join([OUTPUT_DIR, 'windows']), windows)
        print('windows_segments, shape: {}.'.format(
              windows.shape))

        # thetas
        thetas = np.vstack(thetas)
        np.save(''.join([OUTPUT_DIR, 'thetas']), thetas)
        print('thetas, shape: {}'.format(
            thetas.shape))

        # eigs
        all_eigs = np.concatenate(all_eigs)
        np.save(''.join([OUTPUT_DIR, 'all_eigs']), all_eigs)
        print('eigs, shape: {}.'.format(all_eigs.shape))

        t1 = time.time()
        print('calculating thetas, eigevalues and break points took {}'
              .format(t1-t0))

    #     # set up model containers
    #     all_models = []
    #     for kw, window in enumerate(windows):
    #         all_models.append(kw)

    #     # # housekeeping
    #     del windows
    #     gc.collect()

    #     all_models = np.hstack(all_models)  # broadcast this 
    #     len_models = len(all_models)  #
    #     print('{} models'.format(len_models))

    #     # if using combinations
    #     # n_unique = int(len_models * (len_models-1) / 2) #

    #     # all_models_size = np.empty(1, dtype=np.int32)
    #     # all_models_size[:] = len_models

    #     buf2_size = np.empty(1, dtype=np.int64)
    #     buf2_size[:] = int(len_models/size) + 1
    #     print('models per rank: {}, with size: {}'.format(buf2_size[0], size))

    #     buf_len_models = np.empty(1, dtype=np.int64)
    #     buf_len_models[:] = len_models
    #     # print('number models: {}'.format(int(len_models)))

    #     t2 = time.time()
    #     print('setting up models took {}s, total processing time {}s'.format(
    #           t2-t1, t2-t0))

    #     # clean up
    #     gc.collect()

    # else:
    #     buf2_size = np.empty(1, dtype=np.int64)
    #     buf_len_models = np.empty(1, dtype=np.int64)
    #     windows = np.load(''.join([OUTPUT_DIR, 'windows_segments.npy']))
    #     all_models = np.empty([len(windows), 2], dtype=np.int32)


    # # cast models
    # comm.Barrier()
    # comm.Bcast(buf_len_models, root=0)
    # comm.Bcast(all_models, root=0)
    # comm.Bcast(buf2_size, root=0)

    # strt = buf2_size[0] * rank
    # if (buf2_size[0] * rank + buf2_size[0]) < buf_len_models[0]:
    #     stop = buf2_size[0] * rank + buf2_size[0]
    # else:
    #     stop = buf_len_models[0]

    # iter = product([i for i in np.arange(strt, stop)],
    #                [j for j in np.arange(0, buf_len_models[0])])

    # # recv_models = np.empty(buf_len_models,
    # #                        dtype=[('x', '1i'), ('y', '1i'), ('p', '2i')])

    # # creat file to write to on the server
    # # f = h5py.File(
    # #     ''.join([OUTPUT_DIR,'parallel_test.hdf5']),
    # #     'w',
    # #     driver='mpio',
    # #     comm=MPI.COMM_WORLD
    # #     )
    # # dset = f.create_dataset(
    # #     'dissim_mat',
    # #     (buf_len_models, buf_len_models),
    # #     dtype='f'
    # #     )

    # # load data for models
    # sample_tseries = np.load(''.join([OUTPUT_DIR,
    #                          '{}_truncated'.format(TSERIES.split('.')[0]),
    #                                   '.npy']))
    # sample_tseries = ma.masked_invalid(sample_tseries)
    # thetas = np.load(''.join([OUTPUT_DIR, 'thetas.npy']))
    # windows = np.load(''.join([OUTPUT_DIR, 'windows_segments.npy']))
    # # assign models to compare based on rank
    
    # dset = []
    # while True:
    #     try:
    #         combo = next(iter)
    #         idx1 = combo[0]
    #         idx2 = combo[1]
    #         if rank == 0:
    #             models_ = [all_models[idx1], all_models[idx2]]
    #         else:
    #             models_ = [all_models[idx1][0], all_models[idx2][0]]
    #         ld = \
    #             likelihood_distance(
    #                 models_, windows,
    #                 sample_tseries,
    #                 thetas
    #                 )
    #         dset.append([idx1, idx2, ld])
    #     except StopIteration:
    #         print(rank, 'stopping iteration')
    #         break
    # dset = np.concatenate(dset)

    # # gather the dissimilarity matrices from all nodes
    # comm.Barrier()
    # dset = comm.gather(dset, root=0)
    # # dissim_mat = comm.gather(dissim_mat, root=0)

    # if rank == 0:
    #     print('finished calculations, outputting data')
    #     np.save(''.join([OUTPUT_DIR, 'll_models']), dset)