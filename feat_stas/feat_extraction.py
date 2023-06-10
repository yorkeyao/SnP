import os
import pathlib
import torchvision

import numpy as np
import torch
from scipy import linalg
from matplotlib.pyplot import imread, imsave
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
from scipy.special import softmax
from shutil import copyfile
from k_means_constrained import KMeansConstrained
from glob import glob
import os.path as osp
from PIL import Image
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans
import xml.dom.minidom as XD
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x



def make_square(image, max_dim = 512):
    max_dim = max(np.shape(image)[0], np.shape(image)[1])
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image

def get_activations(opt, files, model, batch_size=50, dims=8192,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # if len(files) % batch_size != 0:
    #     print(('Warning: number of images is not a multiple of the '
    #            'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'), len(files))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_remainder=  len(files) % batch_size

    print('\rnumber of batches is %d' % n_batches),
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs+n_remainder, dims))
    if n_remainder!=0:
        n_batches=n_batches+1
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        if n_remainder!=0 and i==n_batches-1:
          end = start + n_remainder
        else:
          end = start + batch_size

        # print (files[start:end])
        images = np.array([resize( imread(str(f)).astype(np.float32), (64, 64, 3) ).astype(np.float32)
                           for f in files[start:end]])

        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()
        
        if opt.FD_model == 'inception':
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        if opt.FD_model == 'posenet':
            pred = model(batch)
            # print (np.shape (pred))
            pred = adaptive_max_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)
        print('\rPropagating batch %d/%d' % (i + 1, n_batches))

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(opt, files, model, batch_size=50,
                                    dims=8192, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(opt, files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    #eigen_vals, eigen_vecs= np.linalg.eig(sigma)
    #sum_eigen_val=eigen_vals.sum().real
    sum_eigen_val = (sigma.diagonal()).sum()
    return mu, sigma, sum_eigen_val


def _compute_statistics_of_path(opt, path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        #random.shuffle(files)
        #files = files[:2000]
        m, s, sum_eigen_val = calculate_activation_statistics(opt, files, model, batch_size,
                                               dims, cuda) 
    return m, s, sum_eigen_val
