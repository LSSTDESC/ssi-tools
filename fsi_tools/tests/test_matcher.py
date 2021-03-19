import numpy as np
from lsst import geom

import pytest

from ..matching import Matcher


def _gen_sphere_pts(n, seed):
    rng = np.random.RandomState(seed=seed)
    ra = rng.uniform(size=n) * 360
    dec = np.arcsin(rng.uniform(size=n, low=-1, high=1)) / np.pi * 180.0
    return ra, dec


@pytest.mark.parametrize('k', [1, 2, 3])
def test_matcher_knn(k):
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)
    d, idx = mch.query_knn(ra, dec, k=k)

    # test via brute force
    sps = [geom.SpherePoint(ra[i], dec[i], geom.degrees) for i in range(ra.shape[0])]
    idxs = np.arange(ra.shape[0])
    for i in range(ra.shape[0]):
        ds = np.array([
            sps[i].separation(sps[j]).asArcseconds() for j in range(ra.shape[0])
        ])
        inds = np.argsort(ds)

        if k != 1:
            assert np.allclose(d[i, :], ds[inds[:k]])
            assert np.array_equal(idx[i, :], idxs[inds[:k]])
        else:
            assert np.allclose(d[i], 0)
            assert np.array_equal(idx[i], i)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_matcher_knn_maxrad(k):
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)
    d, idx = mch.query_knn(ra, dec, distance_upper_bound=5e4, k=k)

    # test via brute force
    sps = [geom.SpherePoint(ra[i], dec[i], geom.degrees) for i in range(ra.shape[0])]
    idxs = np.arange(ra.shape[0])
    for i in range(ra.shape[0]):
        ds = np.array([
            sps[i].separation(sps[j]).asArcseconds() for j in range(ra.shape[0])
        ])
        inds = np.argsort(ds)
        msk = (ds[inds] < 5e4) & (np.arange(ra.shape[0]) < k)
        s = np.sum(msk)

        if k != 1:
            assert np.allclose(d[i, :s], ds[inds[msk]])
            assert np.array_equal(idx[i, :s], idxs[inds[msk]])
        else:
            assert np.allclose(d[i], 0)
            assert np.array_equal(idx[i], i)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_matcher_knn_maxrad_inf(k):
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)
    rap, decp = _gen_sphere_pts(50, 443)
    d, idx = mch.query_knn(rap, decp, distance_upper_bound=1, k=k)
    assert not np.any(np.isfinite(d))
    assert np.all(idx == 50)


def test_matcher_radius():
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)

    rap, decp = _gen_sphere_pts(100, 454)
    idx = mch.query_radius(rap, decp, 6e4)

    sps = [geom.SpherePoint(ra[i], dec[i], geom.degrees) for i in range(ra.shape[0])]
    spsp = [
        geom.SpherePoint(rap[i], decp[i], geom.degrees)
        for i in range(rap.shape[0])
    ]

    for ic in range(ra.shape[0]):
        idxc = []
        for ip in range(rap.shape[0]):
            sep = sps[ic].separation(spsp[ip]).asArcseconds()
            if sep < 6e4:
                idxc.append(ip)
        assert set(idxc) == set(idx[ic])
