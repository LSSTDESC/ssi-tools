import numpy as np

import lsst.geom as geom

from hexalattice.hexalattice import create_hex_grid


def make_hexgrid_for_tract(tract_info, spacing_pixels=64, rng=None):
    """Make a randomly rotated hexagonal grid of object locations over a tract.

    Parameters
    ----------
    tract_info : lsst.skymap.TractInfo
        The tract for which to generate the positions
    spacing_pixels : float, optional
        The spacing in pixels between the centers of the hexagonal grid points.
    rng : int, None, or np.random.RandomState
        An RNG instance to use to generate the random rotation of the grid.

    Returns
    -------
    grid_data : numpy structured array
        A structured aray with columns
            ra: ra in degrees
            dec: dec in degrees
            x: pixel x-location on the tract
            y: pixel y-location on the tract
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(seed=rng)

    # compute the dimensions of the tract
    dimx = tract_info.getBBox().getDimensions().x
    dimy = tract_info.getBBox().getDimensions().y
    nx = int(dimx / spacing_pixels * np.sqrt(2) * 1.05)
    # the factor of 0.866 makes sure the grid is square-ish
    ny = int(dimy / spacing_pixels * np.sqrt(2) * 1.05 / 0.8660254)

    # make a base grid
    # here the spacing between grid centers is 1
    v, _ = create_hex_grid(nx=nx, ny=ny, rotate_deg=rng.uniform() * 365)

    # convert the spacing to right number of pixels
    # we also recenter the grid since it comes out centered at 0,0
    v *= spacing_pixels
    v[:, 0] += (dimx / 2)
    v[:, 1] += (dimy / 2)

    # now cut to things in the tract
    # this is done with
    # 1. an initial cut of the pixel location
    # 2. transforming to ra,dec and using the sky polygon
    # cut 1
    msk = (
        (v[:, 0] >= 0)
        & (v[:, 0] < dimx)
        & (v[:, 1] >= 0)
        & (v[:, 1] < dimy)
    )
    v = v[msk, :]

    # cut 2
    wcs = tract_info.getWcs()
    ra, dec = wcs.pixelToSkyArray(v[:, 0], v[:, 1], degrees=True)
    cvx = tract_info.getOuterBBox()
    msk = np.array([
        cvx.contains(geom.SpherePoint(ra[i], dec[i], geom.degrees).getVector())
        for i in range(len(ra))
    ])
    ra = ra[msk]
    dec = dec[msk]
    v = v[msk, :]

    grid_data = np.zeros(v.shape[0], dtype=[
        ("ra", "f8"),
        ("dec", "f8"),
        ("x", "f8"),
        ("y", "f8")
    ])
    grid_data["ra"] = ra
    grid_data["dec"] = dec
    grid_data["x"] = v[:, 0]
    grid_data["y"] = v[:, 1]

    return grid_data
