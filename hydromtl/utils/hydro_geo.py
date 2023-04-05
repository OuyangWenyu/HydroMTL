import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import transform, CRS
from shapely.geometry import Polygon, Point

from hydroSPB.utils.hydro_utils import serialize_numpy


def spatial_join(points_file, polygons_file):
    """join polygons layer to point layer, add polygon which the point is in to the point"""

    points = gpd.read_file(points_file)
    polys = gpd.read_file(polygons_file)
    # Check the data
    if not (points.crs == polys.crs):
        points = points.to_crs(polys.crs)

    # Make a spatial join
    join = gpd.sjoin(points, polys, how="inner", op="within")
    return join


def crd2grid(y, x):
    ux, indX0, indX = np.unique(x, return_index=True, return_inverse=True)
    uy, indY0, indY = np.unique(y, return_index=True, return_inverse=True)

    minDx = np.min(ux[1:] - ux[0:-1])
    minDy = np.min(uy[1:] - uy[0:-1])
    maxDx = np.max(ux[1:] - ux[0:-1])
    maxDy = np.max(uy[1:] - uy[0:-1])
    if maxDx > minDx * 2:
        print("skipped rows")
    #     indMissX=np.where((ux[1:]-ux[0:-1])>minDx*2)[0]
    #     insertX=(ux[indMissX+1]+ux[indMissX])/2
    #     ux=np.insert(ux,indMissX,insertX)
    if maxDy > minDy * 2:
        print("skipped coloums")
    #     indMissY=np.where((uy[1:]-uy[0:-1])>minDy*2)
    #     raise Exception('skipped coloums or rows')

    uy = uy[::-1]
    ny = len(uy)
    indY = ny - 1 - indY
    return (uy, ux, indY, indX)


def array2grid(data, *, lat, lon):
    (uy, ux, indY, indX) = crd2grid(lat, lon)
    ny = len(uy)
    nx = len(ux)
    if data.ndim == 2:
        nt = data.shape[1]
        grid = np.full([ny, nx, nt], np.nan)
        grid[indY, indX, :] = data
    elif data.ndim == 1:
        grid = np.full([ny, nx], np.nan)
        grid[indY, indX] = data
    return grid, uy, ux


def trans_points(from_crs, to_crs, pxs, pys):
    """
    put the data into dataframe so that the speed of processing could be improved obviously

    Parameters
    ----------
    from_crs
        source CRS
    to_crs
        target CRS
    pxs
        x of every point (list/array)
    pys
        y of every point (list/array)

    Returns
    -------
    np.array
        x and y compared a pair list to initialize a polygon
    """
    df = pd.DataFrame({"x": pxs, "y": pys})
    start = time.time()
    df["x2"], df["y2"] = transform(from_crs, to_crs, df["x"].tolist(), df["y"].tolist())
    end = time.time()
    print("time consuming：", "%.7f" % (end - start))
    # after transforming xs and ys, pick out x2, y2，and tranform to numpy array，then do a transportation. Finally put coordination of every row to a list
    arr_x = df["x2"].values
    arr_y = df["y2"].values
    pxys_out = np.stack((arr_x, arr_y), 0).T
    return pxys_out


def trans_polygon(from_crs, to_crs, polygon_from):
    """transform coordination of every point of a polygon to one in a given coordination system"""
    polygon_to = Polygon()
    # data type: tuples in a list
    boundary = polygon_from.boundary
    boundary_type = boundary.geom_type
    print(boundary_type)
    if boundary_type == "LineString":
        pxs = polygon_from.exterior.xy[0]
        pys = polygon_from.exterior.xy[1]
        pxys_out = trans_points(from_crs, to_crs, pxs, pys)
        polygon_to = Polygon(pxys_out)
    elif boundary_type == "MultiLineString":
        # if there is interior boundary in a polygon，then we need to transform its coordinations. Notice: maybe multiple interior boundaries exist.
        exts_x = boundary[0].xy[0]
        exts_y = boundary[0].xy[1]
        pxys_ext = trans_points(from_crs, to_crs, exts_x, exts_y)

        pxys_ints = []
        for i in range(1, len(boundary)):
            ints_x = boundary[i].xy[0]
            ints_y = boundary[i].xy[1]
            pxys_int = trans_points(from_crs, to_crs, ints_x, ints_y)
            pxys_ints.append(pxys_int)

        polygon_to = Polygon(shell=pxys_ext, holes=pxys_ints)
    else:
        print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return polygon_to


def write_shpfile(geodata, output_folder, id_str="hru_id"):
    """generate a shpfile from geodataframe，the name is id of the pandas dataframe"""
    # Create a output path for the data
    gage_id = geodata.iloc[0, :][id_str]
    # id is number，here turn it to str
    output_file = str(int(gage_id)).zfill(8)
    output_fp = os.path.join(output_folder, output_file + ".shp")
    # Write those rows into a new file (the default output file format is Shapefile)
    geodata.to_file(output_fp)


def nearest_point_index(crs_from, crs_to, lon, lat, xs, ys):
    # x and y are proj coord，lon, lat should be transformed (x is longtitude projection，y is lat)
    x, y = transform(crs_from, crs_to, lon, lat)
    index_x = (np.abs(xs - x)).argmin()
    index_y = (np.abs(ys - y)).argmin()
    return [index_x, index_y]


def create_mask(poly, xs, ys, lons, lats, crs_from, crs_to):
    mask_index = []
    poly_bound = poly.bounds
    poly_bound_min_lat = poly_bound[1]
    poly_bound_min_lon = poly_bound[0]
    poly_bound_max_lat = poly_bound[3]
    poly_bound_max_lon = poly_bound[2]
    index_min = nearest_point_index(
        crs_from, crs_to, poly_bound_min_lon, poly_bound_min_lat, xs, ys
    )
    index_max = nearest_point_index(
        crs_from, crs_to, poly_bound_max_lon, poly_bound_max_lat, xs, ys
    )
    range_x = [index_min[0], index_max[0]]
    range_y = [index_max[1], index_min[1]]
    for i in range(range_y[0], range_y[1] + 1):
        for j in range(range_x[0], range_x[1] + 1):
            if is_point_in_boundary(lons[i][j], lats[i][j], poly):
                mask_index.append((i, j))
    return mask_index


def is_point_in_boundary(px, py, poly):
    point = Point(px, py)
    return point.within(poly)


def calc_avg(mask, netcdf_data, var_type):
    mask = np.array(mask)
    index = mask.T

    def f_avg(i):
        data_day = netcdf_data.variables[var_type][i]
        data_chosen = data_day[index[0], index[1]]
        data_mean = np.mean(data_chosen, axis=0)
        return data_mean

    all_mean_data = list(map(f_avg, range(365)))

    return all_mean_data


def basin_avg_netcdf(netcdf_file, shp_file, mask_file):
    data_netcdf = Dataset(netcdf_file, "r")  # reads the netCDF file
    print(data_netcdf)
    # get all variable names
    print(data_netcdf.variables.keys())
    temp_lat = data_netcdf.variables["lat"]  # temperature variable
    temp_lon = data_netcdf.variables["lon"]  # temperature variable
    for d in data_netcdf.dimensions.items():
        print(d)
    x, y = data_netcdf.variables["x"], data_netcdf.variables["y"]
    x = data_netcdf.variables["x"][:]
    y = data_netcdf.variables["y"][:]
    lx = list(x)
    ly = list(y)
    print(all(ix < jx for ix, jx in zip(lx, lx[1:])))
    print(all(iy > jy for iy, jy in zip(ly, ly[1:])))
    lons = data_netcdf.variables["lon"][:]
    lats = data_netcdf.variables["lat"][:]

    crs_pro_str = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    crs_geo_str = "+proj=longlat +datum=WGS84 +no_defs"
    crs_from = CRS.from_proj4(crs_geo_str)
    crs_to = CRS.from_proj4(crs_pro_str)

    new_shps = gpd.read_file(shp_file)
    polygon = new_shps.at[0, "geometry"]
    start = time.time()
    mask = create_mask(polygon, x, y, lons, lats, crs_from, crs_to)
    end = time.time()
    print("time：", "%.7f" % (end - start))
    serialize_numpy(np.array(mask), mask_file)
    var_types = ["tmax"]
    # var_types = ['tmax', 'tmin', 'prcp', 'srad', 'vp', 'swe', 'dayl']
    avgs = []
    for var_type in var_types:
        start = time.time()
        avg = calc_avg(mask, data_netcdf, var_type)
        end = time.time()
        print("time：", "%.7f" % (end - start))
        print("mean value：", avg)
        avgs.append(avg)

    return avgs


def ind_of_dispersion(coord, points):
    """the ratio of variance and mean value of Euclidean distances between event points and a selected point"""
    points = np.asarray(points)
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    mean_d = np.sqrt(xd * xd + yd * yd).mean()
    var_d = np.sqrt(xd * xd + yd * yd).var()
    ind = var_d / mean_d
    return ind


def coefficient_of_variation(coord, points):
    """the ratio of the standard deviation to the mean (average) value of Euclidean distances between event points
    and a selected point"""
    if len(points) == 0:
        return np.nan
    points = np.asarray(points)
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    mean_d = np.sqrt(xd * xd + yd * yd).mean()
    var_d = np.sqrt(xd * xd + yd * yd).var()
    coefficient = np.sqrt(var_d) / mean_d * 100
    return coefficient
