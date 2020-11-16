"""
Functions for coordinate transforms and dynamics
"""
import numpy as np


""" WGS-84 Oblate Earth Model """
WGS84 = {
        'a' : 6378137,
        'f_inv' : 298.257223563,
        'b' : 6356752,
        'e' : 0.0818191908426,
        'omega_earth' : 7.2921150e-5,
        'GM' : 3986004.418e8,
        'J2' : 1.082626684e-3
        }


def meridian_radius(latitude):
    """
    Computes the meridian (North-South) radius
    Inputs: geodetic latitude
    """
    a = WGS84['a']
    e = WGS84['e']
    M = a * (1 - e ** 2) / ((1 - (e * np.sin(latitude)) ** 2) ** (3 / 2))
    return M


def prime_vertical_radius(latitude):
    """
    Computes the prime vertical radius
    """
    a = WGS84['a']
    e = WGS84['e']
    N = a / ((1 - (e * np.sin(latitude)) ** 2) ** (1 / 2))
    return N


def posGeodetic2Ecef(pos_geodetic):
    """
    Converts a point in geodetic coords to ECEF coords
    """
    lat = pos_geodetic[0,0]
    lon = pos_geodetic[1,0]
    alt = pos_geodetic[2,0]
    e = WGS84['e']
    N = prime_vertical_radius(lat)
    px = (N + alt) * np.cos(lat) * np.cos(lon)
    py = (N + alt) * np.cos(lat) * np.sin(lon)
    pz = (N * (1 - e ** 2) + alt) * np.sin(lat)
    pos_ecef = np.array([px, py, pz]).reshape(-1, 1)
    return pos_ecef


def ecef2Ned(v_ecef, pos_geodetic):
    """
    Converts a vector in ECEF to NED using geodetic position
    """
    lat = pos_geodetic[0,0]
    lon = pos_geodetic[1,0]
    DCM = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                    [-np.sin(lon), np.cos(lon), 0],
                    [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), -np.sin(lat)]])
    v_ned = DCM @ v_ecef
    return v_ned


def velNed2Geo(v_ned, pos_geodetic):
    """
    Converts veocity from NED to geodetic coords
    (latitude, longitude and  altitude rates)
    """
    D = velNed2GeoMatrix(pos_geodetic)
    v_geo = D @ v_ned
    return v_geo


def velNed2GeoMatrix(pos_geodetic):
    """
    The transformation matrix from NED velocity to geodetic
    velocity
    """
    lat = pos_geodetic[0, 0]
    alt = pos_geodetic[2, 0]
    M = meridian_radius(lat)
    N = prime_vertical_radius(lat)
    D = np.diag([1 / (M + alt), 1 / ((N + alt) * np.cos(lat)), -1])
    return D


def angularRateNedECEF(v_ned, pos_geodetic):
    """
    Computes the angular velocity of NED frame w.r.t ECEF in NED frame
    """
    v_geo = velNed2Geo(v_ned, pos_geodetic)
    omega_ne_n = np.array([[v_geo[1,0] * np.cos(pos_geodetic[0,0])],
                           [-v_geo[0,0]],
                           [-v_geo[1,0] * np.sin(pos_geodetic[0,0])]])
    return omega_ne_n


def deg2rad(angle):
    return angle * (np.pi / 180)

