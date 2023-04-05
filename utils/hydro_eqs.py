"""
Some common equations in hydrology

TODO: not fully tested yet
"""
from typing import Union

import numpy as np
import xarray as xr
from numba import njit

from hydroSPB.utils.meteo_utils import (
    calc_press,
    calc_psy,
    calc_vpc,
    calc_lambda_,
    calc_ea,
    calc_es,
)
from hydroSPB.utils.rad_utils import calc_rad_short, calc_rad_long


@njit
def get_priestley_taylor_pet(
    t_min: np.ndarray,
    t_max: np.ndarray,
    s_rad: np.ndarray,
    lat: float,
    elev: float,
    doy: np.ndarray,
) -> np.ndarray:
    """
    From neuralhydrology: https://github.com/neuralhydrology/neuralhydrology/blob/a49f2e43cb2b25800adde2601ebf365db79dd745/neuralhydrology/datautils/pet.py#L239
    Calculate potential evapotranspiration (PET) as an approximation following the Priestley-Taylor equation.
    The ground head flux G is assumed to be 0 at daily time steps (see Newman et al., 2015 [#]_). The
    equations follow FAO-56 (Allen et al., 1998 [#]_).

    Parameters
    ----------
    t_min : np.ndarray
        Daily min temperature (degree C)
    t_max : np.ndarray
        Daily max temperature (degree C)
    s_rad : np.ndarray
        Solar radiation (Wm-2)
    lat : float
        Latitude in degree
    elev : float
        Elevation in m
    doy : np.ndarray
        Day of the year

    Returns
    -------
    np.ndarray
        Array containing PET estimates in mm/day

    References
    ----------
    .. [#] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett,
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale
        hydrometeorological data_source for the contiguous USA: data_source characteristics and assessment of regional
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223,
        doi:10.5194/hess-19-209-2015, 2015
    .. [#] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration-Guidelines for computing
        crop water requirements-FAO Irrigation and drainage paper 56. Fao, Rome, 300(9), D05109.
    """

    lat = lat * (np.pi / 180)  # degree to rad

    # Slope of saturation vapour pressure curve
    t_mean = 0.5 * (t_min + t_max)
    slope_svp = _get_slope_svp_curve(t_mean)

    # incoming netto short-wave radiation
    s_rad = s_rad * 0.0864  # conversion Wm-2 -> MJm-2day-1
    in_sw_rad = _get_net_sw_srad(s_rad)

    # outgoginng netto long-wave radiation
    sol_dec = _get_sol_decl(doy)
    sha = _get_sunset_hour_angle(lat, sol_dec)
    ird = _get_ird_earth_sun(doy)
    et_rad = _get_extraterra_rad(lat, sol_dec, sha, ird)
    cs_rad = _get_clear_sky_rad(elev, et_rad)
    a_vp = _get_avp_tmin(t_min)
    out_lw_rad = _get_net_outgoing_lw_rad(t_min, t_max, s_rad, cs_rad, a_vp)

    # net radiation
    net_rad = _get_net_rad(in_sw_rad, out_lw_rad)

    # gamma
    atm_pressure = _get_atmos_pressure(elev)
    gamma = _get_psy_const(atm_pressure)

    # PET MJm-2day-1
    alpha = 1.26  # Calibrated in CAMELS, here static
    _lambda = 2.45  # Kept constant, MJkg-1
    pet = (alpha / _lambda) * (slope_svp * net_rad) / (slope_svp + gamma)

    # convert energy to evap
    pet = pet * 0.408

    return pet


@njit
def _get_slope_svp_curve(t_mean: np.ndarray) -> np.ndarray:
    """
    Slope of saturation vapour pressure curve
    Equation 13 FAO-56 Allen et al. (1998)

    Parameters
    ----------
    t_mean : np.ndarray
        Mean temperature (degree C)

    Returns
    -------
    np.ndarray
        Slope of the saturation vapor pressure curve in kPa/(degree C)

    """

    delta = (
        4098
        * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3)))
        / ((t_mean + 237.3) ** 2)
    )
    return delta


@njit
def _get_net_sw_srad(s_rad: np.ndarray, albedo: float = 0.23) -> np.ndarray:
    """Calculate net shortwave radiation
    Equation 38 FAO-56 Allen et al. (1998)
    Parameters
    ----------
    s_rad : np.ndarray
        Incoming solar radiation (MJm-2day-1)
    albedo : float, optional
        Albedo, by default 0.23
    Returns
    -------
    np.ndarray
        Net shortwave radiation (MJm-2day-1)

    """

    net_srad = (1 - albedo) * s_rad
    return net_srad


@njit
def _get_sol_decl(doy: np.ndarray) -> np.ndarray:
    """
    Get solar declination
    Equation 24 FAO-56 Allen et al. (1998)

    Parameters
    ----------
    doy : np.ndarray
        Day of the year
    Returns
    -------
    np.ndarray
        Solar declination in rad

    """

    # equation 24 FAO Allen
    sol_dec = 0.409 * np.sin((2 * np.pi) / 365 * doy - 1.39)
    return sol_dec


@njit
def _get_sunset_hour_angle(lat: float, sol_dec: np.ndarray) -> np.ndarray:
    """Sunset hour angle

    Parameters
    ----------
    lat : float
        Latitude in rad
    sol_dec : np.ndarray
        Solar declination in rad

    Returns
    -------
    np.ndarray
        Sunset hour angle in rad
    """
    term = -np.tan(lat) * np.tan(sol_dec)
    term[term < -1] = -1
    term[term > 1] = 1
    sha = np.arccos(term)
    return sha


@njit
def _get_ird_earth_sun(doy: np.ndarray) -> np.ndarray:
    """Inverse relative distance between Earth and Sun
    Equation 23 FAO-56 Allen et al. (1998)

    Parameters
    ----------
    doy : np.ndarray
        Day of the year

    Returns
    -------
    np.ndarray
        Inverse relative distance between Earth and Sun
    """
    ird = 1 + 0.033 * np.cos((2 * np.pi) / 365 * doy)
    return ird


@njit
def _get_extraterra_rad(
    lat: float, sol_dec: np.ndarray, sha: np.ndarray, ird: np.ndarray
) -> np.ndarray:
    """
    Extraterrestrial Radiation
    Equation 21 FAO-56 Allen et al. (1998)

    Parameters
    ----------
    lat : float
        Lat in rad (pos for northern hemisphere)
    sol_dec : np.ndarray
        Solar declination in rad
    sha : np.ndarray
        Sunset hour angle in rad
    ird : np.ndarray
        Inverse relative distance of Earth and Sun

    Returns
    -------
    np.ndarray
        Extraterrestrial radiation MJm-2day-1
    """
    term1 = (24 * 60) / np.pi * 0.082 * ird
    term2 = sha * np.sin(lat) * np.sin(sol_dec) + np.cos(lat) * np.cos(
        sol_dec
    ) * np.sin(sha)
    et_rad = term1 * term2
    return et_rad


@njit
def _get_clear_sky_rad(elev: float, et_rad: np.ndarray) -> np.ndarray:
    """
    Clear sky radiation
    Equation 37 FAO-56 Allen et al. (1998)

    Parameters
    ----------
    elev : float
        Elevation in m
    et_rad : np.ndarray
        Extraterrestrial radiation in MJm-2day-1

    Returns
    -------
    np.ndarray
        Clear sky radiation MJm-2day-1
    """
    cs_rad = (0.75 + 2 * 10e-5 * elev) * et_rad
    return cs_rad


@njit
def _get_avp_tmin(t_min: np.ndarray) -> np.ndarray:
    """Actual vapor pressure estimated using min temperature
    Equation 48 FAO-56 Allen et al. (1998)
    Parameters
    ----------
    t_min : np.ndarray
        Minimum temperature in degree C
    Returns
    -------
    np.ndarray
        Actual vapor pressure kPa
    """
    avp = 0.611 * np.exp((17.27 * t_min) / (t_min + 237.3))
    return avp


@njit
def _get_net_outgoing_lw_rad(
    t_min: np.ndarray,
    t_max: np.ndarray,
    s_rad: np.ndarray,
    cs_rad: np.ndarray,
    a_vp: np.ndarray,
) -> np.ndarray:
    """Net outgoing longwave radiation
    Expects temperatures in degree and does the conversion in kelvin in the function.
    Equation 49 FAO-56 Allen et al. (1998)
    Parameters
    ----------
    t_min : np.ndarray
        Min temperature in degree C
    t_max : np.ndarray
        Max temperature in degree C
    s_rad : np.ndarray
        Measured or modeled solar radiation MJm-2day-1
    cs_rad : np.ndarray
        Clear sky radiation MJm-2day-1
    a_vp : np.ndarray
        Actuatal vapor pressure kPa
    Returns
    -------
    np.ndarray
        Net outgoing longwave radiation MJm-2day-1
    """
    term1 = (
        (t_max + 273.16) ** 4 + (t_min + 273.16) ** 4
    ) / 2  # conversion in K in equation
    term2 = 0.34 - 0.14 * np.sqrt(a_vp)
    term3 = 1.35 * s_rad / cs_rad - 0.35
    stefan_boltzman = 4.903e-09
    net_lw = stefan_boltzman * term1 * term2 * term3
    return net_lw


@njit
def _get_net_rad(sw_rad: np.ndarray, lw_rad: np.ndarray) -> np.ndarray:
    """Net radiation
    Equation 40 FAO-56 Allen et al. (1998)
    Parameters
    ----------
    sw_rad : np.ndarray
        Net incoming shortwave radiation MJm-2day-1
    lw_rad : np.ndarray
        Net outgoing longwave radiation MJm-2day-1
    Returns
    -------
    np.ndarray
        [description]
    """
    return sw_rad - lw_rad


@njit
def _get_atmos_pressure(elev: float) -> float:
    """Atmospheric pressure
    Equation 7 FAO-56 Allen et al. (1998)
    Parameters
    ----------
    elev : float
        Elevation in m
    Returns
    -------
    float
        Atmospheric pressure in kPa
    """
    temp = (293.0 - 0.0065 * elev) / 293.0
    return np.power(temp, 5.26) * 101.3


@njit
def _get_psy_const(atm_pressure: float) -> float:
    """Psychometric constant
    Parameters
    ----------
    atm_pressure : float
        Atmospheric pressure in kPa
    Returns
    -------
    float
        Psychometric constant in kPa/(degree C)
    """
    return 0.000665 * atm_pressure


@njit
def _srad_from_t(et_rad, cs_rad, t_min, t_max, coastal=False):
    """Estimate solar radiation from temperature"""
    # equation 50
    if coastal:
        adj = 0.19
    else:
        adj = 0.16

    sol_rad = adj * np.sqrt(t_max - t_min) * et_rad

    return np.minimum(sol_rad, cs_rad)


def penman_monteith(
    rad: np.array,
    hmd: np.array,
    e0: np.array,
    wnd: np.array,
    tmpr: np.array,
    elev: np.array,
    tau: np.array,
    ref,
    option=0,
):
    """
    From Dr. Chaopeng Shen:
    (E) This function computes the reference ET using Penman Monteith method
     Steps of calculating ET:
     (1) reference ET for water, soil and vegetations (reference type)
     (2) spread solar radiation using Beer-Lambert law and LAI
     (3) compute total ET demand for each layer by considering crop coeff
     (4) compute actual ET by considering water stress and depth distribution
     (5) add up ET at each layer to become ET sink term for VDZ model
     inside this function, Rad, Hb, Hli, Rn are MJ/m2/day
     rET, rETs are mm/day
     RET is in mm/day

     Parameters
     -----------
     rad
        shortwave incident radiation (MJ/m2/day), unless option == 1 is specified
     hmd
        relative humidity ([-])
     e0
        saturation vapor pressure (equation available in makeDay, temp degrees C)
     wnd
        wind speed m/s
     tmpr
        temperature (celcius)
     elev
        elevation (m)
     tau
        air transmittance, but, when option ==1, this does not matter
     ref
        a choice of reference crop. == 0 is FAO grass, == 1 is alfalfa
     option
        when option is 1, rad means net radiation

    Return
    --------
    list
        rET: reference ET; rETs: reference ET for bare soil, Rn: net radiation-
        Hb: outgoing long wave radiation, Hli: incoming long wave radiation
    """
    TK = tmpr + 273.15
    Cp = 1.013e-3
    sigma = 4.903e-9
    rho = 1710 - 6.85 * tmpr
    P = 101.3 - 0.01152 * elev + 0.544e-6 * elev ** 2  # Atmospheric Pressure in kPa
    lambda_ = 2.501 - 2.361e-3 * tmpr  # Specific Heat of evaporation MJ / kg
    gamma = (Cp / 0.622) * (P / lambda_)  # psychrometric constant, kPa / C
    # e0 = exp((16.78. * T - 116.9). / (T + 237.3)); # saturation vapor pressure in kPa
    # Delta = 4098 * (e0. / TK. ^ 2); % Slope of saturation Curve THIS IS A BUG!!!!
    Delta = 4098 * (e0 / (tmpr + 237.3) ** 2)  # kPa / C
    e = hmd * e0
    DE = e0 - e  # kPa
    alb = np.zeros(elev.size)
    alb[:] = 0.23
    # alb(sno > 0.5) = 0.8;
    Rs = (1 - alb) * rad  # SNOW COVER CHANGE  ##################################
    # long - wave radiation.
    emm = -(0.34 - 0.139 * np.sqrt(e))
    Hmx = max(rad)
    # fcld = Rs * (0.9 / Hmx) + 0.1;
    ct = tau / 0.7
    cf = 1 - ct  # cloud fraction. 0.7 is the max in Spokas setting
    fcld = 0.9 * ct + 0.1  # fcld = tau / 0.7 is the cloud cover factor
    # fcld = 0.9 * (tau) + 0.1 # temporarily use this!!!!!!!!!!!!!
    ccc = sigma * TK ** 4
    Hb = fcld * emm * ccc  # UNIT - -MJ / m2 / d
    Rn = Rs + Hb  # Rn(Rn < 0) = 0; # THIS COULD BE A BUG!!!!!!!!!!!!!!!!!!!!!!
    if option == 1:
        # for this option, rad means net radiation
        Rn = rad
    EA = e0 * hmd
    # EA1 = 1.08 * (1.0 - exp(-(EA / 100.0). ^ (TK / 2016.0))); % This is a bug, unit of
    # EA is kPa, but it only impacts snowmelt
    EA1 = 1.08 * (1.0 - np.exp(-((EA * 10) ** (TK / 2016.0))))
    Hli = (cf + ct * EA1) * ccc
    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    # ref = 2;
    if (
        ref == 0
    ):  # FAO reference grass, height 0.12, surface resistance 90 s / m, albedo=0.23
        c = Delta + gamma * (1 + 0.34 * wnd)
        a1 = 0.408 * Delta / c
        a2 = gamma * DE * wnd / TK * 900.0 / c
        rET = a1 * Rn + a2
        ra = 208.0 / wnd  # ra of the grass reference
        rETs = (Rn * Delta + rho * gamma * DE / ra) / (
            lambda_ * (Delta + gamma)
        )  # here rho is a combined term, not density
        # below might have been a long term bug::::
        # rETs = (Rn * Delta + rho * Cp * DE / ra) / (lambda_ * (Delta+gamma)) # Soil Maximum ET. We can discuss this later
        # 0.408 may be substituted by 1 / lambda_
        # assume a 0 of Ground heat flux
        # http://www.fao.org/docrep/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
    elif ref == 1:  # alfalfa
        ra = 114 / wnd
        rc = 49
        CT = 1710 - 6.85 * tmpr
        rET = (Delta * Rn / lambda_ + gamma * CT * DE / (ra * lambda_)) / (
            Delta + gamma * (1 + rc / ra)
        )
        # E0 = (Delta. * Rn + (1710 - 6.85 * T). * DE. / ra). / (Delta + gamma. * (1 + rc / ra))
        # WRONG
    elif ref == 2:  # add Priestley - Taylor as RETs
        rETs = (1.28 * Delta / (Delta + gamma) * Rn) / lambda_
        ra = 114.0 / wnd
        rc = 49
        # % CT = 1710 - 6.85 * T
        rET = (Delta * Rn + gamma * rho * DE / ra) / (
            lambda_ * (Delta + gamma * (1 + rc / ra))
        )
    elif ref == 3:
        # Shuttleworth formula from Li-Dan's paper (They missed u)
        # Unit.Delta lambda gamma   DE Rn rET
        # Shuttleworth Pa/K  J/kg  Pa/K  Pa  MJ/m2/day  mm/day
        # Ours kPa/C  MJ/kg  kPa/C  kPa  mm/day  mm/day
        # Examine this paper Zhou, Journal of Hydrology 2006, Estimating
        # potential ETusing shuttleworth - wallace model
        rET = (
            Delta / (Delta + gamma) * Rn / lambda_
            + gamma / (Delta + gamma) * 6.430 * (1 + 0.536 * wnd) * DE / lambda_
        )
    if ref < 3:
        # Should negative hourly data be made positive? probably not
        rET[rET < 0] = 0
        rETs[rETs < 0] = 0
    return [rET, rETs, Rn, Hb, Hli]


def priestley_taylor(
    t_min: Union[np.ndarray, xr.DataArray],
    t_max: Union[np.ndarray, xr.DataArray],
    s_rad: Union[np.ndarray, xr.DataArray],
    lat: Union[np.ndarray, xr.DataArray],
    elevation: Union[np.ndarray, xr.DataArray],
    doy: Union[np.ndarray, xr.DataArray],
    e_a: Union[np.ndarray, xr.DataArray] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """Evaporation calculated according to [priestley_and_taylor_1965]_.

    Parameters
    ----------
    t_max:
        maximum day temperature [°C]
    t_min:
        minimum day temperature [°C]
    s_rad:
        incoming solar radiation [MJ m-2 d-1]
    lat:
        the site latitude [rad]
    elevation:
        the site elevation [m]
    doy:
        Day of the year
    e_a:
        Actual vapor pressure [kPa].
    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        the calculated evaporation [mm day-1]

    Examples
    --------
    >>> pt = priestley_taylor(t_min, t_max, s_rad, lat, elevation, doy, e_a)

    Notes
    -----

    .. math:: PE = \\frac{\\alpha_{PT} \\Delta (R_n-G)}
        {\\lambda(\\Delta +\\gamma)}

    References
    ----------
    .. [priestley_and_taylor_1965] Priestley, C. H. B., & TAYLOR, R. J. (1972).
       On the assessment of surface heat flux and evaporation using large-scale
       parameters. Monthly weather review, 100(2), 81-92.

    """
    #  tmean: average day temperature [°C]
    t_mean = 0.5 * (t_min + t_max)
    pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(t_mean)
    _lambda = calc_lambda_(t_mean)
    albedo = 0.23
    rns = calc_rad_short(s_rad=s_rad, alpha=albedo)  # [MJ/m2/d]
    # a: empirical coefficient for Net Long-Wave radiation [-]
    a = 1.35
    # b: empirical coefficient for Net Long-Wave radiation [-]
    b = -0.35
    rnl = calc_rad_long(
        s_rad,
        doy,
        t_mean=t_mean,
        t_max=t_max,
        t_min=t_min,
        elevation=elevation,
        lat=lat,
        a=a,
        b=b,
        ea=e_a,
    )
    # The total daily value for Rn is almost always positive over a period of 24 hours, except in extreme conditions
    # at high latitudes. Page43 in [allen_1998]
    rn = rns - rnl
    # g: soil heat flux [MJ m-2 d-1], for daily calculation, it equals to 0
    g = 0
    # alpha: calibration coeffiecient [-]
    alpha = 1.26
    return (alpha * dlt * (rn - g)) / (_lambda * (dlt + gamma))


def pm_fao56(
    t_min: Union[np.ndarray, xr.DataArray],
    t_max: Union[np.ndarray, xr.DataArray],
    s_rad: Union[np.ndarray, xr.DataArray],
    lat: Union[np.ndarray, xr.DataArray],
    elevation: Union[np.ndarray, xr.DataArray],
    doy: Union[np.ndarray, xr.DataArray],
    e_a: Union[np.ndarray, xr.DataArray] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """Evaporation calculated according to [allen_1998]_.
    Parameters
    ----------
    t_max:
        maximum day temperature [°C]
    t_min:
        minimum day temperature [°C]
    s_rad:
        incoming solar radiation [MJ m-2 d-1]
    lat:
        the site latitude [rad]
    elevation:
        the site elevation [m]
    doy:
        Day of the year
    e_a:
        Actual vapor pressure [kPa].
    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        containing the calculated evaporation

    Examples
    --------
    >>> et_fao56 = pm_fao56(t_min, t_max, s_rad, lat, elevation, doy, e_a)

    Notes
    -----
    .. math:: PE = \\frac{0.408 \\Delta (R_{n}-G)+\\gamma \\frac{900}{T+273}
        (e_s-e_a) u_2}{\\Delta+\\gamma(1+0.34 u_2)}

    """
    # t_mean: average day temperature [°C]
    t_mean = (t_max + t_min) / 2
    # pressure: atmospheric pressure [kPa]
    pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(t_mean)

    # wind: mean day wind speed [m/s]
    # Where no wind data are available within the region, a value of 2 m/s can be used as a
    # temporary estimate. This value is the average over 2 000 weather stations around the globe. Page63 in [allen_1998]
    wind = 2
    gamma1 = gamma * (1 + 0.34 * wind)
    if e_a is None:
        e_a = calc_ea(t_mean=t_mean, t_max=t_max, t_min=t_min)
    e_s = calc_es(t_mean=t_mean, t_max=t_max, t_min=t_min)
    albedo = 0.23
    rns = calc_rad_short(s_rad=s_rad, alpha=albedo)  # [MJ/m2/d]
    # a: empirical coefficient for Net Long-Wave radiation [-]
    a = 1.35
    # b: empirical coefficient for Net Long-Wave radiation [-]
    b = -0.35
    rnl = calc_rad_long(
        s_rad=s_rad,
        doy=doy,
        t_mean=t_mean,
        t_max=t_max,
        t_min=t_min,
        elevation=elevation,
        lat=lat,
        a=a,
        b=b,
        ea=e_a,
    )  # [MJ/m2/d]
    rn = rns - rnl

    den = dlt + gamma1
    # g: soil heat flux [MJ m-2 d-1]
    g = 0
    num1 = (0.408 * dlt * (rn - g)) / den
    num2 = (gamma * (e_s - e_a) * 900 * wind / (t_mean + 273)) / den
    return num1 + num2
