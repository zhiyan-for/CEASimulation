import json
import math

import numpy as np
import scipy.optimize
from octopus import Fluid, Manifold, Orifice, PropertySource, utils
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.blends import newFuelBlend

data = json.load(open("variables.json"))

Isoblend = newFuelBlend(fuelL = ["Isopropanol", "H2O"], fuelPcentL = [80, 20])
C = CEA_Obj(oxName = "N2O", fuelName= Isoblend, pressure_units= 'Pa', specific_heat_units='kJ/kg-K', density_units = 'kg/m^3')

# header function for the solver (can copy this format for other solver functions)
def annular_gap_func(mdot, dp):
    Ao = data["area_reduction_ratio"] * 0.25 * np.pi * (data["pintle_gap_OD"] ** 2 - data["pintle_sleeve_OD"] ** 2)
    return dp - utils.dp_annular_gap(
        D_outer=data["pintle_gap_OD"],
        D_inner=data["pintle_sleeve_OD"],
        mdot=mdot,
        L=data["pintle_gap_length"],
        rho=data["fuel_density"],
        mu=data["fuel_viscosity"]
    ) - (0.5 * mdot ** 2) / (data["fuel_density"] * Ao ** 2)


# wrapped functions for mass flow rate as a function of upstream and downstream pressure
def mdot_fuel_calc(p_tank, p_chamber):
    return scipy.optimize.fsolve(func=annular_gap_func, x0=np.array(0.167), args=(p_tank - p_chamber))[0]  # 0.178


def mdot_ox_calc(p_tank, p_chamber, orifice: Orifice):
    orifice.manifold.parent._p = p_tank
    return orifice.m_dot_dyer(p_chamber)


def mdot_nozzle_calc(p_chamber, OF):
    CHAMBER_TEMP = 2000
    (MOLECULAR_MASS, gamma) = C.get_Chamber_MolWt_gamma(p_chamber, OF)
    cp = C.get_Chamber_Cp(p_chamber, OF) #in kJ/kg-K
    R = cp - (cp/gamma)

    return ((data["throat_area"] * p_chamber) / np.sqrt(CHAMBER_TEMP)) * np.sqrt(gamma / R) * math.pow(
        (gamma + 1) / 2, -(gamma + 1) / (2 * (gamma - 1)))


def mass_flow_diff(p_chamber, p_tank, orifice: Orifice):
    m_f = mdot_fuel_calc(p_tank, p_chamber)
    m_o = mdot_ox_calc(p_tank, p_chamber, orifice)
    OF = m_o / m_f
    m_n = mdot_nozzle_calc(p_chamber, OF)
    return m_n - m_f - m_o


def main():
    # DO NOT EDIT -- MUST INCLUDE THIS INIT #
    # precalculating fuel flow area from datafile
    ox_area = data["slot_width"] * data["num_slots"] * (
            data["slot_height"] - data["pintle_sleeve_thickness"] * np.sin(data["alpha"])
    )
    # setup octopus sim for oxidiser
    oxidiser = Fluid(name="NitrousOxide", eos="HEOS")
    ox_property_source = PropertySource(p=19e5, T=data["tank_temp"])
    oxidiser.set_state(P=19e5, T=data["tank_temp"])
    ox_manifold = Manifold(fluid=oxidiser, parent=ox_property_source, A=1)
    ox_orifice = Orifice(manifold=ox_manifold, A=ox_area, Cd=0.7)
    # END INIT #
    chamber_pressure = scipy.optimize.fsolve(func=mass_flow_diff, x0=np.array(10e5), args=(19e5, ox_orifice))[0]

    print(chamber_pressure)

if __name__ == "__main__":
    main()