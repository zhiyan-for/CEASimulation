from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.blends import newFuelBlend
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.optimize
from octopus import Fluid, Manifold, Orifice, PropertySource, utils
import math
# make sure you have the datafile in the same directory as the program
data = json.load(open("variables.json"))
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
    return scipy.optimize.fsolve(func=annular_gap_func, x0=np.array(0.167), args=[p_tank - p_chamber])[0]  # 0.178


def mdot_ox_calc(p_tank, p_chamber, orifice: Orifice):
    orifice.manifold.parent._p= p_tank
    return orifice.m_dot_dyer(p_chamber)

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

# default values (example usage)
p_tank = 19e5
p_chamber = 10e5

# find mass flow rates (example usage)
mdot_fuel = mdot_fuel_calc(p_tank, p_chamber)
mdot_ox = mdot_ox_calc(p_tank, p_chamber, ox_orifice)

OF = mdot_ox/mdot_fuel

Isoblend = newFuelBlend(fuelL = ["Isopropanol", "H2O"], fuelPcentL = [80, 20]) #Fuel blend for IPA-water mixture

C = CEA_Obj(oxName = "N2O", fuelName = Isoblend, pressure_units = 'Bar')

tank_pressures = np.linspace(200, 500, 20)

Pamb = 1 #Ambient pressure
chamber_temperature = 2000 #2000K chamber temperature

molecular_mass = 47.6
R = 8.31/molecular_mass
cp = (OF/(1 + OF)) * 60 + (1/(1 + OF)) * 230 
gamma = cp/(cp-R) #Calculated based on mixture ratio and cp values from https://webbook.nist.gov/cgi/cbook.cgi?ID=C67630&Mask=1#Thermo-Gas and https://webbook.nist.gov/cgi/cbook.cgi?ID=C10024972&Mask=1&Type=JANAFG&Table=on#JANAFG

throat_diameter = 0.0193 #Diameter of white dwarf in m
throat_area = 0.0016242785335254294 #Area in m^2

Isp_list=[]
thrust_list = []

mdot_inj = mdot_fuel + mdot_ox #choked mass flow rate is sum of fuel and oxidiser mass flow rates

def mdot_throat_calc(chamber_pressure):
    mdot = ((throat_area * chamber_pressure[0])/np.sqrt(chamber_temperature)) * np.sqrt(gamma/R) * math.pow((gamma + 1)/2, -(gamma + 1)/(2*(gamma - 1)))

    mdot_fuel = mdot_fuel_calc(p_tank, chamber_pressure[0])
    mdot_ox = mdot_ox_calc(p_tank, chamber_pressure[0], ox_orifice)
    mdot_inj = mdot_fuel + mdot_ox

    return mdot - mdot_inj

def chamber_pressure_solver():
    return scipy.optimize.fsolve(func = mdot_throat_calc, x0 = np.array(p_chamber))[0]

chamber_pressure = chamber_pressure_solver()
print(chamber_pressure)

# for p in chamber_pressures:

#     eps_pamb = C.get_eps_at_PcOvPe(Pc=p, MR=mixture_ratio, PcOvPe=p/Pamb) #Design condition nozzle expansion ratio, where flow is choked at the throat and exit pressure is equal to ambient pressure

#     CF_cea, CF_amb, mode = C.get_PambCf(Pamb = Pamb, Pc = p, MR = mixture_ratio, eps = eps_pamb) #thrust coefficient(CF_amb) and flow pattern at exit (mode) for given pressure 

#     thrust = CF_amb*p*throat_area*100000
#     thrust_list.append(thrust)

#     Isp = C.get_Isp(Pc = p, MR = mixture_ratio, eps = eps_pamb)
#     Isp_list.append(Isp)

#     print(f"Pressure: {p}, Thrust: {thrust}, Specific Impulse: {Isp}")
