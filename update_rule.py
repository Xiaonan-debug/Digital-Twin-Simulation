import math
import random
import torch
import torch.nn as nn
from collections import deque

HOURLY_VASCULAR_RESISTANCE_CHANGE = .0208 #See 2.16.21 both SNs: the natural decline is .5 mmHg/mL/min / 24 hours
VASCULAR_RESISTANCE_STOCHASTIC_FACTOR = 1.0052 # VR can change stochastically an additional +25%
VASCULAR_RESISTANCE_DELTA_T_BASE = .981 # VR decreases almost 2% per degree increase
PERFUSATE_LITERS = 1
GRAFT_GRAMS = 300 # Normal estimate
ANAEROBIC_METABOLIC_FRACTION = 0.7
aerobicMetabolicFraction = 1 - ANAEROBIC_METABOLIC_FRACTION
hematocritInitial = 13.75
transfusionHours = 12
HOURLY_HEMATOCRIT_CHANGE = 1   # Usually it is 1
GLUCOSE_CONSUMPTION_MMOLE = .0043 # From our 2021 and 2022 experimental data (per minute per 100g)
INITIAL_PRESSURE_MMHG = 80
ONE_LPM_ARTERY_GAS_PRESSURE_FRACTION = .5 # Empirical. Could be as low as 0.2
HOURLY_CO2_LOAD_MMHG = 13.4 # Empirical at 0 lpm sweep gas. declines to 0 at 1lpm
PCO2_EQULIBRIUM_AT_ONELPM = 5.7 # Empirical 

# Initialize constants
GLUCOSE_MOLECULAR_WT = 180.156
SOLUBILITY_CO2 = .03 # mM/mmHg
PK = 6.1 # Used in equation to convert CO2 and bicarb to pH
LACTATES_PER_GLUCOSE = 2
OXYGENS_PER_GLUCOSE = 6
CO2S_PER_GLUCOSE = 6
INSULIN_PER_MMOLE_GLUCOSE = 19.5 # mUnits
ACTION_DIMENSION = 7


# ----- PERFUSION AND PHYSIOLOGY FUNCTIONS -----
# ----- A. HEMATOCRIT -----
# Calculate new Hct on the basis of hourly "corrosion"
def NewHct(Hct):
    Hct = Hct - HOURLY_HEMATOCRIT_CHANGE
    if Hct < 0:
        Hct = 0
    return Hct

# ----- B. GLUCOSE -----
# Calculate glucose level based on temperature derated consumption
def ConsumedGlucose(mass, temperature):
    glumMolesConsumed = GLUCOSE_CONSUMPTION_MMOLE * (mass / 100) * 60 * pow(2,(temperature - 37) / 10)                                                                            
    return glumMolesConsumed

# ----- C. LACTATE -----
def NewLactate(glumMolesCons):
    lacmMoles = glumMolesCons * ANAEROBIC_METABOLIC_FRACTION * LACTATES_PER_GLUCOSE
    return lacmMoles

# ----- D. ARTERIAL OXYGEN -----
# D1. Calculate new pO2 - assumes partial pressure of oxygen equilbrates between gas and perfusate i.e oxygenator has significantly excess capacity
def NewPO2(gasFlow, richness):
    FiO2 = (.95 * richness) + (.2 * (1 - richness))
    gasPO2 = FiO2 * 760
    OneLpmArtPO2 = gasPO2 * ONE_LPM_ARTERY_GAS_PRESSURE_FRACTION
    pO2 = OneLpmArtPO2 * gasFlow # Max concentration of gas in blood at 1 lpm, declinining linearly to 0 at 0 lpm
    return pO2

# D2. Calculate oxygen saturation as a function of pO2(see Serianni on Hill, human, 37C, pH 7.4)
def SaturationO2(pO2):
    SaOxygen = (pow((.13534 * pO2),2.62)) / ((pow((.13534 * pO2),2.62)) + 27.4)
    return SaOxygen

# D3. Calculate oxygen content in blood considering SaO2, pO2 and Hct
def ConcentrationaO2(pO2, Hct, SaOxygen):
    HbGperdL = 0.34 * Hct 
    CaOxygen = (SaOxygen * HbGperdL * 1.36) + ( .0031 * pO2) # in mL/dL
    return CaOxygen


# ----- E. VENOUS OXYGEN -----
# E1. Calculate immediate venous oxygen concentration
def ConcentrationvO2 (o2RateOut, perfusionFlowmLpm):
    CvOxygen = o2RateOut / perfusionFlowmLpm *1000 # This is millimolar
    if CvOxygen < 0:
        CvOxygen = 0
    return CvOxygen


# E2. Estimate immediate venous oxygen saturation (see Madan)
# If pO2 is below 40 then simply multiply pO2 by 2 to estimate SvO2
# Using this relationship we can estimate SvO2 from CvO2 assuming SvO2 < 80%
# Calculate the inverse Concentration equation
def VsaturationO2(CvOxygen, Hct):
    HbGperdL = 0.34 * Hct
    SvOxygen = (CvOxygen /((HbGperdL * 1.34) + (.0031 / 2)))
    if SvOxygen > 1:
        SvOxygen = 1
    return SvOxygen


# ----- F. HEMODYNAMICS -----
# Calculate new vascular resistance
def NewVR(oldVR, temperature, vasodilator, lastTemperature):
    # Step 1: Age the VR unless vasodilator is given in which case keep VR the same. Then allow VR to change stochastically 25% of hourly change
    localVR = oldVR + HOURLY_VASCULAR_RESISTANCE_CHANGE * int(not(vasodilator)) * random.uniform(1, VASCULAR_RESISTANCE_STOCHASTIC_FACTOR)
    # Step 2: Adjust VR due to temperature
    localVR = localVR * pow(VASCULAR_RESISTANCE_DELTA_T_BASE,(lastTemperature - temperature)) # VR = VR * base^(-deltaT) as T increases VR decreases
    return localVR


# ----- G. ARTERIAL CO2 -----
def NewPCO2(gasFlow, richness):
    FiCO2 = .05 * richness
    pCO2 = FiCO2 * 760 + PCO2_EQULIBRIUM_AT_ONELPM # Different frrom oxygen: always blow-off excess CO2 and get back to FiCO2 level
    return pCO2


# ----- I. VENOUS PH -----
def NewpH(bcb,pCO2):
    localpH = PK + math.log((bcb/(SOLUBILITY_CO2 * pCO2)),10)
    return localpH


# # 定义Q-Network
# class QNetwork(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


# # 定义经验回放缓冲区
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state):
#         self.buffer.append((state, action, reward, next_state))

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

#     def __len__(self):
#         return len(self.buffer)

