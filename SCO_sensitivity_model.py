###Sensitivity analysis###
#produces simulations using cometspy

import cometspy as c
import cobra.io
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import math
import random



#load model 
sco_model = cobra.io.read_sbml_model('Sco-GEM.xml')

#pathway to environemental variables for COMETS to work 
os.environ['GUROBI_HOME'] = '/Library/gurobi951/macos_universal2'
os.environ['COMETS_HOME'] = '/Applications/COMETS'
os.environ['GUROBI_COMETS_HOME'] = '/Library/gurobi951/macos_universal2/'

#objective function
sco_model.objective = "BIOMASS_SCO"

#replicate the COMETS medium
medium = ['ca2_e', 'cl_e', 'cu2_e', 'fe2_e', 'h_e','mn2_e', 'mobd_e', 'na1_e', 'cobalt2_e', 'zn2_e', 'k_e', 'na1_e', 
                  'ni2_e', 'tungs_e', 'fe3_e', 'h2o_e', 'mg2_e', 'so4_e','pi_e','o2_e',
         'glc__D_e', 'glu__L_e']

sco_model.medium = {"EX_" + key : 1. for key in medium}




#Load model
Sco = c.model(sco_model) #use loaded model to build a cobra model

Sco.open_exchanges() #removes the bounds of everything

#set the initial biomass
Sco.initial_pop = [0, 1, 5e-5]

Sco.change_bounds('ATPM', 0, 1000)

#set as pFBA - maximises the objective function whilst minimising the total flux
Sco.obj_style = "MAX_OBJECTIVE_MIN_TOTAL"




#Create a layout 
test_tube = c.layout([Sco])

## air it 1x1 (0x0)
## media is 1x2 (0x1) 
test_tube.grid = [1, 2]

#add the metabolites - this sets the bounds for the metabolites

##Medium

test_tube.set_specific_metabolite_at_location('o2_e', (0,1), 0.0002188) #concentration of oxygen in water (in mmol) at 30oC (Patel and Vashi, 2015)
test_tube.set_specific_metabolite_at_location('co2_e', (0,1), 0.0284) #carbon dioxide conc in water (in mmol) at 30oC (Engineering ToolBox, 2008)


#trace elements - calcium, chloride, copper, iron ii, hydrogen, manganese, molybdate, sodium, cobalt, zinc, potassium
#sodium, ammonium, tungstate, iron iii, water, magesium, sulfate
trace_elements = ['ca2_e', 'cl_e', 'cu2_e', 'fe2_e', 'h_e','mn2_e', 'mobd_e', 'na1_e', 'cobalt2_e', 'zn2_e', 'k_e', 
                  'ni2_e', 'tungs_e', 'fe3_e', 'h2o_e', 'mg2_e', 'so4_e']

for i in trace_elements: 
    test_tube.set_specific_metabolite_at_location(i, (0,1), 10) #set as unlimited
    test_tube.set_specific_static_at_location(i, (0,1), 10)


#in line with Nieselt et al., (2010) - phosphate minimal media
test_tube.set_specific_metabolite_at_location('pi_e', (0,1), 0.00828) #phosphate 8.28 x 10^-3  mmol/ml
test_tube.set_specific_metabolite_at_location('glc__D_e', (0,1), 0.39965) #glucose 399.65 x 10^-3 mmol/ml
test_tube.set_specific_metabolite_at_location('glu__L_e', (0,1), 0.58542) #glutamate at 585.42 x 10^-3 mmol/ml


##Air

#o2 and co2 are set to a constant in the air layer
#creates a diffusion gradient as they are used/produced in the media layer 
test_tube.set_specific_static_at_location('o2_e', (0, 0), 0.0002188)
test_tube.set_specific_static_at_location('co2_e', (0, 0), 0.0284)

test_tube.set_specific_static_at_location('pi_e', (0,0), 0)
test_tube.set_specific_static_at_location('glc__D_e', (0,0), 0)
test_tube.set_specific_static_at_location('glu__L_e', (0,0), 0)

#Set the diffusion constants of the metabolites

test_tube.set_specific_metabolite_diffusion('o2_e', 2.3e-5) #Diffusion constant 2 in water at at 30oC in cm^2/s (Jordan and Bauer, 1959)
test_tube.set_specific_metabolite_diffusion('co2_e', 2.3e-5)

#these cannot diffuse out of the media
test_tube.set_specific_metabolite_diffusion('pi_e', 0.0)
test_tube.set_specific_metabolite_diffusion('glc__D_e', 0.0)
test_tube.set_specific_metabolite_diffusion('glu__L_e', 0.0)




#set the parameters
sim_params = c.params()

sim_params.set_param('defaultVmax', 10) #this will control the uptake rate for the reactions
sim_params.set_param('defaultKm', 0.01) 
sim_params.set_param('maxCycles', 800)
sim_params.set_param('timeStep', 0.1) #hours
sim_params.set_param('spaceWidth', 1)
sim_params.set_param('maxSpaceBiomass', 10)
sim_params.set_param('minSpaceBiomass', 1e-11)
sim_params.set_param('writeMediaLog', True)
sim_params.set_param('writeBiomassLog', True)
sim_params.set_param('writeFluxLog', True)
sim_params.set_param('defaultDiffConst', 0) 



#parameters from the closest simulation
N = 100 #number of different simulations

#in mmol
glc_Vmax = 8.987181
glc_Km = 0.001201

glu_Vmax = 8.856089
glu_Km = 9.166123e-07

pi_Vmax = 16.997138
pi_Km = 0.000217

co2_Vmax = 0.000986
co2_Km = 0.000176

o2_Km = 0.000001


#o2 Vmax parameters sampled from a range 
o2_Vmax = random.sample(range(0, 100), N) #in nmol
o2_Vmax = np.array(o2_Vmax)
o2_Vmax = o2_Vmax/1e6 #in mmol




#save differnet o2 vmax values from the range
o2_params =  pd.DataFrame({'o2_Vmax': o2_Vmax})
        
o2_params.to_csv('o2_Vmax_params.csv')



all_sim_res = [] #empty vector containing results

for i in range(0, N):
    Sco.change_vmax('EX_glc__D_e', glc_Vmax) #mmol (gCDW)-1 (hour)-1
    Sco.change_km('EX_glc__D_e', glc_Km) #in mmol (cm3)-1
    Sco.change_vmax('EX_glu__L_e', glu_Vmax) 
    Sco.change_km('EX_glu__L_e', glu_Km)
    Sco.change_vmax('EX_pi_e', pi_Vmax)
    Sco.change_km('EX_pi_e', pi_Km)
    Sco.change_vmax('EX_o2_e', o2_Vmax[i]) #only changes the o2 Vmax
    Sco.change_km('EX_o2_e', o2_Km)
    Sco.change_vmax('EX_co2_e', co2_Vmax)
    Sco.change_km('EX_co2_e', co2_Km)
    experiment = c.comets(test_tube, sim_params)
    experiment.run()
    all_sim_res.append(experiment)
    
    
    if i == 24 or i == 49 or i == 74:
        print("\n", "Simulation", i+1, "Done") 
    
    if i == 99:
        print("All simulations done")
        
        
#saving results I want

#total biomass
all_biomass_res = pd.concat([all_sim_res[i].total_biomass for i in range(0,N)], keys = [all_sim_res[i] for i in range(0,N)])
all_biomass_res['time (h)'] = all_biomass_res['cycle']*experiment.parameters.all_params['timeStep']
all_biomass_res.to_csv('o2_Vmax_total_biomass.csv')


#media 
all_media_res = pd.concat([all_sim_res[i].media for i in range(0, N)], keys = [all_sim_res[i] for i in range(0, N)])
#key metabolites in the media from all simulations 
all_media_res = all_media_res.loc[(all_media_res['x'] == 1) & (all_media_res['y'] == 2)] #metabolites in media
key_mets_res = all_media_res.loc[(all_media_res['metabolite']=='o2_e') | (all_media_res['metabolite']=='co2_e') | (all_media_res['metabolite']=='glc__D_e') | (all_media_res['metabolite']=='glu__L_e') | (all_media_res['metabolite']=='pi_e')]
key_mets_res.to_csv('o2_Vmax_metabolites.csv')


#o2 flux into media
all_flux_res = pd.concat([all_sim_res[i].fluxes_by_species.get('Sco_GEM_v1_3_1') for i in range(0,N)], keys = [all_sim_res[i] for i in range(0,N)])
EX_o2_flux = all_flux_res[['cycle', 'x', 'y', 'EX_o2_e']]
EX_o2_flux.to_csv('o2_Vmax_o2_flux.csv')


