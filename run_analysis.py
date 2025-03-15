import uproot
import numpy as np # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
# import matplotlib_inline # to edit the inline plot format
# matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg') # to make plots in pdf (vector) format
from matplotlib.ticker import AutoMinorLocator # for minor ticks
import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import vector # for 4-momentum calculations
import time
import uproot
import worker.infofile as infofile



#%% FUNCTIONS

# check lepton type of 4 leptons to check they come in pairs
def check_lepton_type(lepton_type):
    """
    Check there are pairs of the same lepton type 

    Args:
        lepton_type (awk_array): array of lepton types (electron is 11, muon is 13)

    Returns:
        arr(bool): returns False if there are pairs of the same lepton type in an interaction (accepted) 
                   returns True if lepton types do not match (removed) 
    """
    summed_lepton_type = lepton_type[:, 0] + lepton_type[:, 1] + lepton_type[:, 2] + lepton_type[:, 3]
    # 44=4*11,  48=2*11+2*13,  52=4*13
    check_lepton_type = (summed_lepton_type != 44) & (summed_lepton_type != 48) & (summed_lepton_type != 52)
    return check_lepton_type


# Check lepton charge
def check_lepton_charge(lepton_charge):
    """
    Check if lepton charge is conserved in interaction 

    Args:
        lepton_charge (_type_): array of charges of each lepton

    Returns:
        arr(bool): returns False if there are lepton charge is conserved in an interaction (accepted) 
                   returns True if lepton charge is not conserved (removed) 
    """
    summed_lepton_charge = lepton_charge[:, 0] + lepton_charge[:, 1] + lepton_charge[:, 2] + lepton_charge[:, 3] != 0
    return summed_lepton_charge


# Calculate invariant mass of the 4-lepton state
# [:, i] selects the i-th lepton in each event
def calc_mass(lepton_pt, lepton_eta, lepton_phi, lepton_E):
    """
    Calculate invariant mass of state
    
    ### need to do
    Args:
        lepton_pt (_type_): _description_
        lepton_eta (_type_): _description_
        lepton_phi (_type_): _description_
        lepton_E (_type_): _description_

    Returns:
        _type_: _description_
    """
    # groups into four vectors
    p4 = vector.zip({"pt": lepton_pt, "eta": lepton_eta, "phi": lepton_phi, "E": lepton_E})
    # Adds mass of 4 vectors and converts to MeV
    invariant_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV 
    return invariant_mass


### redo
def calc_weight(relevant_weights, sample, events):
    info = infofile.infos[sample]
    xsec_weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    total_weight = xsec_weight 
    for variable in relevant_weights:
        total_weight = total_weight * events[variable]
    return total_weight





#%% SETUP

# units for conversion
MeV = 0.001
GeV = 1.0

# path to data
path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" 


### redo
samples = {
    'data': {
        'list' : ['data_A','data_B','data_C','data_D'], 
    },

    r'Background $Z,t\bar{t}$' : { # Z + ttbar
        'list' : ['Zee','Zmumu','ttbar_lep'],
        'color' : "#6b59d3" # purple
    },

    r'Background $ZZ^*$' : { # ZZ
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },

    r'Signal ($m_H$ = 125 GeV)' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep','VBFH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    },
}

#%%



# Important variables used in analysis
variables = ['lep_pt','lep_eta','lep_phi','lep_E','lep_charge','lep_type']
relevant_weights = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]



#%%

lumi = 0.5 # fb-1 # data_A only
# lumi = 1.9 # fb-1 # data_B only
# lumi = 2.9 # fb-1 # data_C only
# lumi = 4.7 # fb-1 # data_D only
# lumi = 10 # fb-1 # data_A,data_B,data_C,data_D

# Set luminosity to 10 fb-1 for all data
lumi = 10


# Controls the fraction of all events analysed
# change lower to run quicker
run_time_speed = 1


# Dictionary to hold awkward arrays
all_data = {} 


for sample in samples: 
    # Print which sample is being processed
    print(f'{sample} samples') 

    # Define empty list to hold data
    frames = [] 

    # Loop over each file
    for value in samples[sample]['list']: 
        if sample == 'data': 
            prefix = "Data/" 
        else: 
            prefix = f"MC/mc_{str(infofile.infos[value]['DSID'])}."
        fileString = f"{path}{prefix}{value}.4lep.root" 


        print(f"\t{value}:") 
        start = time.time() 

        # Open file
        tree = uproot.open(f"{fileString}:mini")
        
        sample_data = []

        # Loop over data in the tree
        for data in tree.iterate(variables + relevant_weights, library="ak", 
                                 entry_stop=tree.num_entries*run_time_speed, step_size = 1000000):
             
            # Number of events in this batch
            nIn = len(data) 
                                 
            # Record transverse momenta (see bonus activity for explanation)
            data['leading_lep_pt'] = data['lep_pt'][:,0]
            data['sub_leading_lep_pt'] = data['lep_pt'][:,1]
            data['third_leading_lep_pt'] = data['lep_pt'][:,2]
            data['last_lep_pt'] = data['lep_pt'][:,3]

            # Cuts
            lepton_type = data['lep_type']
            data = data[~check_lepton_type(lepton_type)]
            lepton_charge = data['lep_charge']
            data = data[~check_lepton_charge(lepton_charge)]
            
            # calculate invariant mass
            data['mass'] = calc_mass(data['lep_pt'], data['lep_eta'], data['lep_phi'], data['lep_E'])

            # Store Monte Carlo weights in the data
            if 'data' not in value: # Only calculates weights if the data is MC
                data['totalWeight'] = calc_weight(relevant_weights, value, data)
                nOut = sum(data['totalWeight']) # sum of weights passing cuts in this batch 
            else:
                nOut = len(data)
            elapsed = time.time() - start # time taken to process
            print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after
            calls_per_sec = nIn / elapsed
            print(f"\t\t calls per second: {calls_per_sec:2f}")

            # Append data to the whole sample data list
            sample_data.append(data)

        frames.append(ak.concatenate(sample_data)) 


    # GATHER DATA FROM WORKERS BEFORE MERGING?
    all_data[sample] = ak.concatenate(frames) # dictionary entry is concatenated awkward arrays


print(len(all_data))
# print((all_data).shape)

# PLOTTING

#%%

# x-axis range of the plot
xmin = 80 * GeV
xmax = 250 * GeV

# Histogram bin setup
step_size = 5 * GeV

bin_edges = np.arange(xmin, xmax+step_size, step_size ) 
bin_centres = np.arange(xmin+step_size/2, xmax+step_size/2, step_size ) 


fig, ax = plt.subplots(figsize=(15,5)) 


real_data = np.histogram(ak.to_numpy(all_data['data']['mass']), bin_edges )[0] # histogram the data
real_data_errors = np.sqrt( real_data ) # statistical error on the data

# signal
signal_x = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)']['mass'])
# weights of signal events
signal_weights = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)'].totalWeight)
# colour of signal bar
signal_color = samples[r'Signal ($m_H$ = 125 GeV)']['color']

mc_x = [] # define list to hold the Monte Carlo histogram entries
mc_weights = [] # define list to hold the Monte Carlo weights
mc_colors = [] # define list to hold the colors of the Monte Carlo bars
mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars


print("adding samples")

for sample in [r'Background $Z,t\bar{t}$', r'Background $ZZ^*$']: # loop over samples
    mc_x.append( ak.to_numpy(all_data[sample]['mass']) ) # append to the list of Monte Carlo histogram entries
    mc_weights.append( ak.to_numpy(all_data[sample].totalWeight) ) # append to the list of Monte Carlo weights
    mc_colors.append( samples[sample]['color'] ) # append to the list of Monte Carlo bar colors
    mc_labels.append( sample ) # append to the list of Monte Carlo legend labels

print("done adding samples")
print("start plotting")

#%% Plot data points

# plot the data points
ax.errorbar(x=bin_centres, y=real_data, yerr=real_data_errors,
                    fmt='ko', # 'k' means black and 'o' is for circles 
                    label='Data') 


#%% Plot background mc signals

# plot the Monte Carlo bars
mc_heights = ax.hist(mc_x, bins=bin_edges, 
                            weights=mc_weights, stacked=True, 
                            color=mc_colors, label=mc_labels )

mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value

# calculate MC statistical uncertainty: sqrt(sum w^2)
mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])



#%% plot higgs signal

# plot the signal bar
signal_heights = ax.hist(signal_x, bins=bin_edges, bottom=mc_x_tot, 
                weights=signal_weights, color=signal_color,
                label=r'Signal ($m_H$ = 125 GeV)')

# plot the statistical uncertainty
ax.bar(bin_centres, # x
                2*mc_x_err, # heights
                alpha=0.5, # half transparency
                bottom=mc_x_tot-mc_x_err, color='none', 
                hatch="////", width=step_size, label='Stat. Unc.' )



#%% figure configuration

# set the x-limit of the main axes
ax.set_xlim( left=xmin, right=xmax ) 

# separation of x axis minor ticks
ax.xaxis.set_minor_locator( AutoMinorLocator() ) 

# set the axis tick parameters for the main axes
ax.tick_params(which='both', # ticks on both x and y axes
                        direction='in', # Put ticks inside and outside the axes
                        top=True, # draw ticks on the top axis
                        right=True ) # draw ticks on right axis

# x-axis label
ax.set_xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]',
                    fontsize=13, x=1, horizontalalignment='right' )

# write y-axis label for main axes
ax.set_ylabel('Events / '+str(step_size)+' GeV',
                        y=1, horizontalalignment='right') 

# set y-axis limits for main axes
ax.set_ylim( bottom=0, top=np.amax(real_data)*1.6 )

# add minor ticks on y-axis for main axes
ax.yaxis.set_minor_locator( AutoMinorLocator() ) 

# Add text 'ATLAS Open Data' on plot
plt.text(0.05, # x
            0.93, # y
            'ATLAS Open Data', # text
            transform=ax.transAxes, # coordinate system used is that of main_axes
            fontsize=13 ) 

# Add text 'for education' on plot
plt.text(0.05, # x
            0.88, # y
            'for education', # text
            transform=ax.transAxes, # coordinate system used is that of main_axes
            style='italic',
            fontsize=8 ) 

# Add energy and luminosity
lumi_used = str(lumi*run_time_speed) # luminosity to write on the plot
plt.text(0.05, # x
            0.82, # y
            '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
            transform=ax.transAxes ) # coordinate system used is that of main_axes

# Add a label for the analysis carried out
plt.text(0.05, # x
            0.76, # y
            r'$H \rightarrow ZZ^* \rightarrow 4\ell$', # text 
            transform=ax.transAxes ) # coordinate system used is that of main_axes

# draw the legend
ax.legend( frameon=False ) # no box around the legend

plt.show()






#%% significance

# Signal stacked height
signal_tot = signal_heights[0] + mc_x_tot

# Peak of signal
print(signal_tot[8])

# Neighbouring bins
print(signal_tot[7:10])

# Signal and background events
N_sig = signal_tot[7:10].sum()
N_bg = mc_x_tot[7:10].sum()

# Signal significance calculation
signal_significance = N_sig/np.sqrt(N_bg + 0.3 * N_bg**2) 
print(f"\nResults:\n{N_sig = :.3f}\n{N_bg = :.3f}\n{signal_significance = :.3f}")

#%%




