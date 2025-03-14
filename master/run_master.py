#!/usr/bin/env python
import pika

print("connection starting")
 
# when RabbitMQ is running on localhost
# params = pika.ConnectionParameters('localhost')

# when RabbitMQ broker is running on network
params = pika.ConnectionParameters('rabbitmq')


# when starting services with docker compose
# params = pika.ConnectionParameters(
#     'rabbitmq',
#     heartbeat=0)

print("connection started")

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator 
import awkward as ak 
import json
import pickle as pkl
import os 

MeV = 0.001
GeV = 1.0

lumi=10


def final_anal(all_data, samples):
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
        # if sample not in ['data', r'Signal ($m_H$ = 125 GeV)']: # if not data nor signal
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
                r'$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+ r' fb$^{-1}$', # text
                transform=ax.transAxes ) # coordinate system used is that of main_axes

    # Add a label for the analysis carried out
    plt.text(0.05, # x
                0.76, # y
                r'$H \rightarrow ZZ^* \rightarrow 4\ell$', # text 
                transform=ax.transAxes ) # coordinate system used is that of main_axes

    # draw the legend
    ax.legend( frameon=False ) # no box around the legend

    plt.show()


    # Signal stacked height
    signal_tot = signal_heights[0] + mc_x_tot

    # Peak of signal
    print("peak: ", signal_tot[8])

    # Neighbouring bins
    print("Neighbouring bins: ", signal_tot[7:10])

    # Signal and background events
    N_sig = signal_tot[7:10].sum()
    N_bg = mc_x_tot[7:10].sum()

    # Signal significance calculation
    signal_significance = N_sig/np.sqrt(N_bg + 0.3 * N_bg**2) 
    print(f"\nResults:\n{N_sig = :.3f}\n{N_bg = :.3f}\n{signal_significance = :.3f}")
    





### RECEIVE DATA


# create the connection to broker
connection = pika.BlockingConnection(params)

channel = connection.channel()

# create the queue, if it doesn't already exist
channel.queue_declare(queue='messages')
channel.queue_declare(queue='sending_all')

# create the queue, if it doesn't already exist
channel.queue_declare(queue='sending_all')



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

# luminosity = 0.5 # fb-1 # data_A only
# luminosity = 1.9 # fb-1 # data_B only
# luminosity = 2.9 # fb-1 # data_C only
# luminosity = 4.7 # fb-1 # data_D only
# luminosity = 10 # fb-1 # data_A,data_B,data_C,data_D

# Set luminosity to 10 fb-1 for all data
luminosity = 10

# Controls the fraction of all events analysed
# change lower to run quicker
run_time_speed = 0.5



def use_all_data(ch, method, properties, frames_json):
    
    # print(f"all data compressed master: {all_data_compressed}")
    
    # all_data_json = gzip.decompress(all_data_compressed)
    frames_dict = json.loads(frames_json.decode('utf-8')) 
    
    sample = frames_dict["sample"]
    frames = frames_dict["frames"]

    all_data[sample] = ak.concatenate(frames)
    

def receive_data(ch, method, properties, outputs):
    
    # outputs_dict = pkl.loads(outputs.decode('utf-8'))
    outputs_dict = pkl.loads(outputs)
    
    sample = outputs_dict["sample"]
    frames = outputs_dict["frames"]
    
    print(f"received {sample} data")
    
    # GATHER DATA FROM WORKERS BEFORE MERGING?
    all_data[sample] = ak.concatenate(frames) # dictionary entry is concatenated awkward arrays
    
    if len(all_data) >= len(samples):
        print("master done consuming.")
        ch.stop_consuming()
    
# Dictionary to hold awkward arrays
all_data = {} 

for sample in samples: 
    # print(f"MASTER publishing {sample}")
    inputs_dict = {"samples_sample":samples[sample], "sample":sample, "path":path, "variables":variables, "relevant_weights":relevant_weights, "run_time_speed":run_time_speed}
    inputs = pkl.dumps(inputs_dict) 

    # MASTER WILL NEED TO SEND MESSAGES TO WORKERS
    # send a simple message
    channel.basic_publish(exchange='',
                        routing_key='messages',
                        body=inputs)


    print(f"MASTER published {sample}")
    
# for sample in samples:
# while len(all_data)
# setup to listen for messages on queue 'messages'
channel.basic_consume(queue='sending_all',
                    auto_ack=True,
                    on_message_callback=receive_data)


print("MASTER consuming")

# start listening
channel.start_consuming()

print("MASTER consumed")

print(len(all_data))

if len(all_data)==4:
    print("final analysis")
    final_anal(all_data, samples)
    
# print("closing channel")

# channel.close()
# 
print("finished?")
