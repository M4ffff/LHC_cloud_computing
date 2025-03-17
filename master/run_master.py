#!/usr/bin/env python
import pika

print("connection starting")
 
# when RabbitMQ is running on localhost
# params = pika.ConnectionParameters('localhost')

# when RabbitMQ broker is running on network
# params = pika.ConnectionParameters('rabbitmq')


# when starting services with docker compose
params = pika.ConnectionParameters(
    'rabbitmq',
    heartbeat=0)

print("connection started")

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator 
import awkward as ak 
import json
import pickle as pkl
import os 
import infofile
import time
import uproot


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

    data = np.histogram(ak.to_numpy(all_data['data']['mass']), bin_edges )[0] 
    data_errors = np.sqrt( data ) 

    # signal
    signal = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)']['mass'])
    # weights of each signal events
    signal_weights = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)'].totalWeight)
    # colour of signal bar
    signal_color = samples[r'Signal ($m_H$ = 125 GeV)']['color']

    # monte carlo entries
    mc = [] 
    # weights of each entry
    mc_weights = [] 
    # colour and label of each MC bar
    mc_colours = [] 
    mc_labels = [] 


    print("adding samples")


    #### potentially send too?
    # loop over background samples
    for sample in [r'Background $Z,t\bar{t}$', r'Background $ZZ^*$']: 
        mc.append( ak.to_numpy(all_data[sample]['mass']) ) 
        mc_weights.append( ak.to_numpy(all_data[sample].totalWeight) ) 
        mc_colours.append( samples[sample]['color'] ) 
        mc_labels.append( sample ) 

    print("done adding samples")
    print("start plotting")

    #%% Plot data points

    # plot the data points
    ax.errorbar(x=bin_centres, y=data, yerr=data_errors, fmt='ko', label='Data') 

    # plot the Monte Carlo bars
    mc_heights = ax.hist(mc, bins=bin_edges, weights=mc_weights, stacked=True, 
                                color=mc_colours, label=mc_labels )

    # stacked background MC y-axis value
    mc_tot = mc_heights[0][-1]


    # plot signal bars
    signal_heights = ax.hist(signal, bins=bin_edges, bottom=mc_tot, 
                    weights=signal_weights, color=signal_color,
                    label=r'Signal ($m_H$ = 125 GeV)')
    
    # calculate MC statistical uncertainty
    mc_err = np.sqrt(np.histogram(np.hstack(mc), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])

    # plot statistical uncertainty
    ax.bar(bin_centres, 2*mc_err, alpha=0.5, bottom=mc_tot-mc_err, color='none', 
                    hatch="////", width=step_size, label='Stat. Unc.' )


    # set the limits of the axes
    ax.set_xlim( left=xmin, right=xmax ) 
    ax.set_ylim( bottom=0, top=np.amax(data)*1.6 )
    

    # set axes ticks
    ax.xaxis.set_minor_locator( AutoMinorLocator() ) 
    ax.yaxis.set_minor_locator( AutoMinorLocator() ) 
    
    
    # set the axis tick parameters for the main axes
    ax.tick_params(which='both', direction='in', top=True, right=True ) # draw ticks on right axis


    # axis labels
    ax.set_xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]',
                        fontsize=13, x=1, horizontalalignment='right' )
    ax.set_ylabel('Events / '+str(step_size)+' GeV',
                            y=1, horizontalalignment='right') 


    # Add text 'ATLAS Open Data' on plot
    plt.text(0.05, 0.93, 'ATLAS Open Data', transform=ax.transAxes, fontsize=13 ) 

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

    # save figure to container
    output_path = "/output_container/figure2.png"
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path)


    # Signal stacked height
    signal_tot = signal_heights[0] + mc_tot

    # find index of maximum signal
    max_index = np.argmax(signal_heights[0])
    print("max_index: ", max_index)
    
    # Peak of signal
    print("peak: ", signal_tot[max_index])

    # Neighbouring bins
    print("Neighbouring bins: ", signal_tot[max_index-1:max_index+2])

    # Signal and background events
    N_sig = signal_tot[max_index-1:max_index+2].sum()
    N_bg = mc_tot[max_index-1:max_index+2].sum()

    # Signal significance calculation
    signal_significance = N_sig/np.sqrt(N_bg + 0.3 * N_bg**2) 
    print(f"\nResults:\n{N_sig = :.3f}\n{N_bg = :.3f}\n{signal_significance = :.3f}")
    

def publish(dict, routing_key):
    outputs = pkl.dumps(dict)
    channel.basic_publish(exchange='',
                        routing_key=routing_key,
                        body=outputs)



### RECEIVE DATA

# create the connection to broker
connection = pika.BlockingConnection(params)

channel = connection.channel()

# create the queue, if it doesn't already exist
channel.queue_declare(queue='master_to_worker')
channel.queue_declare(queue='worker_to_master')

# num_workers = channel.queue_declare(queue='master_to_worker').method.consumer_count
num_workers = 3
print("NUMBER OF WORKERS: ", num_workers)



#%% SETUP

# units for conversion
MeV = 0.001
GeV = 1.0

# path to data
path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" 


### order changed, so longest/biggest datasets are calculated first
# allows smaller sets to fill in once workers free up
samples = {
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

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'], 
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

    
# Dictionary to hold awkward arrays
all_data = {} 
counter = 0
    

def receive_data(ch, method, properties, outputs):
    global counter
    global timings
    global start
    # print("MASTER receiving some kind of data")
    outputs_dict = pkl.loads(outputs)
    # print(outputs_dict.keys())
    
    sample = outputs_dict["sample"]
    value = outputs_dict["value"]
    # THIS IS A LIST OF LIST OF AWKWARD ARRAYs
    data = outputs_dict["data"]
    entry_stop = outputs_dict["entry_stop"]
    entry_start = outputs_dict["entry_start"]
    
    print(f"received {sample} {value} data, chunk: {entry_start} to {entry_stop}")
    
    if sample in all_data.keys():
        # print("all_data[sample] exists:")
        current_sample = all_data[sample]
        updated_version = ak.concatenate( [current_sample, data[0]] )
        all_data[sample] = ( updated_version )
        
    else:
        # print("all_data[sample] doesn't exist")
        all_data[sample] = (data[0]) 
        # print("length of all_data: ", len(all_data))
        
    #### hmmm
    # print("consuming cancelled?")
    # print("length of all_data: ", len(all_data))
    
    if entry_stop == all_num_entries[sample]:
        print("COUNTER INCREASING")
        counter += 1
        if counter >= 4:
            ch.stop_consuming()
            print("STOPPED CONSUMING")  
            end = time.time()  
            elapsed = end - start
            print(f"{sample} timing: {elapsed:.2f} s")
    

    
    
    
# for sample in samples:
# while len(all_data)
# setup to listen for messages on queue 'master_to_worker'
channel.basic_consume(queue='worker_to_master',
                    auto_ack=True,
                    on_message_callback=receive_data)


print("MASTER consuming")

# channel.start_consuming()
    

all_num_entries = {}
timings = {}
start = time.time() 

for sample in samples: 
    # Print which sample is being processed
    print(f'{sample} samples') 

    # Define empty list to hold data
    frames = [] 

    # print(f"\t{sample}:") 
    start = time.time() 
    timings[sample] = start

    # Loop over each file
    for value in samples[sample]['list']: 
        if sample == 'data': 
            prefix = "Data/" 
        else: 
            prefix = f"MC/mc_{str(infofile.infos[value]['DSID'])}."
        fileString = f"{path}{prefix}{value}.4lep.root" 


        print(f"{sample} {value} being published")
        

        # Open file
        tree = uproot.open(f"{fileString}:mini")
        
        # calculate number of entries in this sample value
        num_entries = tree.num_entries*run_time_speed
        print("number entries: ", num_entries)
        
        # add to dictionary
        all_num_entries[sample] = num_entries
        
        sample_data = []

        # calculate amount of data to send to each worker
        chunk_size = round(num_entries / num_workers)
        print("chunk size: ", chunk_size)
        
        # ensures tiny bits of data aren't sent to multiple workers - send data of at least size min_chunk_size to each worker.  
        min_chunk_size = 2000
        if chunk_size < min_chunk_size:
            chunk_size=min_chunk_size
            print("new chunk size: ", chunk_size)

        # split into chunks depending on numbers of workers, so each worker gets rouhgly same amount of data to analyse
        # chunk_size+1 ensures there is not more chunks than workers. 
        data_chunks = np.arange(0, num_entries, chunk_size+1)
        print("data chunks:", data_chunks)
        
        # send each chunk of data of this sample value to each worker. 
        for chunk_start in data_chunks:
            entry_start = chunk_start
            entry_stop = min([entry_start + chunk_size, num_entries])
            
            # send info
            inputs_dict = {"tree":tree, "variables":variables, "relevant_weights":relevant_weights,
                                "entry_start":entry_start, "entry_stop":entry_stop, "value":value, "sample":sample}
            
            publish(inputs_dict, "master_to_worker")
    


# start listening
channel.start_consuming()
    


print("\nfinal analysis")
final_anal(all_data, samples)
 
print("finished!")


















