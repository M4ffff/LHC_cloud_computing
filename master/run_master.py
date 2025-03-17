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


    #### potentially send too?
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

    output_path = "/output_container/figure.png"
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path)


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
    

def publish(dict, sample, routing_key, container):
    
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

num_workers = channel.queue_declare(queue='master_to_worker').method.consumer_count
print("NUMBER OF WORKERS: ", num_workers)



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
run_time_speed = 1

    
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
        
        num_entries = tree.num_entries*run_time_speed
        all_num_entries[sample] = num_entries
        
        
        sample_data = []

        # calculate amount of data to send to each worker
        chunk_size = round(num_entries / num_workers)
        print("number entries: ", num_entries)
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
            
            publish(inputs_dict, sample, "master_to_worker", "MASTER")
    


# start listening
channel.start_consuming()
    


print("\nfinal analysis")
final_anal(all_data, samples)
 
print("finished!")


















