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
import docker

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

    #### potentially send too?
    # loop over background samples
    for sample in [r'Background $Z,t\bar{t}$', r'Background $ZZ^*$']: 
        mc.append( ak.to_numpy(all_data[sample]['mass']) ) 
        mc_weights.append( ak.to_numpy(all_data[sample].totalWeight) ) 
        mc_colours.append( samples[sample]['color'] ) 
        mc_labels.append( sample ) 

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
    

publish_counter = 0

def publish(dict, routing_key):
    global publish_counter
    publish_counter += 1
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








#%% SETUP

# units for conversion
MeV = 0.001
GeV = 1.0

# path to data
path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" 


### order changed, so short sample calculate first to ensure all workers are activated
### the longest/biggest datasets are calculated next
### allows smaller sets to fill in once workers free up
samples = {
    r'Background $ZZ^*$' : { # ZZ
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },
    
    r'Signal ($m_H$ = 125 GeV)' : { # H -> ZZ -> llll
        'list' : ['VBFH125_ZZ4lep','ggH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    },
    
    r'Background $Z,t\bar{t}$' : { # Z + ttbar
        'list' : ['Zee','Zmumu','ttbar_lep'],
        'color' : "#6b59d3" # purple
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

# Set luminosity to 10 fb-1 for all data
luminosity = 10

# Controls the fraction of all events analysed
# change lower to run quicker
run_time_speed = 1

    
# Dictionary to hold awkward arrays
all_data = {} 
counter = 0
worker_logs = {}
worker_ids = []
worker_run_times = {}
 
all_values = []

def receive_data(ch, method, properties, outputs):
    global counter
    global publish_counter
    global timings
    global start
    counter += 1
    
    worker_id = properties.app_id
    
    # print("MASTER receiving some kind of data")
    outputs_dict = pkl.loads(outputs)
    
    sample = outputs_dict["sample"]
    value = outputs_dict["value"]
    # THIS IS A LIST OF LIST OF AWKWARD ARRAYs
    data = outputs_dict["data"]
    entry_stop = outputs_dict["entry_stop"]
    entry_start = outputs_dict["entry_start"]
    worker_log = outputs_dict["worker_log"]
    start_time = outputs_dict["start"]
    
    if value not in all_values:
        all_values.append(value)
    
    if worker_id not in worker_ids:
        worker_ids.append(worker_id)
    
    worker_logs[worker_id] = worker_log
    
    
    print(f"received {sample} {value} data, chunk: {entry_start} to {entry_stop}")
    
    if sample in all_data.keys():
        # print("all_data[sample] exists:")
        current_sample = all_data[sample]
        updated_version = ak.concatenate( [current_sample, data[0]] )
        all_data[sample] = ( updated_version )
        
    else:
        all_data[sample] = (data[0]) 
            
     
    end = time.time()
    elapsed = end - start_time
    print(f"{sample} {value} done in {elapsed} second ")
    
    # if this worker already has some entries, get these entries
    if worker_id in worker_run_times.keys():
        new_worker_run_time = worker_run_times[worker_id]
    # else, make an empty dictionary to fill
    else:
        new_worker_run_time = {}
    
    # if this worker already has an entry for this specific vlaue, extend teh time for this value
    if value in new_worker_run_time.keys():
        new_worker_run_time[value] += elapsed
    # else, set this vlaue to the elapsed time
    else:
        new_worker_run_time[value] = elapsed
        
    # add updated dictionary for this worker
    worker_run_times[worker_id] = new_worker_run_time
    
    print(f"received {counter} of {publish_counter} messages ")
    if counter >= publish_counter:
        ch.stop_consuming()
        print("STOPPED CONSUMING")  
        
    print(f"{sample} timing: {elapsed:.2f} s")
    

# one second wait to ensure containers have activated
time.sleep(1)
print("pause done")


start3 = time.time()
# Determine number of available workers
num_workers = channel.queue_declare(queue='master_to_worker', passive=True).method.consumer_count
end3 = time.time()
print(f"time to check number of workers: {end3 - start3:.5f} seconds")

# num_workers = 3
print("FETCHED NUMBER OF WORKERS: ", num_workers) 


    
# for sample in samples:
# while len(all_data)
# setup to listen for messages on queue 'master_to_worker'
channel.basic_consume(queue='worker_to_master',
                    auto_ack=True,
                    on_message_callback=receive_data)




# channel.start_consuming()
    

all_num_entries = {}
timings = {}
start = time.time() 

for i,sample in enumerate(samples): 
    # Print which sample is being processed
    # print(f'\n{sample}') 
    
    # update number of workers in case any activated after initial count. 
    num_workers = channel.queue_declare(queue='master_to_worker', passive=True).method.consumer_count

    # Define empty list to hold data
    frames = [] 

    # print(f"\t{sample}:") 
    # start = time.time() 
    # timings[sample] = start

    # Loop over each file
    for value in samples[sample]['list']: 
        if sample == 'data': 
            prefix = "Data/" 
        else: 
            prefix = f"MC/mc_{str(infofile.infos[value]['DSID'])}."
        fileString = f"{path}{prefix}{value}.4lep.root" 


        # Open file
        tree = uproot.open(f"{fileString}:mini")
        
        # calculate number of entries in this sample value
        num_entries = tree.num_entries*run_time_speed
        # print(f"{sample} {value}"  )
        
        # add to dictionary
        all_num_entries[sample] = num_entries
        
        sample_data = []

        # calculate amount of data to send to each worker
        # +1 ensures chunk_size*num_workers > num_entries
        chunk_size = round(num_entries / (num_workers))+1
        
        # ensures tiny bits of data aren't sent to multiple workers - send data of at least size min_chunk_size to each worker.  
        min_chunk_size = 10000
        if chunk_size < min_chunk_size:
            chunk_size=min_chunk_size

        # print("chunk size: ", chunk_size)

        print(f"\n{sample} {value} being published")

        # split into chunks depending on numbers of workers, so each worker gets rouhgly same amount of data to analyse
        # chunk_size+1 ensures there is not more chunks than workers. 
        data_chunks = np.arange(0, num_entries, chunk_size)
        print(f"\t number entries: {num_entries}     chunk size: {chunk_size}  ")
        print(f"\t data chunks: {data_chunks} \n")
        
        # send each chunk of data of this sample value to each worker. 
        for chunk_start in data_chunks:
            entry_start = chunk_start
            entry_stop = min([entry_start + chunk_size, num_entries])
            
            # send info
            start_time = time.time()
            inputs_dict = {"tree":tree, "variables":variables, "relevant_weights":relevant_weights,
                                "entry_start":entry_start, "entry_stop":entry_stop, "value":value, "sample":sample, "start":start_time}
            
            publish(inputs_dict, "master_to_worker")
            
        # channel.basic_get(queue='worker_to_master')


print("MASTER consuming")

# start listening
channel.start_consuming()
    


fig,ax = plt.subplots(1,3, figsize=(16,6))

# value_names = []

for i in range(num_workers):
    
    current_worker = worker_ids[i]
    
    log = worker_logs[current_worker]
    # print(log.keys())
    # print(list(log.keys())[1])
    # print(log)
    bottom_len = 0
    bottom_time = 0
    # for sample in samples:
    for j, value in enumerate(log["value name list"]):
            # if value not in value_names:
            #     value_names.append(value)
                
            # colour = value_names.index(value)
            colour = all_values.index(value)
            
            height_len = log['nin list'][j]
            height_time = log['elapsed list'][j]
            # print(f"{bottom_len} to {height_len+bottom_len}")
            ax[0].bar(i, height_len, bottom=bottom_len, color=f"C{colour}")
            ax[1].bar(i, height_time, bottom=bottom_time, color=f"C{colour}")
            bottom_len += height_len
            bottom_time += height_time
                
          
for i in range(num_workers):
    
    current_worker = worker_ids[i]
    
    log = worker_run_times[current_worker]
    # print(log)
    # bottom_len = 0
    bottom_time = 0
    # for sample in samples:
    for value in log.keys():
            # if value not in value_names:
            # value_names.append(value)
                
            colour = all_values.index(value)
            
            # height_len = log[value]['len']
            height_time = log[value]
            # print(f"{bottom_len} to {height_len+bottom_len}")
            # ax[0].bar(i, height_len, bottom=bottom_len, label=value, color=f"C{colour}")
            ax[2].bar(i, height_time, bottom=bottom_time, color=f"C{colour}")
            # bottom_len += height_len
            bottom_time += height_time
    
    
# print(all_values)
# print(value_names)
    
    
# value_index = np.arange(0,len(all_values) )
# colour = value_names.index(value_index)
# ax[2].scatter(value_index, np.ones(len(all_values)), label=all_values, c=f"C{colour}")
        
ax[0].set_xlabel("Worker")
ax[0].set_ylabel("Quantity of input data")
ax[1].set_xlabel("Worker")
ax[1].set_ylabel("Time to process input data / s")
ax[2].set_xlabel("Worker")
ax[2].set_ylabel("Time to process input data / s")
# ax[1].set_yscale("log")
ax[0].legend(labels=all_values)
# ax[1].legend(labels=value_names)
plt.savefig("/output_container/workerfig.png")




print("\nfinal analysis")
final_anal(all_data, samples)
 
print("finished!")




