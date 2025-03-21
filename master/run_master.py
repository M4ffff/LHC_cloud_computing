#!/usr/bin/env python
import pika
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator 
import awkward as ak 
import pickle as pkl
import infofile
import time
import uproot





def final_anal(all_data, samples):
    """
    analyse final data and produce final plots

    Args:
        all_data (dict): contains all data to be analysed
        samples (dict): names of all samples and subsamples
    """
    # x-axis range of the plot
    xmin = 80 * GeV
    xmax = 250 * GeV

    # Histogram bin setup
    step_size = 5 * GeV

    bin_edges = np.arange(xmin, xmax+step_size, step_size ) 
    bin_centres = np.arange(xmin+step_size/2, xmax+step_size/2, step_size ) 

    # monte carlo entries
    mc = [] 
    # weights of each entry
    mc_weights = [] 
    # colour and label of each MC bar
    mc_colours = [] 
    mc_labels = [] 

    # loop over background samples
    for sample in [r'Background $Z,t\bar{t}$', r'Background $ZZ^*$']: 
        mc.append( ak.to_numpy(all_data[sample]['mass']) ) 
        mc_weights.append( ak.to_numpy(all_data[sample].totalWeight) ) 
        mc_colours.append( samples[sample]['color'] ) 
        mc_labels.append( sample ) 


    print("start plotting")

    fig, ax = plt.subplots(figsize=(15,5)) 

    # plot data
    data = np.histogram(ak.to_numpy(all_data['data']['mass']), bin_edges )[0] 
    data_errors = np.sqrt( data ) 
    ax.errorbar(x=bin_centres, y=data, yerr=data_errors, fmt='ko', label='Data') 


    
    # plot the background data
    mc_heights = ax.hist(mc, bins=bin_edges, weights=mc_weights, stacked=True, 
                                color=mc_colours, label=mc_labels )
    
    # total height of background data samples combined
    mc_tot = mc_heights[0][-1]

    # plot signal
    signal = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)']['mass'])
    # weights of each signal events
    signal_weights = ak.to_numpy(all_data[r'Signal ($m_H$ = 125 GeV)'].totalWeight)
    # colour of signal bar
    signal_color = samples[r'Signal ($m_H$ = 125 GeV)']['color']
    # plot signal bars
    signal_heights = ax.hist(signal, bins=bin_edges, bottom=mc_tot, 
                    weights=signal_weights, color=signal_color,
                    label=r'Signal ($m_H$ = 125 GeV)')
    
    # calculate uncertainty in background data
    mc_err = np.sqrt(np.histogram(np.hstack(mc), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])

    # plot statistical uncertainty
    ax.bar(bin_centres, 2*mc_err, alpha=0.5, bottom=mc_tot-mc_err, color='none', 
                    hatch="////", width=step_size, label='Stat. Unc.' )


    # SET AXES SETTINS AS IN HZZAnalysis notebook
    
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
    
    ax.legend( frameon=False ) 
    
    # Add text 'ATLAS Open Data' on plot
    plt.text(0.05, 0.93, 'ATLAS Open Data', transform=ax.transAxes, fontsize=13 ) 
    # Add energy and luminosity
    luminosity_used = str(luminosity*data_fraction) # luminosity to write on the plot
    plt.text(0.05, 0.82, r'$\sqrt{s}$=13 TeV,$\int$L dt = '+luminosity_used+ r' fb$^{-1}$', transform=ax.transAxes ) 
    # Add a label for the analysis carried out
    plt.text(0.05, 0.76, r'$H \rightarrow ZZ^* \rightarrow 4\ell$', transform=ax.transAxes ) 

    # save figure to container
    output_path = "/output_container/HZZ_analysis_figure.png"
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path)

    # Signal height
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
    

def publish(dict, routing_key, publish_counter):
    """
    Publish information from this container to another

    Args:
        dict (dict): dictionary containing information to be sent. 
        routing_key (str): queue for message to be sent to
        publish_counter (int): count of how many messages have been published
    """
    publish_counter += 1
    
    outputs = pkl.dumps(dict)
    channel.basic_publish(exchange='',
                        routing_key=routing_key,
                        body=outputs)
    return publish_counter



### RECEIVE DATA
print("connection starting")


# when running on network
# params = pika.ConnectionParameters('rabbitmq')

# when starting services with docker compose
params = pika.ConnectionParameters('rabbitmq', heartbeat=0)

print("connection started")

# create the connection to broker
connection = pika.BlockingConnection(params)

channel = connection.channel()

# create the queue, if it doesn't already exist
channel.queue_declare(queue='master_to_worker')
channel.queue_declare(queue='worker_to_master')


# set units and luminosity
MeV = 0.001
GeV = 1.0

luminosity=10

# path to data
path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" 

### the longest/biggest datasets are calculated first
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

# Important variables used in analysis
variables = ['lep_pt','lep_eta','lep_phi','lep_E','lep_charge','lep_type']
relevant_weights = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]

# Controls the fraction of all events analysed
# change lower to run quicker
data_fraction = 1

    
# Dictionary to hold awkward arrays
all_data = {} 
counter = 0
# logs of outputs from each worker
worker_logs = {}
all_values = []

def receive_data(ch, method, properties, outputs):
    """
    Function called when a message is received by a worker

    Args:
        ch (): channel over which message is sent  
        method (): infomration about message delivery
        properties (): defined properties
        outputs (pkl): pickled dictionary of outputs of worker
    """
    # track how many messages have been received
    global counter
    global publish_counter
    counter += 1
    
    # determine which worker the message is from 
    worker_id = properties.app_id
    
    # get output data
    outputs_dict = pkl.loads(outputs)
    
    sample = outputs_dict["sample"]
    value = outputs_dict["value"]
    # THIS IS A LIST OF LIST OF AWKWARD ARRAYs
    data = outputs_dict["data"]
    entry_stop = outputs_dict["entry_stop"]
    entry_start = outputs_dict["entry_start"]
    worker_log = outputs_dict["worker_log"]
    start_time = outputs_dict["start"]
    
    # track the values that come up in the order they're received
    if value not in all_values:
        all_values.append(value)
    
    # add log of specific information from this worker 
    worker_logs[worker_id] = worker_log
    
    print(f"received {sample} {value} data, chunk: {entry_start} to {entry_stop}")
    
    # add sample to all data
    if sample in all_data.keys():
        current_sample = all_data[sample]
        updated_version = ak.concatenate( [current_sample, data[0]] )
        all_data[sample] = ( updated_version )
    else:
        all_data[sample] = (data[0]) 
            
    # record total time taken to process this chunk of data
    end = time.time()
    elapsed = end - start_time
    print(f"{sample} {value} done in {elapsed} second ")
    
    print(f"received {counter} of {publish_counter} messages ")
    if counter >= publish_counter:
        ch.stop_consuming()
        print("STOPPED CONSUMING")  
        
    print(f"{sample} timing: {elapsed:.2f} s")
    

# one second wait to ensure containers have activated
time.sleep(1)
print("pause done")

# Determine number of available workers and how long it takes to check
start = time.time()
num_workers = channel.queue_declare(queue='master_to_worker', passive=True).method.consumer_count
end = time.time()
print(f"time to check number of workers: {end - start:.5f} seconds")

# num_workers = 3
print("FETCHED NUMBER OF WORKERS: ", num_workers) 

# setup to listen for messages on queue 'master_to_worker'
channel.basic_consume(queue='worker_to_master',
                    auto_ack=True,
                    on_message_callback=receive_data)

# initialise
all_num_entries = {}

# track number of publishes done
publish_counter = 0

for i,sample in enumerate(samples):     
    # update number of workers in case any activated after initial count. 
    num_workers = channel.queue_declare(queue='master_to_worker', passive=True).method.consumer_count

    # Define empty list to hold data
    frames = [] 

    # Loop over each subsample
    for value in samples[sample]['list']: 
        if sample == 'data': 
            prefix = "Data/" 
        else: 
            prefix = f"MC/mc_{str(infofile.infos[value]['DSID'])}."
        fileString = f"{path}{prefix}{value}.4lep.root" 

        # Open file
        tree = uproot.open(f"{fileString}:mini")
        
        # calculate number of entries in this sample value
        num_entries = tree.num_entries*data_fraction
        
        # add to dictionary
        all_num_entries[sample] = num_entries
        
        sample_data = []

        # calculate amount of data to send to each worker
        # +1 ensures chunk_size*num_workers >= num_entries
        chunk_size = round(num_entries / (num_workers))+1
        
        # ensures tiny bits of data aren't sent to multiple workers - send data of at least size min_chunk_size to each worker.  
        # also ensures huge chunks of data aren't sent to each worker - send data of less than size max_chunk_size to each worker.  
        # there may be better choices than 10_000 and 100_000_000 but it is hard to find out
        min_chunk_size = 10000
        max_chunk_size = 100_000_000
        if chunk_size < min_chunk_size:
            chunk_size=min_chunk_size
        elif chunk_size > max_chunk_size:
            chunk_size=max_chunk_size


        # split into chunks depending on numbers of workers, so each worker gets roughly same amount of data to analyse
        # chunk_size+1 ensures there is not more chunks than workers. 
        data_chunks = np.arange(0, num_entries, chunk_size)
        print(f"\n{sample} {value} being published")
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
            
            # publish information and update publish_counter
            publish_counter = publish(inputs_dict, "master_to_worker", publish_counter)

print("MASTER consuming")

# start listening
channel.start_consuming()



# only plot as a graph if there are a small number of workers
if num_workers < 5:
    fig,ax = plt.subplots(1,2, figsize=(16,6))

    for i, current_worker in enumerate(worker_logs.keys()):
        
        log = worker_logs[current_worker]
        bottom_len = 0
        bottom_time = 0
        # for sample in samples:
        for j, value in enumerate(log["value name list"]):
            colour = all_values.index(value)
            
            height_len = log['nin list'][j]
            height_time = log['elapsed list'][j]
            # print(f"{bottom_len} to {height_len+bottom_len}")
            ax[0].bar(i, height_len, bottom=bottom_len, color=f"C{colour}")
            ax[1].bar(i, height_time, bottom=bottom_time, color=f"C{colour}")
            # update bottom of bar
            bottom_len += height_len
            bottom_time += height_time
                    
    ax[0].set_xlabel("Worker")
    ax[0].set_xticks(np.arange(num_workers))
    ax[0].set_xticklabels(worker_logs.keys())
    ax[0].set_ylabel("Quantity of input data")
    ax[1].set_xlabel("Worker")
    ax[1].set_xticks(np.arange(num_workers))
    ax[1].set_xticklabels(worker_logs.keys())
    ax[1].set_ylabel("Time to process input data / s")

    plt.savefig("/output_container/worker_figure.png")


print("\nfinal analysis")
final_anal(all_data, samples)
 
print("finished!")




