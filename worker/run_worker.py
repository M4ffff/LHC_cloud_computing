#!/usr/bin/env python
import pika
 
# when RabbitMQ is running on localhost
# params = pika.ConnectionParameters('localhost')

# when RabbitMQ broker is running on network
params = pika.ConnectionParameters('rabbitmq')

# when starting services with docker compose
# params = pika.ConnectionParameters(
#    'rabbitmq',
#    heartbeat=0)

# create the connection to broker
connection = pika.BlockingConnection(params)
channel = connection.channel()

# create the queue, if it doesn't already exist
channel.queue_declare(queue='messages')
channel.queue_declare(queue='sending_all')





import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import vector # for 4-momentum calculations
import time
import infofile 
import json
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator 
import pickle as pkl

MeV = 0.001
GeV = 1.0

lumi=10



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


#%%

def worker_work(ch, method, properties, inputs):
    # inputs_dict = json.loads(inputs.decode('utf-8'))
    inputs_dict = pkl.loads(inputs)
    
    samples_sample = inputs_dict["samples_sample"]
    sample = inputs_dict["sample"]
    path = inputs_dict["path"]
    variables = inputs_dict["variables"]
    relevant_weights = inputs_dict["relevant_weights"]
    run_time_speed = inputs_dict["run_time_speed"]


    # Print which sample is being processed
    print(f'WORKER received {sample}') 

    # Define empty list to hold data
    frames = [] 

    # Loop over each file
    for value in samples_sample['list']: 
        if sample == 'data': 
            prefix = "Data/" 
        else: 
            prefix = f"MC/mc_{str(infofile.infos[value]['DSID'])}."
        fileString = f"{path}{prefix}{value}.4lep.root" 


        print(f"\t{value}:") 
        start = time.time() 

        # Open file
        tree = uproot.open(f"{fileString}:mini")
        print(f"TREE ENTRIES: {int(tree.num_entries*run_time_speed)}")
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

            # Append data to the whole sample data list
            sample_data.append(data)

        frames.append(ak.concatenate(sample_data)) 

        outputs_dict = {"sample":sample, "value":value, "sample_data":sample_data}
        outputs = pkl.dumps(outputs_dict)
        
        print(f"WORKER publishing {sample}, {value}")
            # MASTER WILL NEED TO SEND MESSAGES TO WORKERS
        # send a simple message
        channel.basic_publish(exchange='',
                            routing_key='sending_all',
                            body=outputs)
        print(f"WORKER published {sample}, {value}")






# setup to listen for messages on queue 'messages'
channel.basic_consume(queue='messages',
                      auto_ack=True,
                      on_message_callback=worker_work)



# log message to show we've started
print('Waiting for messages. To exit press CTRL+C')

# start listening
channel.start_consuming()

# log message to show we've started
print('Waiting to re-run. To exit press CTRL+C')

# WORKERS WILL ALSO NEED TO SEND DATA BACK


