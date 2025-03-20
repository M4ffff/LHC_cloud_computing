#!/usr/bin/env python
import pika
import vector 
import infofile 
import pickle as pkl
import time
import socket 



# start service
params = pika.ConnectionParameters('rabbitmq', heartbeat=0)

# create connection to network
connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.basic_qos(prefetch_count=1)

# create the queues to send daya to and from master
channel.queue_declare(queue='master_to_worker')
channel.queue_declare(queue='worker_to_master')


# get id name of each worker
worker_id = socket.gethostname()
print(f"SOCKET NAME: {worker_id}")

# set units and luminosity
MeV = 0.001
GeV = 1.0

luminosity=10


def publish(dict, routing_key):
    """
    Publish information from this container to another

    Args:
        dict (dict): dictionary containing information to be sent. 
        routing_key (str): queue for message to be sent to
    """
    
    # convert dictionary to pickle file
    outputs = pkl.dumps(dict)

    # send to master
    channel.basic_publish(exchange='',
                        routing_key=routing_key,
                        properties=pika.BasicProperties( app_id=worker_id ),  # send which worker this message is from 
                        body=outputs)

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
    
    Args:
        lepton_pt (awk array): lepton transverse momentum
        lepton_eta (awk array): angle of particle relative to beam
        lepton_phi (awk array): angle of particles movement
        lepton_E (awk array): lepton energies

    Returns:
        float: invariant mass
    """
    # groups into four vectors
    p4 = vector.zip({"pt": lepton_pt, "eta": lepton_eta, "phi": lepton_phi, "E": lepton_E})
    # Adds mass of 4 vectors and converts to MeV
    invariant_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV 
    return invariant_mass


def calc_weight(relevant_weights, sample, events):
    """
    Caluclate weighting of events

    Args:
        relevant_weights (list): list of relevant weights
        sample (str): name of subsample
        events (awk arr): data descirbing events

    Returns:
        float: MC weighting
    """
    info = infofile.infos[sample]
    # determine all weightings of MC evennts
    mc_weights = (luminosity*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) 
    total_weight = mc_weights 
    
    # multiply events by their weightings
    for variable in relevant_weights:
        total_weight = total_weight * events[variable]
    return total_weight






#%%

worker_log = {}
# tot_data_analysed = 0
elapsed_list = []
value_name_list = []
nin_list = []

def worker_work(ch, method, properties, inputs):
    # global tot_data_analysed
    # global worker_log
    
    # load inputs
    inputs_dict = pkl.loads(inputs)
    
    tree = inputs_dict["tree"]
    variables = inputs_dict["variables"]
    relevant_weights = inputs_dict["relevant_weights"]
    entry_start = inputs_dict["entry_start"]
    entry_stop = inputs_dict["entry_stop"]
    sample = inputs_dict["sample"]
    value = inputs_dict["value"]
    start_time = inputs_dict["start"]

    # start timing of analysis
    start = time.time()
    # time.sleep(1)
    
    # Print which sample is being processed
    print(f'WORKER received {sample} {value}, chunk {entry_start} to {entry_stop}') 
 

    local_sample_data = []

    # Loop over data in the tree
    for data in tree.iterate(variables + relevant_weights, library="ak", entry_start=entry_start,
                            entry_stop=entry_stop, step_size = 1_000_000):
        
        # Number of events in this batch
        nIn = len(data) 
        # record how much data this worker has analysed
        # tot_data_analysed += nIn
        
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
         # time taken to process
        # print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after

        local_sample_data.append(data)
        
    elapsed = time.time() - start + 0.1
    print(f"WORKER analysed {value}, chunk {entry_start} to {entry_stop}\n\t\t\t\t in {elapsed} seconds")
    
    # if value in worker_log.keys():
    #     worker_log[value]['len'] += nIn
    #     worker_log[value]['time'] += elapsed
    # else:
    #     worker_log[value] = {}
    #     worker_log[value]['len'] = nIn
    #     worker_log[value]['time'] = elapsed

    elapsed_list.append(elapsed)
    value_name_list.append(value)
    nin_list.append(nIn)
    worker_log["elapsed list"] = elapsed_list
    worker_log["value name list"] = value_name_list
    worker_log["nin list"] = nin_list

    outputs_dict = {"sample":sample, "data":local_sample_data, "entry_stop":entry_stop, "entry_start":entry_start,
                    "value":value, "worker_log":worker_log, "start":start_time}
    
    publish(outputs_dict, "worker_to_master")

    




# setup to listen for messages on queue 'master_to_worker'
channel.basic_consume(queue='master_to_worker',
                      auto_ack=True,
                      on_message_callback=worker_work)



# log message to show we've started
print('Waiting for messages. To exit press CTRL+C')



# start listening
channel.start_consuming()

# log message to show we've started
print('Waiting to re-run. To exit press CTRL+C')

# WORKERS WILL ALSO NEED TO SEND DATA BACK


