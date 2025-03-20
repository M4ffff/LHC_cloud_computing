#!/usr/bin/env python
import pika
 
# when RabbitMQ is running on localhost
# params = pika.ConnectionParameters('localhost')

# when RabbitMQ broker is running on network
# params = pika.ConnectionParameters('rabbitmq')

# when starting services with docker compose
params = pika.ConnectionParameters(
   'rabbitmq',
   heartbeat=0)

# create the connection to broker
connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.basic_qos(prefetch_count=1)

# create the queue, if it doesn't already exist
channel.queue_declare(queue='master_to_worker')
channel.queue_declare(queue='worker_to_master')





import vector # for 4-momentum calculations
import infofile 
import pickle as pkl
import time
import socket 

worker_id = socket.gethostname()
print(f"SOCKET NAME: {worker_id}")

MeV = 0.001
GeV = 1.0

lumi=10


def publish(dict, routing_key):
    
    outputs = pkl.dumps(dict)
    
    # print(f"{container} publishing {sample}")

    channel.basic_publish(exchange='',
                        routing_key=routing_key,
                        properties=pika.BasicProperties( app_id=worker_id ),
                        body=outputs)
    # print(f"{container} published {sample}")

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

worker_log = {}
tot_data_analysed = 0
elapsed_list = []
value_name_list = []
nin_list = []

def worker_work(ch, method, properties, inputs):
    global tot_data_analysed
    global worker_log
    
    inputs_dict = pkl.loads(inputs)
    
    tree = inputs_dict["tree"]
    variables = inputs_dict["variables"]
    relevant_weights = inputs_dict["relevant_weights"]
    entry_start = inputs_dict["entry_start"]
    entry_stop = inputs_dict["entry_stop"]
    sample = inputs_dict["sample"]
    value = inputs_dict["value"]
    start_time = inputs_dict["start"]


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
        tot_data_analysed += nIn
        # print(f"WORKER total amount of data analysed by this worker: {tot_data_analysed}")
        
        
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


