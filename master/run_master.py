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

### RECEIVE DATA


# create the connection to broker
connection = pika.BlockingConnection(params)

channel = connection.channel()

# create the queue, if it doesn't already exist
channel.queue_declare(queue='messages')




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

inputs_dict = {"samples":samples, "path":path, "variables":variables, "relevant_weights":relevant_weights, "run_time_speed":run_time_speed}
inputs = json.dumps(inputs_dict).encode('utf-8')

# MASTER WILL NEED TO SEND MESSAGES TO WORKERS
# send a simple message
channel.basic_publish(exchange='',
                      routing_key='messages',
                      body=inputs)




# setup to listen for messages on queue 'messages'
# channel.basic_consume(queue='sending_all',
#                       auto_ack=True,
#                       on_message_callback=receive_data)

# start listening
# channel.start_consuming()







