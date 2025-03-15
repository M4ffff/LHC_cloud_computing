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
channel.queue_declare(queue='counter')





import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import infofile 
import pickle as pkl


#%%

def counter_work(ch, method, properties, counter_inputs):
    # inputs_dict = json.loads(inputs.decode('utf-8'))
    counter_inputs_dict = pkl.loads(counter_inputs)
    
    samples = counter_inputs_dict["samples"]
    path = counter_inputs_dict["path"]
    run_time_speed = counter_inputs_dict["run_time_speed"]


    # Print which sample is being processed
    print(f'ENTRY_COUNTER received') 
    
    num_entries = {}
    for sample in samples: 
        # Print which sample is being processed
        print(f'{sample} samples') 

        # Define empty list to hold data
        total_num_entries = 0 
        sample_num_entries = []
        
        
        # Loop over each file
        for value in samples[sample]['list']: 
            if sample == 'data': 
                prefix = "Data/" 
            else: 
                prefix = f"MC/mc_{str(infofile.infos[value]['DSID'])}."
            fileString = f"{path}{prefix}{value}.4lep.root" 

            print(f"\t{value}:") 

            # Open file
            tree = uproot.open(f"{fileString}:mini")
            
            value_entries = int(tree.num_entries*run_time_speed)
            total_num_entries += value_entries
            sample_num_entries.append(value_entries)
            print(f"TREE ENTRIES: {total_num_entries}")

        
        num_entries[sample] = sample_num_entries
        
    entries_dict = {"num_entries":num_entries, "total_num_entries":total_num_entries}
    entry_outputs = pkl.dumps(entries_dict)
        
        
    print(f"ENTRY COUNTER publishing results")
    channel.basic_publish(exchange='',
                        routing_key='counter',
                        body=entry_outputs)
    print(f"ENTRY COUNTER published results")
    
    ch.stop_consuming()

    print(f"ENTRY COUNTER stopped for good")





# setup to listen for messages on queue 'messages'
channel.basic_consume(queue='counter',
                      auto_ack=True,
                      on_message_callback=counter_work)



# log message to show we've started
print('Waiting to count')

# start listening
channel.start_consuming()

# log message to show we've started
print('Counting complete')

# WORKERS WILL ALSO NEED TO SEND DATA BACK


