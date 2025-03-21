# Big Data Analysis using Cloud Technology

Project to analyse data from LHC, using Docker for cloud computing.

Analysis based on HZZAnalysis Jupyter notebook ([url](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/blob/master/13-TeV-examples/uproot_python/HZZAnalysis.ipynb)).

to build images:

```
docker image build -t master_image master/.
docker image build -t worker_image worker/.
```

to run on network (not recommended):

```
docker run --rm -d -p 15672:15672 -p 5672:5672 --network rabbit --name rabbitmq rabbitmq:3-management
docker run --rm -it --network rabbit --name master_container --mount src="$(pwd)",target=/output_container,type=bind master_image
docker run --rm -it --network rabbit --name worker_container worker_image
```

(If running on network, need to make sure the params is set correctly by uncommenting the line indicated in the run_master and run_worker scripts) 

to run as docker compose (recommended):
 
```
docker compose up --scale consumer=3 --abort-on-container-exit
```

(number of workers currently set as 3)  

(stops running once a container finishes - should be when the master container finishes) 
