**Header**
Project to analyse data from LHC, using Docker for cloud computing

docker image build -t master_image master/.
docker image build -t worker_image worker/.

docker run --rm -d -p 15672:15672 -p 5672:5672 --network rabbit --name rabbitmq rabbitmq:3-management

docker run --rm -it --network rabbit --name master_container master_image
docker run --rm -it --network rabbit --name worker_container worker_image
