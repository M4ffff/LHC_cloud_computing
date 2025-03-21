# Header

Project to analyse data from LHC, using Docker for cloud computing.

Analysis based on HZZAnalysis Jupyter notebook (url).

<!-- to build images -->
docker image build -t master_image master/.
docker image build -t worker_image worker/.

<!-- to run on network -->
docker run --rm -d -p 15672:15672 -p 5672:5672 --network rabbit --name rabbitmq rabbitmq:3-management
docker run --rm -it --network rabbit --name master_container --mount src="$(pwd)",target=/output_container,type=bind master_image
docker run --rm -it --network rabbit --name worker_container worker_image

<!-- to run as docker compose -->
<!-- need to make sure the params is set correctly depending on if its docker compose or network -->
<!-- number of workers set as 3 -->
<!-- stop running once master container finishes -->
docker compose up --scale consumer=3 --abort-on-container-exit

<!-- runs in detached mode - doesnt clog up terminal: -->
docker compose up -d --scale consumer=3
