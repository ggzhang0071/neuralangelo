img="docker.io/chenhsuanlin/colmap:3.8"
# cuda 11.3  torch 1.12.1


docker run --gpus all  --privileged=true   --workdir /git --name "neuralangelo"  -e DISPLAY --ipc=host -d --rm  -p 3832:4452  \
-v /localhome/local-vili/git/neuralangelo:/git/neuralangelo \
 -v /localhome/local-vili/git/datasets:/git/datasets \
 $img sleep infinity

docker exec -it neuralangelo  /bin/bash
# sudo docker images  |grep "pytorch"  |grep "22."

# sudo docker stop  neuralangelo 

