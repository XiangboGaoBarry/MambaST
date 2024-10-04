if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

#in gerneral do only read mode but since cache created need to write 
docker run -it \
  -u $(id -u):$(id -g) \
  --gpus '"device='$GPU'"' \
  --hostname $(hostname) \
  -e HOME \
  -v /mnt/workspace/datasets:/mnt/workspace/datasets \
  -w /home/your_username \
  --ipc "host"\
  -v $(pwd):/home/your_username \
   mambast:latest  \

#   -v path_to_project_directory/runs:/home/akanu/runs \
#    -p 6006:6006 \
#   -u $(id -u):$(id -g) \
#   -v $(pwd)/.bash_history:$HOME/.bash_history \
