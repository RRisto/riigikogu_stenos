APP_PATH=$PWD

docker stop riigikogu_run
docker rm riigikogu_run
docker run -it  --rm \
   -p 8888:8888 \
    -v $APP_PATH:/home/jovyan/work \
    --name riigikogu_run \
    riigikogu