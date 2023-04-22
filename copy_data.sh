!/bin/sh

docker cp counsel:/counsel/data .
docker cp counsel:/counsel/logs .
docker cp counsel:/counsel/infer_logs .
docker cp counsel:/counsel/inf_logs .

mkdir -p charts
