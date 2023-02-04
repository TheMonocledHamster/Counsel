!/bin/sh

docker cp counsel:/Counsel/data .
docker cp counsel:/Counsel/logs .
docker cp counsel:/Counsel/infer_logs .
docker cp counsel:/Counsel/inf_logs .

mkdir -p charts
