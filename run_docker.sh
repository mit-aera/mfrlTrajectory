#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run \
    -it \
    --publish-all \
    --rm \
    --volume "${DIR}:/root/mfrl" \
    --name mfrl \
    --privileged \
    --gpus all \
    --net "host" \
    mfrl
