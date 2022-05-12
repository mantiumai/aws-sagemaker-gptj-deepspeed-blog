#!/usr/bin/env bash

IMAGE_NAME=$1

export IMAGE_NAME=$IMAGE_NAME

### Build container--------
cd container
chmod +x src/serve
docker build -t $IMAGE_NAME .


IMAGE_ID="$(docker inspect --format="{{.Id}}" $IMAGE_NAME)"

### Download model--------
cd ..
# Install huggingface_hub
pip install -r run_local/requirements.txt

# Check if model directory or files exist, if they don't download the model
DIR="./run_local/test_dir/"
# init
# look for empty dira
if [ -d "$DIR" ]
then
    if [ "$(ls -A $DIR)" ]; then
        echo "Existing model directory detected..."
    fi
else
    echo "Downloading model..."
    mkdir ./run_local/test_dir
    python run_local/download_hf_model.py EleutherAI/gpt-j-6B --revision float16 --allow_regex *.json *.txt *.bin
fi

if [ $2 = "test_local" ] ; then
    docker run --name $IMAGE_NAME --gpus all -v $(pwd)/run_local/test_dir:/opt/ml -p 8080:8080 --rm ${IMAGE_ID} serve
fi





