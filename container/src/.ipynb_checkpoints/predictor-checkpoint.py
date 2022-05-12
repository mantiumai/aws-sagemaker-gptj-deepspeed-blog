# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.
import json
import os
from pathlib import Path
import flask
import logging
import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer
)
import deepspeed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Inference Endpoint')

# Path where SageMaker mounts the model
root_model_path = "/opt/ml/"

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


# Sometimes tarballs get packed incorrectly, e.g. with model files nested in a series of subdirectories.
# This function helps ensure that the full path to the model directory.

def find_config(rootdir):
    """
    Find directory that contains Transformers model files.
    """
    for path in Path(rootdir).rglob('config.json'):
        return str(path.parents[0])

def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


class ScoringService(object):
    pipeline = None  # Where we keep the model when it's loaded

    @classmethod
    def get_pipeline(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.pipeline is None:
            # Determine device -- needs to be updated for multi gpu

            model_path = find_config(root_model_path)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            pipe_device = -1 if device == "cpu" else 0

            logger.info('Loading model without init...')
            model = no_init(lambda: AutoModelForCausalLM.from_pretrained(
                                        model_path,
                                        revision='float16',
                                        torch_dtype=torch.float16,
                                        low_cpu_mem_usage=True))


            tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info('Initializing DeepSpeed Inference...')
            model = deepspeed.init_inference(model,
                                             mp_size=1,
                                             dtype=model.dtype,
                                             replace_method='auto',
                                             replace_with_kernel_inject=True)

            cls.pipeline = pipeline(task='text-generation', model=model, tokenizer=tokenizer, device=pipe_device)

            return cls.pipeline

        return cls.pipeline

    @classmethod
    def predict(cls, inputs):
        """For the input, do the predictions and return them.

        Args:
            inputs: dict
                A dictionary that contains the input text and hyperparameters that should be used for
                conditional generation. Input text is expected to be associated with a "text_inputs" key and
                hyperparameters are expected to be stored in a dictionary associated with a "parameters" key.

        Examples:

            inputs = {
                "text_inputs" = "This is an exciting blog post about Natural Language Processing and AI.,
                "parameters": {
                    "temperature": .8,
                    "min_tokens": 200,
                    "max_new_tokens": 500,
                }
            }

            predictions = predict(inputs)
        """

        pipe = cls.get_pipeline()

        parameters = inputs.pop('parameters')

        return pipe(**inputs, **parameters)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_pipeline() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single input.
    """

    # data should be a dictionary with an 'inputs' key
    data = flask.request.get_json()
    if isinstance(data, str):
        data = json.loads(data)

    inputs = data['inputs']

    predictions = ScoringService.predict(inputs)

    return flask.jsonify(response=predictions, status=200)
