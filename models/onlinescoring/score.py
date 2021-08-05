import os
import json

import dill
        

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    """
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "ms-distance-calculator.pkl"
    )

    # Open the pickle back into a model
    with open(model_path, "rb") as f:
        model = dill.load(f)


def run(request):

    print(request)
    text = json.loads(request)
    q = str(text["query"])
    # Run the model
    return model.get_materials(q.strip('"'))