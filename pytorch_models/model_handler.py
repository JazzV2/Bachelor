from ts.torch_handler.base_handler import BaseHandler
import cv2 as cv
import numpy as np
import importlib.util
import inspect
import torch
import json
import os
import base64
from ts.metrics.metric_type_enum import MetricTypes

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.model_pt_path = None
        self.manifest = None

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        serialized_file = self.manifest["model"]["serializedFile"]
        self.model_pt_path = os.path.join(model_dir, serialized_file)
        model_file = self.manifest["model"].get("modelFile", "")

        model_class = self.getModel(model_file)
        self.model = model_class()
        self.model.to(self.device)
        state_dict = torch.load(self.model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        
        self.initialized = True

    def preprocess(self, data):
        metrics = self.context.metrics
        
        request_batch_size_metric = metrics.get_metric(
            metric_name="RequestBatchSizeSum", metric_type=MetricTypes.COUNTER
        )

        request_batch_count_metric = metrics.get_metric(
            metric_name="RequestBatchSizeCount", metric_type=MetricTypes.COUNTER
        )

        request_batch_size_metric.add_or_update(
            value=len(data), dimension_values=[self.context.model_name]
        )

        request_batch_count_metric.add_or_update(
            value=1, dimension_values=[self.context.model_name]
        )
        
        images = []
        for row in data:
            preprocessed_data = row.get("data")
            if preprocessed_data is None:
                preprocessed_data = row.get("body")
    
            base_64_data = base64.urlsafe_b64decode(preprocessed_data["data"][2:len(preprocessed_data["data"])-1])
            np_preprocessed_data = np.frombuffer(base_64_data, dtype=np.uint8)
            image = cv.imdecode(np_preprocessed_data, flags=cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.resize(image, (100, 100))
            image = torch.tensor(image, dtype=torch.float32)
            image = image / 255.0
            image = image.unsqueeze(0)
            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, model_input):
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output):
        postprocess_output = self.translateOutput(inference_output)
        return postprocess_output

    def handle(self, data, context):
        self.context = context

        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
    
    def getModel(self, model_file):
        module = importlib.import_module(model_file.split(".")[0])
        model = None
        for attrName in dir(module):
                attribute = getattr(module, attrName)
                if inspect.isclass(attribute):
                    model = attribute
                    return model
                
    def translateOutput(self, output):
        emotions = {'sad': 0, 'scared': 1, 'neutral': 2, 'happy': 3, 'angry': 4}
        answers = []
        exp_output = torch.exp(output)
        for pred in exp_output:
            answer = {}
            for i in emotions:
                answer[i] = pred[emotions[i]].item()
            answers.append(json.dumps(answer))

        return answers
                