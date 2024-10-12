from ts.torch_handler.base_handler import BaseHandler
import cv2 as cv
import numpy as np
import importlib.util
import inspect

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
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        serialized_file = self.manifest["model"]["serializedFile"]
        self.model_pt_path = os.path.join(model_dir, serialized_file)
        model_file = self.manifest["model"].get("modelFile", "")

        model_class = self.getModel(model_file)
        self.model = model_class()
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        
        image = cv.imread(preprocessed_data)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        resize_image = cv.resize(rgb_image, (100, 100))
        std_image = resize_image / 255
        std_image = np.moveaxis(std_image, 2, 0).astype(dtype='float32')

        return preprocessed_data


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
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
                