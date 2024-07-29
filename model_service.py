# 1. Pickup Model : If config file exist then load the training model and If not exist then load train model
# 2. Make the Prediction

from pathlib import Path
import pickle as pk
from model import build_model



class ModelService:
    
    def __init__(self):
        self.model = None
        
    def load_model(self, model_name='rf_v1'):
        model_path = Path(f'models/{model_name}')

        if not model_path.exists():
            build_model()

        self.model = pk.load(open(f'models/{model_name}', 'rb'))

    def predict(self, input_parameters):
        return self.model.predict([input_parameters])
    
    
    # testing
#ml_service = ModelService()
    
#ml_service.load_model('rf_v1')

#input_parameters = [85,2015,2,20,1,1,0,0,1]
#pred = ml_service.predict(input_parameters)    
#print(f'The Predicted Rent Based On Input Parameter is {pred} Dollars.')