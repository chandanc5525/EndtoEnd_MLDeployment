from model_service import ModelService
from loguru import logger # loguru is for logging purposes

logger.add('app_log',rotation= '1 MB')
logger.info('Running ModelService')
def main():
    ml_service = ModelService()
    ml_service.load_model('rf_v1')
    logger.warning('Please Ensure Indepedent Variable Data for Prediction Model Service')
    pred = ml_service.predict([85, 2015, 2, 20, 1, 1, 0, 0, 1])
    print(f'The predicted value is ',pred)

try:
    if __name__ == '__main__':
        main()
except:
    logger.error('Error, Please Debug Your Code')