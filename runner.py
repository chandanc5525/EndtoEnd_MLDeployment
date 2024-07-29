from model_service import ModelService


def main():
    ml_service = ModelService()
    ml_service.load_model('rf_v1')
    pred = ml_service.predict([85, 2015, 2, 20, 1, 1, 0, 0, 1])
    print(f'The predicted value is ',pred)

if __name__ == '__main__':
    main()