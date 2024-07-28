from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
from preparation import prepare_data
import os

def build_model():
    
    # 1. Load the Dataset
    df = prepare_data()
    # 2. Identify X and y values
    X,y = get_data(df)
    # 3. Split the Dataset into X_train,X_test and y_train,y_test
    X_train,X_test,y_train,y_test = split_train_test(X, y)
    # 4. Evaluate the model performance
    rf = train_model(X_train, y_train)
    # 5. Model Score  
    score = evaluate_model(rf, X_test, y_test)
    print(f'Models score = {score}')
    # 5. Save the model as pickle file
    save_model(rf)
              
    
def get_data(data,
            col_X = ['area', 
                  'constraction_year', 
                  'bedrooms', 
                  'garden', 
                  'balcony_yes', 
                  'parking_yes', 
                  'furnished_yes', 
                  'garage_yes', 
                  'storage_yes'],
            col_y = 'rent'):
    
    return data[col_X],data[col_y] 
    
def split_train_test(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21)
    
    return X_train,X_test,y_train,y_test


def train_model(X_train,y_train):
    
    grid_space = {'n_estimators': [100, 200, 300], 
                  'max_depth': [3, 6, 9, 12]}
    
    grid = GridSearchCV(RandomForestRegressor(), 
                        param_grid=grid_space, 
                        cv=5, 
                        scoring = 'r2')
    
    model_grid = grid.fit(X_train, y_train)
    
    return model_grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)

def save_model(model):
    os.makedirs('models', exist_ok=True)
    pk.dump(model, open('models/rf_v1', 'wb'))
    
 # test build_model()   
# build_model()
    
    
    

