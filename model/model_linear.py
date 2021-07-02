from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge, LogisticRegression

from abc import abstractmethod
class Base_Model(object):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError

class LinearRidge(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def train(self, CFG, x_train, y_train):
        model =Ridge(self.model_params)
        model.fit(x_train,y_train)
        return model

    def valid(self, CFG, x_valid, model):
        preds = model.predict(x_valid)
        return preds

    def inference(self, CFG, x_test, model):
        preds = model.predict(x_test)
        return preds
        

class LinearLasso(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def train(self, CFG, x_train, y_train):
        model =Lasso(self.model_params)
        model.fit(x_train,y_train)
        return model

    def valid(self, CFG, x_valid, model):
        preds = model.predict(x_valid)
        return preds

    def inference(self, CFG, x_test, model):
        preds = model.predict(x_test)
        return preds
    
class BayesReg(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def train(self, CFG, x_train, y_train):
        model =BayesianRidge(self.model_params, compute_score=True)
        model.fit(x_train,y_train)
        return model

    def valid(self, CFG, x_valid, model):
        preds = model.predict(x_valid)
        return preds

    def inference(self, CFG, x_test, model):
        preds = model.predict(x_test)
        return preds      

class LogReg(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        
    def train(self, CFG, x_train, y_train):
        model =LogisticRegression(self.model_params)
        model.fit(x_train,y_train)
        return model

    def valid(self, CFG, x_valid, model):
        preds = model.predict(x_valid)
        return preds

    def inference(self, CFG, x_test, model):
        preds = model.predict(x_test)
        return preds
