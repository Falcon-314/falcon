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
    def __init__(self, model_params):
        self.model_params = model_params
        self.model = None
    
    def fit(self,x_train,y_train):
        model =Ridge(self.model_params)
        model.fit(x_train,y_train)
        self.model = model
        return model

    def predict(self,x_test):
        return self.model.predict(x_test)

    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train)
        oof_df = self.predict(x_valid)
        return oof_df, self.model   

class LinearLasso(Base_Model):
    def __init__(self, model_params):
        self.model_params = model_params
        self.model = None
    
    def fit(self,x_train,y_train):
        model =Lasso(self.model_params)
        model.fit(x_train,y_train)
        self.model = model
        return model

    def predict(self,x_test):
        return self.model.predict(x_test)

    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train)
        oof_df = self.predict(x_valid)
        return oof_df, self.model   
    
class LinearReg(Base_Model):
    def __init__(self):
        self.model = None
    
    def fit(self,x_train,y_train):
        model =LinearRegression()
        model.fit(x_train,y_train)
        self.model = model
        return model

    def predict(self,x_test):
        return self.model.predict(x_test)

    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train)
        oof_df = self.predict(x_valid)
        return oof_df, self.model      
    
class BayesReg(Base_Model):
    def __init__(self):
        self.model = None
    
    def fit(self,x_train,y_train):
        model =BayesianRidge(compute_score=True)
        model.fit(x_train,y_train)
        self.model = model
        return model

    def predict(self,x_test):
        return self.model.predict(x_test)

    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train)
        oof_df = self.predict(x_valid)
        return oof_df, self.model      

class LogReg(Base_Model):
    def __init__(self, model_params):
        self.model_params = model_params
        self.model = None
    
    def fit(self,x_train,y_train):
        model =LogisticRegression(self.model_params)
        model.fit(x_train,y_train)
        self.model = model
        return model

    def predict(self,x_test):
        return self.model.predict(x_test)

    def train(self,x_train,y_train,x_valid,y_valid):
        self.fit(x_train,y_train)
        oof_df = self.predict(x_valid)
        return oof_df, self.model   
