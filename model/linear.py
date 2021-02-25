class Rid(Base_Model):
    def __init__(self):
      self.model = None
    def fit(self,x_train,y_train,x_valid,y_valid):
        model =Ridge(
            alpha=1, #L2係数
            max_iter=1000,
            random_state=10,
                              )
        model.fit(x_train,y_train)
        return model

    def predict(self,model,features):
      return model.predict(features)
