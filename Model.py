import sklearn
import numpy as np
import torch

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from  sklearn.linear_model import SGDRegressor, Perceptron
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.preprocessing import StandardScaler

class SKModel(BaseEstimator, ClassifierMixin):
    def __init__(self, composer_model, instruments_model, withScalar = False):
        self.composer_model = composer_model
        self.instruments_model = instruments_model
        self.withScalar = withScalar
        self.scalar = StandardScaler()
        


    def partial_fit(self,x,y:torch.Tensor,classes=[None,None],sample_weight=None):
        (composer, instruments) = y
       
        if self.withScalar:
           
            self.scalar = self.scalar.partial_fit(x)
            x = self.scalar.transform(x)
        
        self.composer_model = self.composer_model.partial_fit(x,composer,classes=classes[0])
        self.instruments_model = self.instruments_model.partial_fit(x,instruments, classes = classes[1])
        
        return self

    def predict(self,x):
        
        if self.withScalar:
            x = self.scalar.transform(x)
        
        return self.composer_model.predict(x), self.instruments_model.predict(x)
        
    



        

        


        


