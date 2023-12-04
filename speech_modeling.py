#imports
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeClassifier
import numpy as np
import random

#input class
class h_input():
    def __init__(self, raw_data, id_vars=['id', 'gender', 'age', 'location', 'phoneme', 'word', 'midpoint_time']):
        
        self.raw_data = raw_data
        self.id_vars = id_vars
           
        melted_df = raw_data.melt(id_vars=['id', 'gender', 'age', 'location', 'phoneme', 'word', 'midpoint_time'])
        melted_df["value"] = pd.to_numeric(melted_df["value"], errors='coerce')
        
        input_df = melted_df.pivot_table(index=['id', 'gender', 'age', 'location'], 
                                    values=["value"], 
                                    columns=['variable', 'word', 'phoneme'], 
                                    fill_value=0)
        print(input_df)
        input_df.columns = input_df.columns.droplevel().to_flat_index().str.join("-")
        input_df = input_df.reset_index()
        features = input_df.columns[4:]
        
        #finds the phoneme of a given feature
        def phoneme_find(feature):
            return feature.split("-")[2]

        #finds the word of a given feature
        def word_find(feature):
            return feature.split("-")[1]
        
        #finds the country from a given location
        def extract_country(location):
            l_list = location.split("_")
            if len(l_list) > 1:
                return l_list[1]
            else:
                return l_list[0]
        
        self.original_df = input_df
        self.input_df = input_df
        self.features = features
        self.not_features = list(input_df.columns)[0:4]
        self.f1_features = [f for f in features if f[0:3] == "F1-"]
        self.f1v_features = [f for f in self.f1_features if phoneme_find(f)[0] in ["A", "I", "E", "O", "U", "Y"]]
        self.f2_features = [f for f in features if f[0:3] == "F2-"]
        self.f2v_features = [f for f in self.f2_features if phoneme_find(f)[0] in ["A", "I", "E", "O", "U", "Y"]]
        self.f3_features = [f for f in features if f[0:3] == "F2-"]
        self.f3v_features = [f for f in self.f3_features if phoneme_find(f)[0] in ["A", "I", "E", "O", "U", "Y"]]
        self.dur_features = [f for f in features if f[0:3] == "dur"]
        self.durv_features = [f for f in self.dur_features if phoneme_find(f)[0] in ["A", "I", "E", "O", "U", "Y"]]
     
    #method for fitting and finding accuracy
    def process(self, drop_cols=True):
        new_df = self.original_df.copy()
        other_df = self.original_df.copy()
        not_features = list(self.input_df.columns)[0:4]

        if drop_cols==True:
            for column in self.features:
                count = (new_df[column] == 0).sum()
                if count >= 0.50*len(new_df):
                    new_df = new_df.drop(columns=column)
            new_features = list(set(self.features).intersection(list(new_df.columns)))
        
        self.input_df = pd.merge(other_df[not_features], new_df[new_features], left_index=True, right_index=True)
                
        return self.input_df
        
    def normalize(self, method="z"):
        new_df = self.original_df.copy()
        not_features = list(self.input_df.columns)[0:4]

        if method=="z":
            for column in self.features: 
                new_df[column]=(new_df[column]-new_df[column].mean())/new_df[column].std()
                
        if method=="minmax":
            for column in self.features: 
                new_df[column]=(new_df[column]-new_df[column].min())/(new_df[column].max()-new_df[column].min())
        
        self.input_df = pd.merge(self.input_df[not_features], new_df[self.features], left_index=True, right_index=True)
                
        return self.input_df
    
    def revert(self):
        self.input_df = self.original_df.copy()
        return self.input_df

#model class
class h_model():
    def __init__(self, data, features, y_feature, y_main):
        self.data = data
        self.features = features
        self.y_feature = y_feature
        self.y_main = y_main
        
        X = data[features].values
        y = data[y_feature].values
        y[y==y_main] = "M"
        y[y!="M"] = "N"
        
        self.X_m = X[y=="M"]
        self.X_n = X[y=="N"]
        self.y_m = y[y=="M"]
        self.y_n = y[y=="N"]
        self.X = X
        self.y = y
        
        self.n = len(self.y_m)
        self.N = len(y)
        self.nfeatures = len(self.features)
     
    #method for fitting and finding accuracy
    def fit(self, model_type="rforest", NUM_SAMPLES=5):
        
        #initiate leave one out cross validation class
        cv = LeaveOneOut()
        #create lists for measuring final accuracy
        true_output = []
        predicted_output = []
        var_imp = np.empty((NUM_SAMPLES*2*self.n, self.nfeatures))
        count = 0
        #loop through the desired number of bootstrap samples
        print(f'Fitting Model of type {model_type}')
        for i in range(NUM_SAMPLES):
            #undersample the majority class
            bs = self.X_n[np.random.choice(self.X_n.shape[0], self.n, replace=True)]
            X_s = np.concatenate((self.X_m, bs), axis=0)
            y_s = np.concatenate((self.y_m, self.y_n[:self.n]), axis=0)
            print(f"Modeling Sample {i+1} of {NUM_SAMPLES}")
            #loop over every split and make predictions
            for train, test in cv.split(X_s):
                xtrain = X_s[train, :]
                ytrain = y_s[train]
                xtest = X_s[test, :]
                ytest = y_s[test]
                
                ### model selection based on function parameter
                if model_type=='rforest':
                    m = RandomForestClassifier(random_state=1)
                elif model_type=='ridge_classification':
                    m = RidgeClassifier(alpha=0.0001, random_state=0)
                else:
                    raise Exception("Improper model type provided.") 
                ###
                
                m.fit(xtrain, ytrain)
                prediction = m.predict(xtest)
                true_output.append(ytest[0])
                predicted_output.append(prediction[0])
                var_imp[count] = permutation_importance(m, X_s, y_s, random_state=20, n_jobs=2).importances_mean
                count += 1
        #calculate accuracy
        print(f"\nAccuracy: {round(accuracy_score(true_output, predicted_output), 3)}")
        
        #calculate variable importance and show top 10
        res_df = pd.DataFrame({"Feature":self.features, "Importance":np.mean(var_imp, axis=0)})
        print('\nVariable Importance Measurements:')
        print(res_df.sort_values(by='Importance', ascending=False)[0:10])
        
        
        
        
        
        
        
        
        
        
        
    
    
        