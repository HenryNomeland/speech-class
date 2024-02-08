#imports
import pandas as pd
import sklearn
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeClassifier
import numpy as np
import random
import os

#input class
class h_input():
    def __init__(self, raw_data, id_vars=['id', 'gender', 'age', 'location', 'phoneme', 'word', 'midpoint_time']):
        
        self.raw_data = raw_data
        self.id_vars = id_vars
           
        input_df = raw_data.melt(id_vars=['id', 'gender', 'age', 'location', 'phoneme', 'word', 'midpoint_time'])
        input_df["value"] = pd.to_numeric(input_df["value"], errors='coerce')
        
        input_df = input_df.pivot_table(index=['id', 'gender', 'age', 'location'], 
                                    values=["value"], 
                                    columns=['variable', 'word', 'phoneme'], 
                                    fill_value=0)
        input_df.columns = input_df.columns.droplevel().to_flat_index().str.join("-")
        input_df = input_df.reset_index()
        
        def phoneme_find(feature):
            return feature.split("-")[0][0:2] + "-" + feature.split("-")[2]
        
        self.original_df = input_df
        self.input_df = input_df
        self.features = input_df.columns[4:]
        self.not_features = list(input_df.columns)[0:4]
        self.vowel_features = [f for f in self.features if phoneme_find(f)[0] in ["A", "I", "E", "O", "U"]]
     
    # method for wrangling features
    def process(self, location_specificity="country"):
        
        def extract_place(location):
            l_list = location.split("_")
            if location_specificity=="country":
                if len(l_list) > 2:
                    return l_list[2]
                elif len(l_list) > 1:
                    return l_list[1]
                else:
                    return l_list[0]
            if location_specificity=="region":
                if len(l_list) > 1:
                    return l_list[1]
                else:
                    return l_list[0]
            if location_specificity=="specific":
                return l_list[0]

        def word_find(feature):
            return feature.split("-")[0][0:2] + "-" + feature.split("-")[1]

        self.input_df['location'] = self.input_df['location'].apply(extract_place)
        self.input_df.gender = self.input_df.gender.replace(to_replace=['m', 'f'], value=[1, 0])
        self.input_df['age'] = pd.to_numeric(self.input_df['age'])

        opt_df = self.input_df.copy()[['id','gender','age','location']]

        words = [word_find(f) for f in self.input_df.columns[4:]]
        for w in words:
            for c in self.input_df.columns[4:]:
                if word_find(c) == w:
                    opt_df[w] = self.input_df[c]
                    break

        in_list = list(self.input_df.columns[4:])
        opt_list = list(opt_df.columns[4:])
        for f in range(len(opt_list)):
            for s in range(len(opt_df)):
                if opt_df.iloc[s, f+4] == 0:
                    for i in range(len(in_list)):
                        if word_find(in_list[i]) == opt_list[f]:
                            if self.input_df.iloc[s, i+4] != 0.0:
                                opt_df.iloc[s, f+4] = self.input_df.iloc[s,i+4]
                                break
                else:
                    continue

        self.input_df = opt_df
        return self.input_df
    
    def select_features(self, selected_features=['F1']):
        new_df = self.input_df.copy()[self.not_features]
        
        for c in self.input_df.columns[4:]:
            if "F1" in selected_features:
                if c[0:2] == "F1":
                    new_df[c] = self.input_df[c]
            if "F2" in selected_features:
                if c[0:2] == "F2":
                    new_df[c] = self.input_df[c]
            if "F3" in selected_features:
                if c[0:2] == "F2":
                    new_df[c] = self.input_df[c]
            if "duration" in selected_features:
                if c[0:2] == "du":
                    new_df[c] = self.input_df[c]
            if "F1mF2" in selected_features:
                if c[0:2] == "F1":
                    name = "F1mF2" + c[2:]
                    word = c.split("-")[1]
                    c2 = "F2-" + word
                    new_df[name] = self.input_df[c2] - self.input_df[c]
        
        self.input_df = new_df
        return self.input_df
        
    def normalize(self, method="z"):
        new_df = self.input_df.copy()
        new_df = new_df.drop(self.not_features, axis=1)

        for column in new_df.columns: 
            if method=="z":
                new_df[column]=(new_df[column]-new_df[column].mean())/new_df[column].std()
            if method=="minmax":
                new_df[column]=(new_df[column]-new_df[column].min())/(new_df[column].max()-new_df[column].min())
        
        self.input_df = pd.merge(self.input_df[self.not_features], 
                                 new_df[list(new_df.columns)], 
                                 left_index=True, right_index=True)
        return self.input_df
    
    def select_places(self, places=['uk', 'usa']):
        self.input_df = self.input_df.loc[self.input_df['location'].isin(places)].reset_index(drop=True)
        return self.input_df
    
    def revert(self):
        self.input_df = self.original_df.copy()
        return self.input_df

    def output_input_df(self, filename='input_df.csv'):
        self.input_df.to_csv(os.path.join(filename))
    
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
        
        self.n = len(self.y_m) #number of samples with main y label
        self.N = len(y) #total number of samples
        self.nfeatures = len(self.features)
     
    #method for fitting and finding accuracy
    def fit(self, model_type="rforest"):
        
        ### undersample the majority class and establish X and y values
        bs = self.X_n[np.random.choice(self.X_n.shape[0], self.n, replace=True)]
        X_s = np.concatenate((self.X_m, bs), axis=0)
        y_s = np.concatenate((self.y_m, self.y_n[:self.n]), axis=0)

        ### model selection based on function parameter
        print(f'Fitting model of type {model_type}')
        if model_type=='rforest':
            self.clf = RandomForestClassifier()
        elif model_type=='ridge_classification':
            self.clf = RidgeClassifier(alpha=0.0001)
        elif model_type=='knn':
            self.clf = sklearn.neighbors.KNeighborsClassifier()
        else:
            raise Exception("Improper model type provided.") 
        
        ### calculate accuracy using LOO cross validation
        cv = LeaveOneOut()
        true_output = []
        predicted_output = []
        var_imp = np.empty((2*self.n, self.nfeatures))
        count=0
        for train, test in cv.split(X_s):
            xtrain = X_s[train, :]
            ytrain = y_s[train]
            xtest = X_s[test, :]
            ytest = y_s[test]
            self.clf.fit(xtrain, ytrain)
            prediction = self.clf.predict(xtest)
            true_output.append(ytest[0])
            predicted_output.append(prediction[0])
            var_imp[count] = permutation_importance(self.clf, X_s, y_s).importances_mean
            if count%10 == 0: 
                print(f'Fitting LOOCV split {count}')
            count+=1

        #calculate and show accuracy
        print(f"\nAccuracy: {round(accuracy_score(true_output, predicted_output), 3)}")

        #final fit of the model on all data
        self.clf.fit(X_s, y_s)
        
        #calculate variable importance and show top 10
        var_importance_df = pd.DataFrame({"Feature":self.features, "Importance":np.mean(var_imp, axis=0)})
        print('\nVariable Importance Measurements:')
        print(var_importance_df.sort_values(by='Importance', ascending=False)[0:10])

    def sample_predict(self, index=0, custom=None):
        if isinstance(custom, pd.Series):
            prediction = self.clf.predict(custom[self.features].values.reshape(1, -1))
        else:
            prediction = self.clf.predict(self.data.iloc[index][self.features].values.reshape(1, -1))
        if prediction == "N":
            pred_label = f"not {self.y_main}"
        elif prediction == "M":
            pred_label = self.y_main
        print(f'Predicted location: {pred_label}')
    
if __name__ == "__main__":
    print("This file is not yet intended to be run as a script")   
        
        
        
        
        
        
        
    
    
        