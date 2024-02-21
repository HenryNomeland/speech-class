#imports
import pandas as pd
import sklearn
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeClassifier
import numpy as np
import random
import os
from statistics import mean

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
        
        self.original_df = input_df
        self.input_df = input_df
        self.features = input_df.columns[4:]
        self.not_features = list(input_df.columns)[0:4]
     
    # method for wrangling features
    def process(self, location_specificity="country", vowels_only=True):

        # extracting the specified type of place from the location feature
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
        self.input_df['location'] = self.input_df['location'].apply(extract_place)

        # encoding gender so that 0 => female and 1 => male
        self.input_df.gender = self.input_df.gender.replace(to_replace=['m', 'f'], value=[1, 0])

        # change age column to numeric
        self.input_df['age'] = pd.to_numeric(self.input_df['age'])

        # filter for only vowels if the option is selected
        if vowels_only==True:
            def phoneme_find(feature):
                return feature.split("-")[2]
            v_features = [f for f in self.input_df.columns[4:] if phoneme_find(f)[0] in ["A", "I", "E", "O", "U"]]
            feats = self.not_features + v_features
            self.input_df = self.input_df[self.input_df.columns.intersection(feats)]

        # identifying the first non-zero feature for each word and combining like columns
        def word_find(feature):
            return feature.split("-")[0][0:2] + "-" + feature.split("-")[1]
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
        
        self.X_m = X[y=="M"] # data that matches with the 'main' category of interest
        self.X_n = X[y=="N"] # data that matches with the other category
        self.y_m = y[y=="M"]
        self.y_n = y[y=="N"]
        self.X = X
        self.y = y
        self.n = len(self.y_m) #number of samples with main y label
    
    #method for fitting and finding accuracy
    def fit(self, model_type="rforest", cv_method="LOO", test_size=0.30, var_imp_type="mdi"):

        ### randomly partition the majority class into an odd-number of n-sized partitions
        np.random.shuffle(self.X_n)
        num_partitions = self.X_n.shape[0] // self.n
        necessary_size = np.arange(num_partitions*self.n)
        np.random.shuffle(necessary_size)
        partitions = np.split(necessary_size, num_partitions)

        ### setting up a few key objects
        var_imp_list = []
        self.classifier_dictionary = []
        ens_num=1
        accuracy_list = []
        
        ### loop through all partitions
        for random_values in partitions:
            ### establish X and y values from the random partition
            self.X_s = np.concatenate((self.X_m, self.X_n[random_values]), axis=0)
            self.y_s = np.concatenate((self.y_m, self.y_n[random_values]), axis=0)
    
            ### model selection based on function parameter
            print(f'\nFitting model of type {model_type}')
            if model_type=='rforest':
                self.clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
            elif model_type=='ridge_classification':
                self.clf = RidgeClassifier(alpha=0.0001)
            else:
                raise Exception("Improper model type provided.") 
            
            ### calculate accuracy using LOO cross validation
            if cv_method=="LOO":
                cv = LeaveOneOut()
                true_output = []
                predicted_output = []
                count=0
                mini_varimp_list = []
                for train, test in cv.split(self.X_s):
                    xtrain = self.X_s[train, :]
                    ytrain = self.y_s[train]
                    xtest = self.X_s[test, :]
                    ytest = self.y_s[test]
                    self.clf.fit(xtrain, ytrain)
                    prediction = self.clf.predict(xtest)
                    true_output.append(ytest[0])
                    predicted_output.append(prediction[0])
                    mini_varimp_list.append(self.calc_importance(var_imp_type))
                    if count%10 == 0: 
                        print(f'Fitting LOOCV split {count}')
                    count+=1
                #calculate and show accuracy
                var_imp_list.append(pd.concat(mini_varimp_list, axis=1).mean(axis=1))
                acc = accuracy_score(true_output, predicted_output)
                print(f"Learner {ens_num} Accuracy: {round(acc, 3)}")
                accuracy_list.append(acc)
    
            ### calculate accuracy using train/test/split
            if cv_method=="train-test":
                xtrain, xtest, ytrain, ytest = train_test_split(self.X_s, self.y_s, test_size=test_size)
                self.clf.fit(xtrain, ytrain)
                var_imp_list.append(self.calc_importance(var_imp_type))
                acc = self.clf.score(xtest, ytest)
                print(f"Learner {ens_num} Test Accuracy with {test_size*100}% test split: {round(acc, 3)}")
                accuracy_list.append(acc)
    
            ### calculate final aggregate variable importance measures
            self.var_imp = pd.concat(var_imp_list, axis=1).mean(axis=1)
            print(f"Learner {ens_num} Variable Importance Measures:")
            sorted_varimp = self.var_imp.sort_values(ascending=False)[0:10]
            print(pd.DataFrame(sorted_varimp).to_string(header=False))

            ### add to the dictionary for later use to be put together into an ensemble
            self.classifier_dictionary.append((str(ens_num),self.clf))
            ens_num += 1
    
        ### final fit of the model on all data
        self.clf = VotingClassifier(self.classifier_dictionary)
        self.clf.fit(self.X_s, self.y_s) # trains on the last randomized split set
        self.var_imp = self.var_imp.sort_values(ascending=False)

        ### report final accuracy
        print(f'\nAverage accuracy across all learners: {round(mean(accuracy_list),3)}')

    # method for calculating variable importance
    def calc_importance(self, X=None, type="mdi"):
        # calculates variable importance as the mean decrease in impurity
        if type=="mdi":
            mdi_importances = pd.Series(self.clf.feature_importances_, 
                                        index=self.data.columns[4:])
            return mdi_importances

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
        
        
        
        
        
        
        
    
    
        