import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def MarvellousplayPredictor(data_path):
   
    #step 1 : Load data
    data=pd.read_csv(data_path,index_col=0)
    print("dataset loaded successfully with the size:",len(data))
    
    #step 2 : clean manipulate and prepare data
    feature_names=["Wether","Temperature"]
    print("feature names are",feature_names)
    
    Wether=data.Wether
    Temperature=data.Temperature
    Play=data.Play
    
    #creating LabelEncoder
    le=preprocessing.LabelEncoder()
    
    #converting string labels into numbers.
    Wether_encoded=le.fit_transform(Wether)
    print(Wether_encoded)
    
    #converting string labels into numbers
    temp_encoded=le.fit_transform(Temperature)
    Label=le.fit_transform(Play)
    
    print(temp_encoded)
    
    #combining wether and temp into single list of tuples
    features=list(zip(Wether_encoded,temp_encoded))
    #{[0,2],[0,2],[0,2],[0,2]}
    
    #step 3 : train data
    model=KNeighborsClassifier(n_neighbors=3)
    
    #train the model using training set
    model.fit(features,Label)
    
    
    #step 4: Test data
    predicted=model.predict([[0,2]]) #0:overcast,2:Mild
    print(predicted)
    
    if predicted ==1:
        print("you can play")
    else:
        print("you cant play")
    
def main():
    print("_____playpredictor______")
    
    print("machine learning application")
    
    print("play predictor application using k nearest knighbor algorithm")
    
    #print("enter the path of the file which contains dataset")
    #data_path=input()
    
    MarvellousplayPredictor("playpredictor.csv")

if __name__ =="__main__":
    main()


