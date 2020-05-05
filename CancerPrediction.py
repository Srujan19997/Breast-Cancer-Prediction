#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:16:06 2018

@author: srujan
"""

import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import time 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import mglearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

window=tk.Tk()
vscrollbar = tk.Scrollbar(window)

window.geometry("2000x1000")
window.title("BREAST CANCER DIAGNOSIS")
cwgt=Canvas(window,width=2000,height=1000,yscrollcommand=vscrollbar.set)

vscrollbar.config(command=cwgt.yview)
vscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
f=tk.Frame(cwgt) #Create the frame which will hold the widgets

cwgt.pack(side="right", fill="both", expand=True)

#Updated the window creation
cwgt.create_window(0,0,window=f, anchor='nw')

#Added more content here to activate the scroll

#Removed the frame packing
#f.pack()

#Updated the screen before calculating the scrollregion


image1=PhotoImage(file="F:\\MLPROJECT\\ML_scikit_Cancer\\pink.PGM")


# keep a link to the image to stop the image being garbage collected
cwgt.img=image1
cwgt.create_image(0, 0, anchor=NW, image=image1)
#b1=Button(cwgt, text="Hello", bd=0)
window.iconbitmap('F:\\MLPROJECT\\ML_scikit_Cancer\\logo_bxb_icon.ico')
ip_label=tk.Label(text="Upload Your Report CSV File: ")
#ip_label.grid(column=1,row=1)



cwgt.create_window(1,1, window=ip_label, anchor=NW)

def relapse(): 
    for label in labels:
        label.destroy()
        


relapse_button=tk.Button(text="relapse",command=relapse)
#relapse_button.grid(column=2,row=20)
cwgt.create_window(600,1, window=relapse_button, anchor=NW)







quit_button=tk.Button(text="quit",command=quit)

cwgt.create_window(700,1, window=quit_button, anchor=NW)

class CancerClassification(object):
    def __init__(self,pid,ctype):
        self.pid=pid
        self.ctype=ctype
    def tostring(self):
        return str(self.pid)+"--->"+self.ctype
    

def browse():
    
    global filename
    filename=filedialog.askopenfilename()
    global labels
    labels=[]
    #print(filename)
    
    if(filename.endswith('.csv')):
        

        
            global ip_csv
            labelFilepath=Label(text=":"+filename,bg='cyan')
        
            cwgt.create_window(300,1, window=labelFilepath, anchor=NW)

            ip_path_label=tk.Label(text=filename)
            #ip_path_label.grid(row=1,column=3)
            knn_button=tk.Button(text="KNN",command=KNN)
            cwgt.create_window(10,50, window=knn_button, anchor=NW)


            svm_button=tk.Button(text="SVM",command=SVM)
            #svm_button.grid(column=2,row=2)
            cwgt.create_window(10,100, window=svm_button, anchor=NW)


            cnn_button=tk.Button(text="CNN",command=CNN)
           # cnn_button.grid(column=3,row=2)
            cwgt.create_window(10,150, window=cnn_button, anchor=NW)
 
            
            print(filename)
            #filename=filename.replace("/","\\")
            ip_csv=pd.read_csv(filename,delimiter=',')
            
            print(filename)
    else:
        ip_path_label=tk.Label(text="selected is not a csv file, please select again")
        ip_path_label.grid(row=1,column=3)
        cwgt.create_window(20,50, window=ip_path_label, anchor=NW)

    
    

def KNN():
    c=load_breast_cancer()
   
    ip_csv.drop(['id'], axis = 1, inplace = True)
    ip_np_data=ip_csv.values
    print("---------------numpy array formatted-----------------")
        #print(ip_np_data)
    print("---------------loading module to train---------------")
    knn=KNeighborsClassifier()
    print("---------------splitting testing and training data---------------")
    X_train,X_test,y_train,y_test=train_test_split(c.data,c.target,stratify=c.target,test_size=0.20,random_state=42)
    print("--------------------training data------------------")
    start_time=time.time()
    knn.fit(X_train,y_train)
    end_time=time.time()
    print("training time :",end_time-start_time)
    print("---------------testing accuracy data---------------\n")
    
    train_acc=(knn.score(X_train,y_train))*100
    test_acc=(knn.score(X_test,y_test))*100
    print(type(train_acc))
    print(test_acc)
    statement="Training data accuracy score: "
    label_training=tk.Label(text=statement)
    #label_training_acc.grid(row=3,column=1)
    cwgt.create_window(500,100, window=label_training, anchor=NW)
    label_training_acc=tk.Label(text=train_acc)
    #label_training_acc.grid(row=3,column=2)
    cwgt.create_window(700,100, window=label_training_acc, anchor=NW)
   

                  
    statement="Testing data accuracy score: "
    label_testing=tk.Label(text=statement)
    #label_testing_acc.grid(row=4,column=1)
    cwgt.create_window(500,130, window=label_testing, anchor=NW)
    label_testing_acc=tk.Label(text=test_acc)
    #label_testing_acc.grid(row=4,column=2)
    cwgt.create_window(700,130, window=label_testing_acc, anchor=NW)
    matrix=metrics.confusion_matrix(y_test, knn.predict(X_test))
    label_matrix=tk.Label(text=matrix)
    statement="Confusion Matrix is: "
    label_M=tk.Label(text=statement)
    cwgt.create_window(500,160, window=label_M, anchor=NW)
    cwgt.create_window(650,160, window=label_matrix, anchor=NW)  
    
    matrix=metrics.confusion_matrix(y_test, knn.predict(X_test))
    label_matrix=tk.Label(text=matrix)
    statement="Classification Report is: "
    label_M=tk.Label(text=statement)
    cwgt.create_window(500,210, window=label_M, anchor=NW)
    cReport=classification_report(y_test,knn.predict(X_test))
    label_cReport=tk.Label(text=cReport)
    cwgt.create_window(670,210, window=label_cReport, anchor=NW)

    
   
                       
    
    print("---------------predicting input data---------------\n")
    result_list=knn.predict(ip_np_data)
    ip_ids=pd.read_csv(filename, usecols=['id'])
    print("ID\t\t\tTYPE")
    diagnose_report=[]
    cn=CancerClassification(0,"t")
    var=350
    for i in range(0,ip_ids.size):
        
        ctype='malignant' if result_list[i]==0 else 'begnine'
        #print(ctype)
        diagnose_report.append(CancerClassification(ip_ids.values[i],ctype).tostring())
    print(diagnose_report)  
    for i in range(0,len(diagnose_report)):
        label=tk.Label(text=diagnose_report[i])
        #label.grid(row=var+i,column=1)
        labels.append(label)
        cwgt.create_window(500,var+i*30, window=label, anchor=NW)
    print(dir(knn))
    print(knn.kneighbors_graph())

        
def CNN():
        
    cancer=load_breast_cancer()
    data = pd.read_csv("F:\\MLPROJECT\\ML_scikit_Cancer\\breast-cancer-dataset.csv", index_col=False)
    data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
    data = data.set_index('id')
    del data['Unnamed: 32']
    Y = data['diagnosis'].values
    X = data.drop('diagnosis', axis=1).values
    
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=21)
    #Scaling
    scaler=StandardScaler()
    X_train_scaled=scaler.fit(X_train).transform(X_train)
    X_test_scaled = scaler.fit(X_test).transform(X_test)
    
    mlp=MLPClassifier(max_iter=10000,random_state=42)
    mlp.fit(X_train_scaled,y_train)
    print('Accuracy on training set after scaling :{:3f}'.format(mlp.score(X_train_scaled,y_train)))
    print('Accuracy on test set after scanp.ling:{:3f}'.format(mlp.score(X_test_scaled,y_test)))
    
    train_acc=(mlp.score(X_train_scaled,y_train))*100
    
    statement="Training data accuracy score: "
    label_training=tk.Label(text=statement)
    #label_training_acc.grid(row=3,column=1)
    cwgt.create_window(500,100, window=label_training, anchor=NW)
    label_training_acc=tk.Label(text=train_acc)
    #label_training_acc.grid(row=3,column=2)
    cwgt.create_window(700,100, window=label_training_acc, anchor=NW)

    
    
    test_acc=(mlp.score(X_test_scaled,y_test))*100
    statement="Testing data accuracy score: "
    label_testing=tk.Label(text=statement)
    #label_testing_acc.grid(row=4,column=1)
    cwgt.create_window(500,130, window=label_testing, anchor=NW)
    label_testing_acc=tk.Label(text=test_acc)
    #label_testing_acc.grid(row=4,column=2)
    cwgt.create_window(700,130, window=label_testing_acc, anchor=NW)
    matrix=metrics.confusion_matrix(y_test, mlp.predict(X_test_scaled))
    label_matrix=tk.Label(text=matrix)
    statement="Confusion Matrix is: "
    label_M=tk.Label(text=statement)
    cwgt.create_window(500,160, window=label_M, anchor=NW)
    cwgt.create_window(650,160, window=label_matrix, anchor=NW)
    
    matrix=metrics.confusion_matrix(y_test, mlp.predict(X_test_scaled))
    label_matrix=tk.Label(text=matrix)
    statement="Classification Report is: "
    label_M=tk.Label(text=statement)
    cwgt.create_window(500,210, window=label_M, anchor=NW)
    cReport=classification_report(y_test,mlp.predict(X_test_scaled))
    label_cReport=tk.Label(text=cReport)
    cwgt.create_window(670,210, window=label_cReport, anchor=NW)

    
    

    inputData=pd.read_csv(filename)
    ip_ids=pd.read_csv(filename, usecols=['id'])
    
    ip_ids=ip_ids.values
    for i in range(0,len(ip_ids)):
        print(ip_ids[i])
    temp = inputData.drop('id', axis=1).values
    scalertemp=StandardScaler().fit(temp)
    temp_scaled=scaler.transform(temp);
    result=mlp.predict(temp_scaled)
    
    
    
    report=[]
    var=350
    for i in range(0,ip_ids.size):
        ctype='malignant' if result[i]=='1' else 'begnine'
        report.append(CancerClassification(ip_ids[i],ctype).tostring())
    for i in range(0,len(report)):
        label=tk.Label(text=report[i])
        #label.grid(row=var+i,column=1)
        labels.append(label)
        cwgt.create_window(500,var+i*30, window=label, anchor=NW)
    print(dir(mlp))
           

def SVM():
    cancer=load_breast_cancer()
    data=pd.read_csv("F:\\MLPROJECT\\ML_scikit_Cancer\\breast-cancer-dataset.csv",delimiter=',')
    
    data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
    data = data.set_index('id')
    
    del data['Unnamed: 32']

    Y = data['diagnosis'].values
    X = data.drop('diagnosis', axis=1).values
    #print(X)
    print(data.groupby('diagnosis').size())
    inputData=pd.read_csv(filename)
    temp = inputData.drop('id', axis=1).values
    scaler=StandardScaler().fit(temp)
    temp_scaled=scaler.transform(temp);
                            
        
            
    model = SVC(C=2.0, kernel='rbf')
    
    model.fit(X,Y)
    print(model.score(X,Y))
    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=21)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    svm=SVC(C=2.0,kernel='rbf')
    svm.fit(X_train_scaled, Y_train)
            #svm.fit(X_train,X_test)
    print(svm.score(X_train_scaled,Y_train))
    print(svm.score(X_test_scaled,Y_test))

    
    print('prediction:',svm.predict(temp_scaled))
    result=svm.predict(temp_scaled)
    
    
    train_acc=(svm.score(X_train_scaled,Y_train))*100
    test_acc=(svm.score(X_test_scaled,Y_test))*100
    print(type(train_acc))
    #print(test_acc)
    statement="Training data accuracy score: "
    label_training=tk.Label(text=statement)
    cwgt.create_window(500,100, window=label_training, anchor=NW)

    #label_training_acc.grid(row=3,column=1)
    label_training_acc=tk.Label(text=train_acc)
    #label_training_acc.grid(row=3,column=2)
    cwgt.create_window(700,100, window=label_training_acc, anchor=NW)

                       
    statement="Testing data accuracy score: "
    label_testing=tk.Label(text=statement)
    #label_testing_acc.grid(row=4,column=1)
    cwgt.create_window(500,130, window=label_testing, anchor=NW)

    label_testing_acc=tk.Label(text=test_acc)
    #label_testing_acc.grid(row=4,column=2)
    cwgt.create_window(700,130, window=label_testing_acc, anchor=NW)
    
    
    matrix=metrics.confusion_matrix(Y_test, svm.predict(X_test_scaled))
    label_matrix=tk.Label(text=matrix)
    statement="Confusion Matrix is: "
    label_M=tk.Label(text=statement)
    cwgt.create_window(500,160, window=label_M, anchor=NW)
    cwgt.create_window(650,160, window=label_matrix, anchor=NW)
    
    statement="Classification Report is: "
    label_M=tk.Label(text=statement)
    cwgt.create_window(500,210, window=label_M, anchor=NW)

    cReport=classification_report(Y_test,svm.predict(X_test_scaled))
    label_cReport=tk.Label(text=cReport)
    cwgt.create_window(670,210, window=label_cReport, anchor=NW)

    

    ip_ids=pd.read_csv(filename, usecols=['id'])
    ip_ids=ip_ids.values
    report=[]
    var=350
    for i in range(0,ip_ids.size):
        ctype='malignant' if result[i]=='1' else 'begnine'
        report.append(CancerClassification(ip_ids[i],ctype).tostring())
    for i in range(0,len(report)):
        label=tk.Label(text=report[i])
        #label.grid(row=var+i,column=1)
        labels.append(label)

        cwgt.create_window(500,var+i*30, window=label, anchor=NW)

    print(report)
    print(dir(svm))
    c_report=[]
    
    classify_report(matrix,cReport,train_acc=train_acc,test_acc=test_acc)
    
def classify_report(matrix,c_report,train_acc,test_acc):
    report_win=tk.Tk()
    test_button=tk.Button(text="test")
    test_button.pack()
    report_win.mainloop()
    
    
def quit():
    window.destroy()
        
        
        
browse_button=tk.Button(text="browse file:  ",command=browse)
cwgt.create_window(210,1, window=browse_button, anchor=NW) 
    

window.update()
cwgt.config(scrollregion=cwgt.bbox("all"))




    
window.mainloop()

