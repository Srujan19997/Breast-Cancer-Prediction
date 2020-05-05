
import tkinter as tk
import pandas as pd
import numpy as np
import random
from tkinter import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from PIL import ImageTk, Image
from tkinter import filedialog
import os

#initiating tk window

window=tk.Tk()
window.title("Predict Employee Turnover")
window.geometry("1000x1000")

#label
title=tk.Label(text="Welcome!!!",font=("Times New Roman",20))
title.grid(row=0,column=0)
lab=tk.Label(text="Enter the following Details ",font=("Times New Roman",14))
lab.grid(row=1,column=0)
varb=tk.Label(text="Satisfaction Level")
varb.grid(row=4,column=0)
varb1=tk.Label(text="Last Evaluvation")
varb1.grid(row=5,column=0)
varb2=tk.Label(text="Time spent at company")
varb2.grid(row=6,column=0)
varb3=tk.Label(text="Work Accident")
varb3.grid(row=7,column=0)

#Accepting the Entries
entry1=tk.Entry()
entry1.grid(row=4,column=2)
entry2=tk.Entry()
entry2.grid(row=5,column=2)
entry3=tk.Entry()
entry3.grid(row=6,column=2)
entry4=tk.Entry()
entry4.grid(row=7,column=2)



# Logistic Regression
def display():
    hr = pd.read_csv('C://Users/shreya devi/Downloads/turnover.csv')
    col_names = hr.columns.tolist()
    hr=hr.rename(columns = {'sales':'department'})
    hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])
    hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])
    cat_vars=['department','salary']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(hr[var], prefix=var)
        hr1=hr.join(cat_list)
        hr=hr1
    hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)
    hr.columns.values
    hr_vars=hr.columns.values.tolist()
    y=['left']
    X=[i for i in hr_vars if i not in y]
    cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident'] 
    X=hr[cols]
    y=hr['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    cm1=metrics.confusion_matrix(y_test, logreg.predict(X_test))
    total=sum(sum(cm1))
    accuracy=(cm1[0,0]+cm1[1,1])/total
    sensitivity=cm1[0,0]/(cm1[0,0]+cm1[0,1])
    specificity=cm1[1,1]/(cm1[1,0]+cm1[1,1])
    
    title=tk.Label(text="Logistic Regression",font=("Times New Roman",16))
    title.grid()
    greet_display=tk.Text(height=10,width=20)
    greet_display.grid()
    greet_display.insert(tk.END,"Confusion Matrix:\n")
    greet_display.insert(tk.END,cm1)
    greet_display.insert(tk.END,"\n")
    greet_display.insert(tk.END,"Accuracy:\n")
    greet_display.insert(tk.END,accuracy)
    greet_display.insert(tk.END,"\n")
    greet_display.insert(tk.END,"Sensitivity:\n")
    greet_display.insert(tk.END,sensitivity)
    greet_display.insert(tk.END,"\n")
    greet_display.insert(tk.END,"Specificity:\n")
    greet_display.insert(tk.END,specificity)
    
 #random forest
def display1():
    hr = pd.read_csv('C://Users/shreya devi/Downloads/turnover.csv')
    col_names = hr.columns.tolist()
    hr=hr.rename(columns = {'sales':'department'})
    hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])
    hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])
    cat_vars=['department','salary']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(hr[var], prefix=var)
        hr1=hr.join(cat_list)
        hr=hr1
    hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)
    hr.columns.values
    hr_vars=hr.columns.values.tolist()
    y=['left']
    X=[i for i in hr_vars if i not in y]

    cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
    X=hr[cols]
    y=hr['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
   
    cm1=metrics.confusion_matrix(y_test, rf.predict(X_test))
    total=sum(sum(cm1))
    accuracy=(cm1[0,0]+cm1[1,1])/total
    sensitivity=cm1[0,0]/(cm1[0,0]+cm1[0,1])
    specificity=cm1[1,1]/(cm1[1,0]+cm1[1,1])
   
    title=tk.Label(text="Random Forest",font=("Times New Roman",16))
    title.grid()
    greet_display1=tk.Text(height=10,width=20)
    greet_display1.grid()
    greet_display1.insert(tk.END,"Confusion Matrix:\n")
    greet_display1.insert(tk.END,cm1)
    greet_display1.insert(tk.END,"\n")
    greet_display1.insert(tk.END,"Accuracy:\n")
    greet_display1.insert(tk.END,accuracy)
    greet_display1.insert(tk.END,"\n")
    greet_display1.insert(tk.END,"Sensitivity:\n")
    greet_display1.insert(tk.END,sensitivity)
    greet_display1.insert(tk.END,"\n")
    greet_display1.insert(tk.END,"Specificity:\n")
    greet_display1.insert(tk.END,specificity)
   
   
    
#svm
def display2():
    hr = pd.read_csv('C://Users/shreya devi/Downloads/turnover.csv')
    col_names = hr.columns.tolist()
    hr=hr.rename(columns = {'sales':'department'})
    hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])
    hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])
    cat_vars=['department','salary']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(hr[var], prefix=var)
        hr1=hr.join(cat_list)
        hr=hr1
    hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)
    hr.columns.values
    hr_vars=hr.columns.values.tolist()
    y=['left']
    X=[i for i in hr_vars if i not in y]
    cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
    X=hr[cols]
    y=hr['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)
   
    cm1=metrics.confusion_matrix(y_test,svc.predict(X_test))
    total=sum(sum(cm1))
    accuracy=(cm1[0,0]+cm1[1,1])/total
    sensitivity=cm1[0,0]/(cm1[0,0]+cm1[0,1])
    specificity=cm1[1,1]/(cm1[1,0]+cm1[1,1])
    
    title=tk.Label(text="Support Vector Machine",font=("Times New Roman",16))
    title.grid()
    greet_display2=tk.Text(height=10,width=20)
    greet_display2.grid()
    greet_display2.insert(tk.END,"Confusion Matrix:\n")
    greet_display2.insert(tk.END,cm1)
    greet_display2.insert(tk.END,"\n")
    greet_display2.insert(tk.END,"Accuracy:\n")
    greet_display2.insert(tk.END,accuracy)
    greet_display2.insert(tk.END,"\n")
    greet_display2.insert(tk.END,"Sensitivity:\n")
    greet_display2.insert(tk.END,sensitivity)
    greet_display2.insert(tk.END,"\n")
    greet_display2.insert(tk.END,"Specificity:\n")
    greet_display2.insert(tk.END,specificity)
 
    
def quit():
    return window.destroy()

def roc():
    #title=tk.Label(text="ROC Curve Comparing Logistic Regression,SVM,Random Forest",font=("Times New Roman",16))
    #title.grid()
    
    hr = pd.read_csv('C://Users/shreya devi/Downloads/turnover.csv')
    col_names = hr.columns.tolist()
    hr=hr.rename(columns = {'sales':'department'})
    hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])
    hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])
    cat_vars=['department','salary']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(hr[var], prefix=var)
        hr1=hr.join(cat_list)
        hr=hr1
    hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)
    hr.columns.values
    hr_vars=hr.columns.values.tolist()
    y=['left']
    X=[i for i in hr_vars if i not in y]
    cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 
    X=hr[cols]
    y=hr['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    
    svc = SVC(probability=True)
    svc.fit(X_train, y_train)

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

    svc_roc_auc = roc_auc_score(y_test, svc.predict(X_test))
    svc_fpr,svc_tpr, svc_thresholds = roc_curve(y_test, svc.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
    plt.plot(svc_fpr, svc_tpr, label='svm (area = %0.2f)' % svc_roc_auc)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC')
    plt.show()

def predict():
    hr = pd.read_csv('C://Users/shreya devi/Downloads/turnover.csv')
    col_names = hr.columns.tolist()
    hr=hr.rename(columns = {'sales':'department'})
    hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])
    hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])
    cat_vars=['department','salary']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(hr[var], prefix=var)
        hr1=hr.join(cat_list)
        hr=hr1
    hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)
    hr.columns.values
    hr_vars=hr.columns.values.tolist()
    y=['left']
    X=[i for i in hr_vars if i not in y]

    cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident'] 
    X=hr[cols]
    y=hr['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    a=entry1.get()
    b=entry2.get()
    c=entry3.get()
    d=entry4.get()
    #if(a!=NULL||b!=NULL||c!=NULL||d!=NULL):
	
    #p=rf.predict(X_test)
    #predict=random.choice(p)
    predict=(rf.predict([[a,b,c,d]]))
    
    

    #title=tk.Label(text="Random Forest",font=(" The Employee may",16))
    #title.grid()
    greet_disp=tk.Text(height=4,width=60)
    greet_disp.grid()
    if(predict==[0]):
    	#lab=tk.Label(text="The Employee may leave the organisation")
    	#lab.grid()
        greet_disp.insert(tk.END," The Employee may leave the organisation")
    	#greet_disp.insert(tk.END,predict)
	#greet_disp=tk.Message(" The Employee may leave the organisation")
        greet_disp.grid()
   
    else:
    	#lab1=tk.Label(text="The Employee may leave the organisation")
    	#lab1.grid()
        greet_disp.insert(tk.END,"The Employee will mostly not leave the organisation")
	#greet_disp=tk.(" The Employee may leave the organisation")
        greet_disp.grid()
   

   


#buttons


button1=tk.Button(text="Logistic Regression",command=display)
button2=tk.Button(text="Support vector machine",command=display2)
button3=tk.Button(text="Random Forest",command=display1)
btn = Button( window,text='Roc Curve', command=roc)
#btn1 = Button( window,text='svm roc', command=roc2)
#btn2 = Button( window,text='Random forest roc', command=roc3)
button =tk.Button(text="Quit",command=quit)
submit=tk.Button(text="Predict",command=predict)



button1.grid(row=11,column=0)
button2.grid(row=11,column=1)
button3.grid(row=11,column=2)
button.grid(row=11,column=3)
btn.grid(row=11,column=4)
submit.grid(row=8,column=2)
#btn1.grid(row=5,column=5)
#btn2.grid(row=6,column=5)



window.mainloop()


