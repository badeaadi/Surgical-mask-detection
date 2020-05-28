# Badea Adrian Catalin, 2nd year, grupa 235, Facultatea de Matematica si Informatica


import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from datetime import datetime
# https://www.programiz.com/python-programming/datetime/current-datetime

import librosa.display
# https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html



names = [
     # name of classifier / prediction accuraccy on validation
     #"Logistic Regression", #0.735
     #"Linear SVM",  # 0.7 
     #"Decision Tree",  # 0.667
     #"Random Forest", #0.650
     #"Neural Net Classifier",  # 0.734
     #"AdaBoost", # 0.694
     "AdaBoost", # 0.694

]

classifiers = [
    
     #LogisticRegression(),
     #SVC(kernel="linear", C=0.025),
     #DecisionTreeClassifier(max_depth=7),
     #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
     #MLPClassifier(alpha=1, max_iter=1000),
     #AdaBoostClassifier(),
     AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
]


   
if __name__ == '__main__':

        
    # This is the main python file, which predicts on validation set
    # As it is very similar to the tester.py, the explanation will be only on the 
    # different part, which regards confusion matrixes on the ensemble
    # Training data, number of files read (can choose to read more or less)
    
    read_files = 8000
    
    train_path_start = ['', '10000', '1000', '100', '10']
    train_audios_paths = ['train/train/' + train_path_start[len(str(x))]  + str(x) + ".wav" for x in range(1, read_files + 1)]
    train_x_list = []
    


    
    print('Started reading train data at: ', datetime.now())
                
    for path in train_audios_paths:
       
        aud, sr = librosa.load(path)
       
       
        x = librosa.stft(aud, n_fft = 8192, win_length = 8192, center = False)
       
       
        
        xstft = librosa.amplitude_to_db(abs(x))
       
       
        nx, ny = xstft.shape
        new_xstft = xstft.reshape(nx*ny)
       
        train_x_list.append(new_xstft)
       
      
    train_x = np.array(train_x_list)
    train_y = np.zeros(read_files)
    
     
    input_train = open('train.txt')
    for line in input_train:
        line_split = line.split(',')
        if int(line_split[0][:6]) - 100001 < read_files: 
            train_y[int(line_split[0][:6]) - 100001] = line_split[1]
        
    

    
    print('Started validation at: ', datetime.now())    

    
    # Validation
    
    
    test_path_start = ['', '20000', '2000', '200', '20']
    test_audios_paths = ['validation/validation/'  + test_path_start[len(str(x))]  + str(x) + ".wav" for x in range(1, 1001)]
    test_x_list = []
    
                
    for path in test_audios_paths:
       wave, sr = librosa.load(path)
       
       
       x = librosa.stft(wave, n_fft = 8192, win_length = 8192, center = False)
       xstft = librosa.amplitude_to_db(abs(x))
       
       nx, ny = xstft.shape
       new_xstft = xstft.reshape(nx*ny)
       
       test_x_list.append(new_xstft)
       
    
    test_x = np.array(test_x_list)
    test_y = np.zeros(1000)
    
    
    input_train = open('validation.txt')
    
    for line in input_train:
        line_split = line.split(',')
        test_y[int(line_split[0][:6]) - 200001] = line_split[1]    
    
    
    print('\n')
    
    
    test_label = np.zeros(1000)
    clf_number = 0
    
    
    for name, clf in zip(names, classifiers):
        clf_number += 1
        
        print('Started fitting ' + name + ' at: ', datetime.now())   
        clf.fit(train_x, train_y)
        print('Classfier ' + name + ' finished fitting at: ', datetime.now())
        test_now_y = clf.predict(test_x)
        
        for i in range(0, 1000):
            test_label[i] += test_now_y[i]
            
        print(name + ' classifier finished at: ', datetime.now(), '\n')
       
        
    print(np.sum(test_label))
    
    
    conf = np.array([[0,0],
                     [0,0],
                     [0,0],
                     [0,0],
                     [0,0],
                     [0,0]])
                    
    print('Classifier number:' , clf_number)
    
    score = 0.0
    for i in range(0, 1000):
        
        if test_y[i] == 1 :
            
            if test_label[i] * 2 >= clf_number:
                score += 0.001
                
            conf[int(test_label[i])][1] += 1
        else:
            
            if test_label[i] * 2 < clf_number:
                score += 0.001
                
            conf[int(test_label[i])][0] += 1
            
            
    print(conf)
    
            
    print('Finished fitting at: ', datetime.now(), ' with total score : ', score, ' and read files: ', read_files, ' with number of classifiers: ', clf_number)   
