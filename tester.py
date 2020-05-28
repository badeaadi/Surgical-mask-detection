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
     "AdaBoost", # 0.694
]

classifiers = [
    
     #LogisticRegression(),
     #SVC(kernel="linear", C=0.025),
     #DecisionTreeClassifier(max_depth=7),
     #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
     #MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
]


if __name__ == '__main__':

        
    # Training data, number of files read (can choose to read more or less)
    
    read_files = 8000
    
    # File format
    
    train_path_start = ['', '10000', '1000', '100', '10']
    train_audios_paths = ['train/train/' + train_path_start[len(str(x))]  + str(x) + ".wav" for x in range(1, read_files + 1)]
    
    # 
    
    train_x_list = []
    train_y = np.zeros(read_files)

    
    print('Started reading train data at: ', datetime.now())
                
    for path in train_audios_paths:
       
       aud, sr = librosa.load(path)
       
       # Extract wave and sampling ration from path with librosa.load
       
       # Used a bigger n_fft, but still strictly smaller than the sampling ratio,       
       # we reduced the number of features per audio in order to reduce overfitting 
             
       # Using short fourier transformation from librosa
       # https://librosa.github.io/librosa/generated/librosa.core.stft.html

                                                                                                                                           
       x = librosa.stft(aud, n_fft = 8192, win_length = 8192, center = False)       
       
       # https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
           
       
       xstft = librosa.amplitude_to_db(abs(x))
       
       # https://librosa.github.io/librosa/generated/librosa.core.amplitude_to_db.html       
       
       nx, ny = xstft.shape
       new_xstft = xstft.reshape(nx*ny)
       
       train_x_list.append(new_xstft)
       
      
    
    # Reading 0 or 1 from the train.txt comma separated text file
     
    input_train = open('train.txt')
    for line in input_train:  
        line_split = line.split(',')
        if int(line_split[0][:6]) - 100001 < read_files: 
            train_y[int(line_split[0][:6]) - 100001] = line_split[1]
    
    train_x = np.array(train_x_list)
    
    
    
    
    
    # Testing - now using the same functions as when reading train data
    
    print('Started testing at: ', datetime.now())    
    
    
    test_path_start = ['', '30000', '3000', '300', '30']
    test_audios_paths = ['test/test/'  + test_path_start[len(str(x))]  + str(x) + ".wav" for x in range(1, 3001)]
    test_x_list = []
    
                
    for path in test_audios_paths:
       aud, sr = librosa.load(path)
       
       # Very important to use the same parameters in order to scale 
       x = librosa.stft(aud, n_fft = 8192, win_length = 8192, center = False)
       xstft = librosa.amplitude_to_db(abs(x))
       
       # xtsft is the information after the short fourier transformation
       
       nx, ny = xstft.shape
       
       # nx, ny are the dimensions of the information which will be reshaped to a one-dimensional array
       
       new_xstft = xstft.reshape(nx*ny)
       
       
       # the new information is now appended to the list which will be tranformated into a numpy array later on
       test_x_list.append(new_xstft)
    
    test_x = np.array(test_x_list)
    
    test_label = np.zeros(3000)
    
    clf_number = 0
    
    
    
    # Choosing every model from sklearn from above
    
    
    for name, clf in zip(names, classifiers):
        
        # Counting the classifier numbers
        
        clf_number += 1
        
        print('Started fitting ' + name + ' at: ', datetime.now())    
        
        clf.fit(train_x, train_y)
        
        print('Classfier ' + name + ' finished fitting at: ', datetime.now())
        
        test_y = clf.predict(test_x)
        
        # Summed predictions go to test_labels(which is the sum of classifiers)
        for i in range(0, 3000):
            test_label[i] += test_y[i]
    
    
    out = open('sub.txt', 'w')  
    
    out.write('name,label\n')

    # For every testing file

    for i in range(0, 3000):
        
        # If more than a half of the classifiers called it as 1 then prediction becomes 1
        
        if test_label[i] * 2 >= clf_number:
            test_label[i] = 1 
        else: 
            test_label[i] = 0
        
        
    # Writing format in sub.txt

    for iterat in range(len(test_y)):
        aux = iterat
        out.write(str(iterat + 300001) + '.wav,' + str(int(test_label[iterat])) + '\n')
        
    out.close()
    

    print('Finished fitting at: ', datetime.now())    

   
    
   
        
    
    
