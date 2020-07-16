# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:46:58 2020

@author: avales
"""
"""
Trying to guess the winner only by using the following columns : 


Winner Loser  to fit the column Winner  
    
"""
import pandas as pd 
import random
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os 
from time import time    
import glob
import itertools

#https://github.com/edouardthom/ATPBetting/blob/master/Python/main.py
#https://www.kaggle.com/edouardthomas/beat-the-bookmakers-with-machine-learning-tennis

#recupéation des données dans les fichiers excel 
path = "data/*.xlsx"
fullDataSet = []
for fileName in glob.glob(path) : 
    fullDataSet.append(pd.read_excel(fileName))
data = pd.concat(fullDataSet)

#récupération du gagne de chaque match
targetData = data['Winner'].values.tolist()


#On récuprère les joueurs de chaque match et on mélange les colonnes
#afin que de ne plus avoir le gagna et le perdant mais deux joueurs
extractedPlayers = data[['Winner','Loser']]
matchPlayers = extractedPlayers.values.tolist()
for match in matchPlayers:
    random.shuffle(match) 

#TODO 
"""
ajouter les deux colonnes de matchPlayers aux autres données
"""
#cleanedData = data[['Tournament','Series','Court','Surface','Round','Winner','Loser','WRank','LRank']]


#TRANSFORMATION DES DONNEES 
#Transformation des joueurs
allNames = []
allNames.append(data['Winner'].values.tolist())
allNames.append(data['Loser'].values.tolist())
TousLesJouerus = list(itertools.chain.from_iterable(allNames))

mylist = list(set(TousLesJouerus))

numberedTargetData = [mylist.index(joueur) for joueur in targetData]
NumberredListOfPalyer = listOfPlayers
for i, match in enumerate(listOfPlayers):
    for j, joueur in enumerate(match) : 
        NumberredListOfPalyer[i][j] = mylist.index(joueur)



X_train, X_test, y_train, y_test = train_test_split(NumberredListOfPalyer, numberedTargetData, train_size=0.75, test_size=0.25, random_state=42)
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=-1)

t0 = time()
tpot.fit(np.array(X_train), np.array(y_train))

print(time() - t0)
print(tpot.score(np.array(X_test), np.array(y_test)))
#tpot.export('tpot_digits_pipeline.py')











