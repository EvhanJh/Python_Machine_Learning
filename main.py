import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Projet : Prédiction de la qualité du vin #
# Théo CAPITAINE & Evhan JOSEPH

WineQT = pd.read_csv('WineQT.csv')
WineQT = WineQT.drop(['Id'], axis=1)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Phase d'analyse : Visualisation des données #

# Tout afficher :
# print(WineQT.describe(include='all'))

# Voir les Index :
# print(WineQT.columns)

# Voir toutes les informations (types, values non-null, colonnes) :
# print(WineQT.info())

# Pourcentage des données manquantes (on peut voir qu'il ne manque aucune donnée) :
# print(100*WineQT.isnull().sum()/WineQT.shape[0])

# Cardinalité pour visualiser les données intéressantes à garder :
# print(WineQT.nunique().sort_values())

#Regrouper les donnes par qualité de vin
grpbyquality = WineQT.groupby("quality").mean()
print(grpbyquality)

#graphique des données regroupé
grpbyquality.plot(kind="bar",figsize=(20,10)).set_title("Moyenne de chaque critère en fonction de la qualité")

#HYPOTHESES DE DEPART :
# On peut sembler observer que :
# la qualité des vins baissent lorsque :
#   -l'acidité volatile augmentent
#   -les chlorides augmentent
#   -l'acidité citric baissent

#  la qualité des vins montent lorsque :
#   -l'alcool monte
#   -les sulphates montent
#
# le dioxide semble influencé la qualité du vin seulement quand celui ci est très faible
# le sucre/sulfur dioxide/densité/acidité fixé ne semblent pas avoir d'impact sur la qualité


# Graphique influence des sulphates sur la qualité
plt.figure(figsize=(15,7))
sns.lineplot(data=WineQT, x="quality",y="alcohol").set_title("Influence de l'alcool sur la qualité")


# Le graphique montre bien la corrélation sulfate sur la qualité
plt.figure(figsize=(15,7))
sns.boxplot(x=WineQT.quality,y=WineQT['sulphates']).set_title("Influence des sulfates sur la qualité")
#Ce graphique montre que le sulfate a une influence sur la qualité du vin jusqu'à une certaine quantité après laquelle la qualité du vin ne semble plus être affectée

# Le graphique montre bien la corrélation de l'acidité volatile sur la qualité
plt.figure(figsize=(15,7))
sns.boxplot(x=WineQT.quality,y=WineQT['volatile acidity']).set_title("Influence de l'acidité volatile sur la qualité")

# ----------------------------------------------------- #

# Prétraitements à prévoir avec scikit-learn : #

# Variables catégorielle  : quality
# Variables numériques : tout le reste sauf Id

Y = WineQT.loc[:,['quality']]
X = WineQT.loc[:,[c for c in WineQT.columns if (c != 'quality')]]

# print(X.head())
# print(Y.head())

# Matrice de corrélation

corr = WineQT.corr()
matplotlib.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr, annot=True,annot_kws={"fontsize":15}).set_title('Matrice de corrélation')
plt.show()

# Top 5 corrélation

print("-----------------------------------")
print("Top 5 corrélation")
corrMatrixSort = corr.sort_values(by="quality", ascending=False)['quality']
print(corrMatrixSort.head(5))
print("-----------------------------------")

# Standardisation des données

X_features = X
X = StandardScaler().fit_transform(X)

# Split the data train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Model predictif : Random Forest

model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)


# Facteur influent sur la qualité du vin avec le model 1

feat_importances = pd.Series(model.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10)).set_title('Facteur important dans l\'influence de la qualité du vin avec le model Random Forest')
plt.show()

# Model predictif 2 : XGBoost

model2 = xgb.XGBClassifier(random_state=1)
model2.fit(X_train, y_train.values.ravel())
y_pred2 = model2.predict(X_test)

print("Accuracy on model 1 : ", accuracy_score(y_test, y_pred))

print("Accuracy on model 2 : ", accuracy_score(y_test, y_pred2))

# Facteur influent sur la qualité du vin avec le model 2

feat_importances = pd.Series(model2.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10)).set_title('Facteur important dans l\'influence de la qualité du vin avec le model XGBClassifier')
plt.show()

# Conclusion #

# Les observations faites au début de l'étude se valident suite à l'entrainement à l'aide des deux modèles choisis.
# Le modèle XGBoost semble moins précis que le modèle Random Forest L'alcool et les sulfates sont les facteurs les
# plus important dans le fondement d'un vin de qualité. Mais pas que ! L'acidité volatile est très importante et très
# technique car on peut voir qu'au delà d'une certaine valeur, le vin deviens de moins en moins bonne qualité.

# Merci,

# Théo CAPITAINE & Evhan JOSEPH
