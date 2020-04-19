## BIBLIOTECAS

#pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#pip install pandas
import pandas as pd
#pip install decision-tree-id3
from id3 import Id3Estimator, export_text #, export_graphviz

## MINERANDO OS DADOS E SEPARANDO AS VARIÁVEIS DEPENDENTES E INDEPENDENTES

#carregando o dataset iris
iris = load_iris()

#informações sobre o dataset
print(load_iris.__doc__)

#colunas / features
dados = pd.DataFrame(iris.data, columns=iris.feature_names)

#seleciona apenas os dados das features / variáveis independentes
X = iris.data

#objetivo / target
objetivo = pd.DataFrame(iris.target, columns=['target'])

#seleciona apenas os dados target / variável dependente
Y = iris.target

#nomes dos targets
objetivo_nomes = pd.DataFrame(iris.target_names, columns=['target_names'])

## CONSTRUÇÃO DO MODELO BASEADO EM ÁRVORE DE DECISÃO (ID3)

#separação do conjunto de treino e teste usando 80 treino/20 teste e usando estado randômico a cada execução
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.2, random_state=None)

#formato dos dados de treino e teste
print(X_treino.shape, Y_treino.shape)
print(X_teste.shape, Y_teste.shape)

#criação do modelo de árvore ID3
modelo = Id3Estimator()

#treinamento do modelo
modelo_treinado = modelo.fit(X_treino, Y_treino, check_input=True)

#predicoes (objetivo)
predicoes = modelo_treinado.predict(X_teste)

## AVALIAÇÃO DO MODELO CRIADO

#matriz de confusão
matriz_confusao = confusion_matrix(Y_teste, predicoes)

#relatorio de classificacao
relat_classificacao = classification_report(Y_teste, predicoes)

#acuracia
acuracia = accuracy_score(Y_teste, predicoes)

## IMPRESSÃO DA ÁRVORE CRIADA

print(export_text(modelo.tree_, iris.feature_names))
