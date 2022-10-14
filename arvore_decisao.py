import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from tabulate import tabulate
from imblearn.over_sampling import SMOTE

# Preparando a Base de Dados
df1 = pd.read_excel("C:/Users/marin/Downloads/base_de_dados.xlsx", sheet_name = 1)
df2 = pd.read_excel("C:/Users/marin/Downloads/base_de_dados.xlsx", sheet_name = 3)
df1_df2 = pd.merge(df1,df2[['SUBJECTID','sstat']],on='SUBJECTID', how='left')
df = df1_df2[df1_df2.sstat != 9] # Data Frame não irá conter os dados com o status = "Lost" (Perdido)
df_clean = df.dropna() # Limpando dados "Nulos"
df_round = df_clean.round() # Arredondando os resultados

print("\n Base de Dados:\n")
print(tabulate((df_round.iloc[:,[0,2,3,4,5,6,7,10,11,16]]), headers = ['SUBJECTID', 'age', 'race_id', 'ERpos', 'PgRpos',
'HR Pos', 'Her2MostPos', 'BilateralCa', 'Laterality', 'sstat']))

# Unindo as diferentes abas do Excel para haver uma única tabela
dfx = df_round.iloc[:,[0,2,4,5,6,7,10,11]]
dfy = df_round.iloc[:,[0,16]]
dfbom = pd.merge(dfx, dfy, on="SUBJECTID")

# Separação das variáveis que serão utilizadas para cada coordenada do gráfico
x = dfbom.iloc[:,[1,2,3,4,5,7]] # age, ERpos, PgRpos, HR Pos, Her2MostPos, Laterality 
y = dfbom.iloc[:,8] # sstat (variável utilizada como "rótulo")

# Balanceando os Dados - Oversampling
smote = SMOTE(random_state = 42)
x_smote, y_smote = smote.fit_resample(x,y)

# Treinando os dados, agora balanceados
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.3, random_state = 42) # 70% treino e 30% teste

tree = DecisionTreeClassifier(criterion="gini", 
                                max_depth = 4)

# Árvore com os dados de treinamento
tree = tree.fit(x_train,y_train)

# Predicão de acordo com o treinamento
y_predict = tree.predict(x_test)

# Avaliação de Desempenho
print("\n Avaliação de Desempenho \n")
print("Matriz de Confusão:\n",metrics.confusion_matrix(y_test, y_predict), "\n")
print("Erro: " + str(1 - metrics.accuracy_score(y_test, y_predict)))
print("Acurácia:",metrics.accuracy_score(y_test, y_predict))
print("Precisão - Vivos:",metrics.precision_score(y_test, y_predict, pos_label = 7))
print("Precisão - Mortos:",metrics.precision_score(y_test, y_predict, pos_label = 8))
print("Recall - Vivos:",metrics.recall_score(y_test, y_predict, pos_label = 7))
print("Recall - Mortos:",metrics.recall_score(y_test, y_predict, pos_label = 8))

# Construção do gráfico
dot_file = export_graphviz(tree,
                    out_file="resultado_final",
                    filled= True,
                    rounded=True,
                    special_characters=True,
                    feature_names=x.columns,
                    class_names=["Alive", "Dead"])

# Seleção de Variáveis que demonstraram maior importância no treinamento (merecem maior atenção)
x_selectBest = dfbom.iloc[:,[2,3,4,5,7]]
y_selectBest = dfbom.iloc[:,8]

best_features = SelectKBest(score_func=chi2, k='all') #chi2 = distribuição qui-quadrado
fit = best_features.fit(x_selectBest, y_selectBest)
df_score = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x_selectBest.columns)

feature_score = pd.concat([df_columns,df_score], axis=1)
print("\n Variáveis que demonstraram maior importância no treinamento do algoritmo \n")
feature_score.columns = ["Fator", "Valor"]
print(feature_score.nlargest(7, "Valor"))
print("\n")





