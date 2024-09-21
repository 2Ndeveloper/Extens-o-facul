import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns 

# Pré-processamento
data['data'] = pd.to_datetime(data['data'])
data['dia_da_semana'] = data['data'].dt.dayofweek
data['mes'] = data['data'].dt.month

# Lidar com valores ausentes
data.fillna(method='ffill', inplace=True)

# Análise Exploratória
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='data', y='vendas')
plt.title('Vendas de Gelo ao Longo do Tempo')
plt.show()

# Definindo variáveis independentes e dependentes
X = data[['temperatura', 'umidade', 'dia_da_semana', 'mes']]
y = data['vendas']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelagem Preditiva
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsão
y_pred = modelo.predict(X_test)

# Avaliação do Modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualização das Previsões
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Vendas Reais')
plt.plot(y_pred, label='Vendas Previstas')
plt.title('Comparação entre Vendas Reais e Previstas')
plt.legend()
plt.show()