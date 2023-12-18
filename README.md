# House Prices - Advanced Regression Techniques

## Barema da ponderada segue:

**Desenvolvimento e Submissão de Soluções para o Desafio House Kaggle**

*1. Compreensão do desafio House Kaggle, incluindo o contexto e os objetivos do desafio;*
*2. Implementação das soluções do desafio no notebook preparatório de forma que passem nos testes pré-setados;*
*3. Utilização de técnicas, algoritmos e bibliotecas para a resolução dos problemas propostos;*
*4. Submissão das soluções desenvolvidas ao Kaggle Open Challenge;*
*5. Descrição dos requisitos para o desenvolvimento das soluções e compreensão do desafio House Kaggle;*
*6. Explicação detalhada do desenvolvimento das soluções, avaliação, otimização e submissão ao Kaggle utilizando comentários no código;*

## Aluno: Vinícius Oliveira Fernandes
### Professor: Afonso Cesar Lelis Brandão

A atividade proposta no Desafio Kaggle Open Challenge consiste no entendimento e inferência do tipo de análise em relação aos padrões dentro de dados não consolidados.

## Entendimento do Conteúdo

Em específico o desafio "House Prices" (com fonte de dados relacionada as informações de casas) tem como objetivo a previsão dos preços de imóveis, indo além das características convencionais consideradas por compradores ao descrever a casa dos sonhos. O conjunto de dados é composto por 79 variáveis que abrangem praticamente todos os aspectos e configurações das residências em Ames, Iowa, e visa antecipar o preço final de cada casa. 

Neste contexto diferentemente das preferências comuns, como o número de quartos ou a presença de uma cerca branca, este conjunto de dados inclui elementos adicionais, como a altura do teto do porão ou a proximidade de uma ferrovia leste-oeste. A avaliação dos modelos submetidos é baseada no Root-Mean-Squared-Error (RMSE), calculado entre o logaritmo do valor previsto e o logaritmo do preço observado de vendas. Essa abordagem logarítmica visa equalizar os impactos de erros na previsão de casas caras e baratas.

Segue estruturas de pastas do projeto são: 

- Pasta **data** com os conjuntos de dados, contendo os "arquivos houses_test_raw.csv" e "houses_train_raw.csv" os dados brutos de teste e treinamento, respectivamente.
- Pasta **houses_trainer_package:** com scripts e arquivos para treinamento do modelo. Segue lista de arquivos que estão dentro da pasta:

    - **__init__.py:** Arquivo para inicializar o pacote.
    - **pipeline.py:** Pré-processamento e modelagem.
    - **preprocessors.py:** Transformações específicas dos dados.
    - **trainer.py:** Arquivo de treinamento do modelo.
    - **.ipynb_checkpoints:** Pasta gerada automaticamente para armazenar estados salvos dos notebooks Jupyter.
    - **tests:** Contém scripts de teste e arquivos pickle associados à avaliação do modelo e validação dos resultados.
    - **.gitignore:** Arquivo utilizado para especificar quais arquivos ou tipos de arquivo devem ser ignorados pelo Git.
    - **autotest.sh:** Script shell usado para automatizar testes.

## Funcionalidades e Estruturação dos arquivos

Alguns dos arquivos contém funções que possuem uma importância significante para tratamento dos dados e estruturação de todo o pipeline. Uma das funções mais importantes para o modelo são as funções de pré-processo, contidas no arquivo "preprocessors" com funções específicas, direcionadas para o pré-processamento dos dados. 

- **Definição de funções e sua funcionalidade:** As funções do arquivo realizam transformações específicas para variáveis ordinais, numéricas e nominais, preparando os dados para o treinamento do modelo. O uso do OrdinalEncoder transforma variáveis ordinais em números, respeitando a ordem especificada. O pipeline resultante inclui imputação de valores faltantes, codificação ordinal e aplicação da escala Min-Max. A função create_preproc_numerical concentra-se em variáveis numéricas, utilizando um pipeline que realiza imputação de valores faltantes usando KNNImputer e aplica a escala Min-Max. A função create_preproc_nominal lida com variáveis nominais, empregando um pipeline que realiza imputação de valores faltantes usando a estratégia "most_frequent" e aplica codificação one-hot com OneHotEncoder. Esses pré-processadores são posteriormente integrados ao pipeline principal, conforme definido no arquivo pipeline.

O arquivo pipeline é fundamental para o pré-processamento e modelagem de dados e necessário para a realização da análise de dados. 

- **Definição de funções e sua funcionalidade:** A função create_preproc desempenha um papel na definição de etapas de pré-processamento, utilizando pré-processadores para variáveis ordinais, numéricas e nominais. Já a função create_model é responsável pela criação de modelos de regressão, incluindo Gradient Boosting Regressor, Ridge Regression, Support Vector Machine Regressor e AdaBoost Regressor com Decision Tree como estimador base. Por fim, a função de pré-processamento é unificada ao modelo em um único pipeline usando a função make_pipeline. Esse pipeline completo facilita o treinamento e a avaliação de modelos em conjuntos de dados específicos, fornecendo uma abordagem organizada e eficiente para lidar com tarefas de aprendizado de máquina.

Já o arquivo "trainer", há a definição da classe "Trainer", responsável pelo treinamento e avaliação de um modelo de regressão.

- **Definição de funções e sua funcionalidade:** Aqui destaca-se a realização da validação cruzada para avaliar o desempenho do modelo, calculando a raiz quadrada do erro médio quadrático (RMSLE) como métrica de avaliação. O método imprime a média e o desvio padrão do RMSLE para as dobras de validação cruzada. A instância é criada, os dados brutos são carregados, o pipeline é construído aplicando o pré-processamento e a modelagem, e, por fim, é realizada a validação cruzada para avaliar o desempenho do modelo.

Os testes fornecidos na classe TestSubmissionBaseline visam avaliar diferentes aspectos da submissão de previsões de preços de casas no contexto de um desafio Kaggle. A biblioteca nbresult é utilizada para comparar os resultados obtidos com os resultados esperados, armazenados em arquivos pickle. Nos testes é incluído a verificação da pontuação do modelo, do formato da submissão, das colunas na submissão e dos tipos de dados das colunas na submissão.

Os conjuntos de dados associados ao desafio incluem train.csv (conjunto de treinamento), test.csv (conjunto de teste), data_description.txt (descrição completa de cada coluna, originalmente preparada por Dean De Cock), e sample_submission.csv (uma submissão de referência a partir de uma regressão linear sobre o ano e mês da venda, área do lote em pés quadrados e número de quartos). Os campos dessas bases de dados abrangem uma variedade de características, incluindo informações sobre a construção, zoneamento, tamanho do lote, tipo de acesso, configuração do lote, localização, qualidade e condição geral, ano de construção, entre outros. A variável alvo é SalePrice, representando o preço de venda da propriedade em dólares.

## Algoritmos de Treinamento

O código no arquivo "houses_kaggle_competition.ipynb" oferece oito métodos de treinamento, cada um com sua breve descrição:

### Árvore de Decisão:
Pode ser definido num algoritmo que toma decisões com base em regras hierárquicas:

- **Parâmetros importantes:** max_depth (profundidade máxima) e min_samples_leaf (número mínimo de amostras em uma folha).

### KNN (K-Nearest Neighbors):
Algoritmo que é utilizado em problemas de classificação, como neste caso em específico, basicamente define pontos com base na proximidade com os "vizinhos" mais próximos.

### Ridge (Regressão Linear com Regularização L2):
Extensão da regressão linear com termo de regularização L2. Sensível à escala dos recursos e útil para evitar overfitting.

### SVM (Support Vector Machine):
Algoritmo parecido com a definição classificatório do knn, ele busca uma linha de separação entre duas classes distintas analisando os dois pontos, um de cada grupo, mais próximos da outra classe.

### Random Forest:
Algoritmo que combina vários modelos de machine learning em um único resultado. Ele utiliza um conjunto de "árvores de decisão" treinadas de forma independentemente. Esse modo pode reduzir o overfitting e capturar relações não lineares.

### Boosted Trees (AdaBoost e Gradient Boosting):
Algoritmos que utiliza a replicação de modelos "fracos" como método para consolidar suas previsões.

### Stacking:
Algoritmo que combina previsões de vários modelos usando um meta-modelo. Pode melhorar o desempenho, mas pode ser considerado exaustivo computacionalmente.

### XGBoost:
Implementação otimizada de Gradient Boosting com eficiência computacional e tratamento de dados ausentes.

## Diferenças na seleção de algoritmos e particularidades (feito com os modelos mais consolidados)

O conteúdo de aprendizado de treinamento é modelado a partir de alguns algoritmos. Todos eles possuem métodos de treinamento diferentes, e foram selecionados tendo isso em mente. As principais diferenças de treinamento entre esses métodos são respectivamente: a Natureza do Aprendizado, a Configuração de Hiperparâmetros, a Complexidade de Treinamento e a Interpretabilidade:

### Natureza do Aprendizado

- Os algoritmos **Árvores de Decisão**, **KNN**, **Ridge**, **SVM** aprendem de forma independente. Já o **Random Forest** e Boosted Trees realizam coleções sequenciais de modelos fracos. O **Stacking** combina modelos individuais usando um meta-modelo e o **XGBoost** utiliza árvores de decisão como modelos base.

### Configuração de Hiperparâmetros:

- Cada modelo tem sua própria definição de hiperparâmetros "básicos", principalmente os algoritmos de Árvores de Decisão, KNN, Ridge e SVM. No entanto, o Random Forest e Boosted Trees têm hiperparâmetros dos modelos base e do ensemble. E o outro tipo de utilização diferente é dos algoritmos Stacking e XGBoost que têm hiperparâmetros para modelos base e meta-modelo.

### Complexidade de Treinamento:

- A complexidade é velocidade dos algoritmos é outro fator determinante. Os modelos individuais (Árvores de Decisão, KNN, Ridge, SVM) são relativamente simples e rápidos. Já o Random Forest e Boosted Trees são mais lentos devido à construção sequencial. O Stacking requer treinamento de vários modelos individuais e do meta-modelo, o que gera uma maior complexidade em sua construção. Por fim, o XGBoost também possui processsamento lento devido ao uso de otimização gradiente.

### Interpretabilidade:

- Os algoritmos Árvores de Decisão, KNN, Ridge, SVM são fáceis de interpretar individualmente. O Random Forest e Boosted Trees são mais desafiadores devido à natureza combinada das previsões.
A Interpretabilidade do Stacking depende do meta-modelo escolhido e o XGBoost é o mais difícil de interpretar, sendo um método de boosting.

**Otimização e Aprimoramento do Modelo**

Ao explorar e entender os algoritmos e suas implementações podemos especular alguns ajustes para otimizar o desempenho, a interpretabilidade e generalização da análise. Primeiramente, faremos esta análise para cada um dos algoritmos utilizados:

**Árvores de Decisão:**
- Como possibilidade para aumento da performance e interpretabilidade nas Árvores de Decisão, é possível aplicar a técnica de seleção de atributos 'SelectKBest' e otimização de hiperparâmetros usando GridSearchCV. O objetivo é reduzir a dimensionalidade dos dados e encontrar os melhores valores para 'max_depth' e 'min_samples_leaf'.

**KNN:**
Incrementação do "BallTree" e a técnica de seleção de atributos 'SelectKBest' para aprimorar a performance e interpretabilidade do modelo.

**Ridge:**
No caso do modelo Ridge, uso do algoritmo de otimização L-BFGS e incrementação da técnica de regularização para evitar o overfitting. A busca pelos melhores hiperparâmetros, especialmente o coeficiente de regularização 'alpha', é conduzida utilizando GridSearchCV.

**SVM:**
Utilizar um kernel "Polynomial" e técnicas de regularização, para aprimorar o modelo e para melhorar sua performance. A busca por hiperparâmetros, incluindo 'C', 'epsilon', e 'degree', é realizada através do GridSearchCV.

**Random Forest:**
Para Random Forest, é possível aumentar o número de árvores para melhorar a precisão e utilizar RandomizedSearchCV para encontrar os melhores valores de hiperparâmetros, como 'n_estimators', 'max_depth' e 'min_samples_leaf'.

**Boosted Trees:**
No contexto de Boosted Trees, é possível ajustar a taxa de aprendizagem e o número de árvores. O uso do GridSearchCV facilita a busca pelos melhores valores dos hiperparâmetros para aprimorar o desempenho do modelo.

**Stacking:**
Ao implementar o Stacking, utilizar um meta-modelo com regressão linear e um número maior de modelos base. A otimização dos hiperparâmetros pode também ser realizada com RandomizedSearchCV para encontrar configurações ideais.

**XGBoost:**
Para o XGBoost, utilizar um número maior de árvores, uma taxa de aprendizagem menor e o uso de early stopping para evitar o overfitting. A busca por hiperparâmetros, especialmente 'max_depth', é conduzida através do GridSearchCV.

**Num contexto Geral para toda a análise:**
- **Otimização de Hiperparâmetros:** Em todos os casos, a busca pelos melhores hiperparâmetros para ajustar os modelos.
- **Feature Engineering:** A exploração de novas características para melhorar a capacidade preditiva do modelo e compreensão das áreas em que os modelos estão falhando para ajustes adicionais.
- **Tuning:** Ajuste dos pesos dos modelos no ensemble para otimizar o desempenho geral.
- **Cross-Validation:** Aumentar o número de folds na validação cruzada pode fornecer uma avaliação mais estável do desempenho do modelo.
- **Tratamento de Outliers:** Tratamento de outliers para evitar distorções nos resultados.
- **Experimentação com Outros Modelos:** Exploração diferentes algoritmos para determinar o mais adequado ao seu conjunto de dados (utilizar um "pycaret" da vida por exemplo).

Essas estratégias combinadas podem aprimorar a performance do modelo.
