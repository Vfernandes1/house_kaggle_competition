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
