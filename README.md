# Exercícios do Módulo de Operacionalização de Machine Learning

## Instalar o ambiente

No Anaconda Prompt, ir para a diretoria deste projeto e executar os comandos

* `conda env create -f conda.yaml`

Isto irá criar um novo ambiente (`rumos-class-oml`) com as depedências necessárias para correr este projecto. Caso já tenham o ambiente feito na aula não é preciso fazer um novo, basta utilizar esse.

Caso queiram criar um ambiente com um outro nome basta adicionar ao 1º comando acima `-n <nome>` (substitudo "<nome>" pelo nome que querem dar)

## Notebooks

Os notebooks do projeto estão presentes na diretoria `notebooks`.

Dentro desta diretoria temos 2 outras diretorias, `logistic_regression` e `random_forest`, e dentro dessas diretorias temos os seguintes notebooks:

* `09_{model}.ipynb`: Notebook original desenvolvido no módulo de Fundamentos de Machine Learnig.
* `10_{model}_mlflow.ipynb`: Notebook em que é introduzido o mlflow e como dar track a uma experiência utilizando o mlflow.
* `11_{model}_mlflow_register_model.ipynb`: Notebook em que é introduzido o conceito de "Model Registry" do mlflow
* `12_{model}_mlflow_read_registered_model.ipynb`: Notebook em que carregamos e usamos o modelo registado no notebook 11.
* `13_{model}_mlflow_register_model_pipeline.ipynb`: Similar ao 11, mas em que registamos uma pipeline (que inclui o scaler + o model) em vez dos componentes separados.
* `14_{model}_mlflow_read_registered_model_pipeline.ipynb`: Notebook em que carregamos e usamos a pipeline registada no notebook 13.

Em que `{model}` é substituido por `logistic_regression` e `random_forest` nas suas respetivas diretorias.

Por ultimo, existe uma 3a diretoria chamada de `serve` que contém o seguinte notebook:

* `test_requests.ipynb`: Notebook auxiliar para testar o mlflow serve e a fastapi app com o modelo. As respetivas secções de `mlflow serve` e `fastapi` só podem ser corridas quando o `mlflow serve` e `fastapi` estiverem a correr respetivamente.

## App

Para expor-mos o nosso modelo registado numa API podemos utilizar ou a funcionalidade de models serve do mlflow ou a framework FastAPI.

### mlflow models serve

Para expor uma api utilizando a funcionalidade de serving do mlflow basta correr, no Anaconda Prompt, com o ambiente deste projeto ativo e na raiz do projeto (pasta rumos), os seguintes comandos

```
set MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db
mlflow models serve -m models:/{model_name}/{model_version}
```

**Nota:** Caso utilizem MacOS ou Linux, em vez de `set MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db` devem de utilizar `export MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db`

Onde o `{model_name}` e o `{model_version}` devem ser substituidos pelo nome com que o modelo foi registado e a versão do mesmo a ser utilizada, respetivamente. Por exemplo, para utilizarmos a versão 2 do modelo random_forest os comandos devem ser 

```
set MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db
mlflow models serve -m models:/random_forest/2
```

**Nota:** Caso utilizem MacOS ou Linux, em vez de `set MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db` devem de utilizar `export MLFLOW_TRACKING_URI=sqlite:///./mlruns/mlflow.db`

Este funcionalidade de mlflow utiliza a especificação de ambiente criada automaticamente pelo mlflow para o modelo, para criar um ambiente de conda para o modelo e servir o modelo isolado nesse ambiente virtual de conda.

Esta API expõe o endpoint `/invocations` na qual espera receber as features de input do modelo, e retorna a previsão dada pelo modelo. Para testar a API basta correr o notebook `test_requests.ipynb` na secção de `mlflow serve`.

### FastAPI

No python script `src\app\main.py` foi desenvolvida uma applicação simples com a fastapi.

O nome e a versão do modelo registado a ser utilizado na app tem que ser especificado no ficheiro de configuração presente na diretoria `config` no fichiro `app.json`. 

Esta app expõe o endpoint `/predict` na qual espera receber as features de input do modelo (em formato json, no body do pedido) e retorna a previsão dada pelo modelo.

Para correr a app: no Anaconda Prompt, com o ambiente deste projeto activo, na raiz do projeto (pasta rumos) executar o comando em baixo

```
uvicorn src.app.main:app
```

Para testar se o modelo ficou corretamente exposto na app podemos correr a secção `FastAPI` do notebook `test_requests.ipynb` ou então podemos utilizar o pagina html presente na diretoria `frontend` deste projeto e realizar um pedido (e ver a resposta) através desse frontend.

## Tests

Para testarmos o nosso modelo registado utilizou se a framework de Python `pytest`.

Os testes estão presentes no script de Python `tests\random_forest\test_rf_out.py`:
* `test_model_out`: testa o output do modelo, e verifica se coincide com com o output esperado
* `test_model_out_shape`: test se a shape do output do modelo coincide com a shape esperada

Para se correr estes testes: no Anaconda Prompt, com o ambiente deste projeto activo, na raiz do projeto (pasta rumos) executar o comando em baixo

```
pytest tests
```
