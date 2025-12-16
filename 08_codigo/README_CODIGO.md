# README - Código-Fonte

## Visão Geral

Esta pasta contém todo o código necessário para reproduzir o pipeline de Machine Learning, desde a limpeza de dados até a otimização de modelos.

## Estrutura

```
08_codigo/
├── README_CODIGO.md (este arquivo)
├── requirements.txt (dependências)
├── notebooks/
│   ├── 0.0.1_limpeza.ipynb
│   ├── 0.2.3_juntar_dados.ipynb
│   ├── 3.1.ipynb
│   ├── EDA_dados_unidos.ipynb
│   ├── 4.1.ipynb
│   └── 4.3.ipynb
└── scripts/
    └── 4.3_optuna_shap.py
```

## Notebooks

### 1. 0.0.1_limpeza.ipynb

**Etapa:** 0 - Limpeza e Validação  
**Documentação:** `../01_metodologia/1.1_limpeza_dados.md`

**Objetivo:** Limpar dados brutos da Receita Federal

**Entrada:** `estabelecimentos_rs.csv` (dados brutos)  
**Saída:** `1_estabelecimentos_rs_sem_duplicados.csv`

**Tempo de execução:** ~5-10 minutos

**Operações:**
- Validação de CNPJs (8 dígitos)
- Remoção de duplicatas
- Segregação de inválidos

**Como executar:**
```python
# No Jupyter/JupyterLab
# Abrir notebook e executar células sequencialmente
# Cell → Run All
```

### 2. 0.2.3_juntar_dados.ipynb

**Etapa:** 1 - Agregação  
**Documentação:** `../01_metodologia/1.2_agregacao_features.md`

**Objetivo:** Combinar dados de múltiplas fontes

**Entrada:** Dados limpos da Etapa 0  
**Saída:** Dataset agregado

**Tempo:** ~10-15 minutos

### 3. 3.1.ipynb

**Etapa:** 2 - Feature Engineering  
**Documentação:** `../01_metodologia/1.3_feature_engineering.md`

**Objetivo:** Criar features de sobrevivência e derivadas

**Entrada:** Dados agregados  
**Saída:** `06_dados/processados/6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv`

**Tempo:** ~15-20 minutos

**Features criadas:**
- `sobreviveu_pandemia` (target)
- `sobreviveu_enchente` (target)
- `idade_empresa_anos`
- `tempo_situacao_anos`
- Indicadores binários
- Encoding de categóricas

### 4. EDA_dados_unidos.ipynb

**Etapa:** 3 - Análise Exploratória  
**Documentação:** `../01_metodologia/1.4_eda_analise_exploratoria.md`

**Objetivo:** Análise exploratória e visualizações

**Entrada:** Dataset com targets  
**Saída:** Estatísticas, gráficos em `../10_visualizacoes/eda_plots/`

**Tempo:** ~20-30 minutos

**Análises:**
- Distribuições
- Correlações
- Valores ausentes
- Padrões de sobrevivência

### 5. 4.1.ipynb

**Etapa:** 4 - Pipeline ML Base  
**Documentação:** `../01_metodologia/1.5_pipeline_ml_base.md`

**Objetivo:** Processar posts e combinar datasets

**Entrada:**
- `6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv`
- `7_dados_unidos.csv`

**Saída:**
- `../06_dados/processados/dataset_unificado.csv`
- `../06_dados/processados/dataset_com_posts.csv`
- `../06_dados/processados/dataset_sem_posts.csv`

**Tempo:** ~30-60 minutos (dependendo do tamanho)

**Operações:**
- Agregação de features de posts
- Merge de datasets
- Estratégias (unified/separate/hybrid)
- Pré-processamento

### 6. 4.3.ipynb

**Etapa:** 5 - Otimização e SHAP  
**Documentação:** `../01_metodologia/1.6_otimizacao_shap.md`

**Objetivo:** Otimizar modelos e analisar explicabilidade

**Entrada:** `dataset_unificado.csv`  
**Saída:**
- `../07_modelos/best_*.joblib`
- `../10_visualizacoes/shap_plots/`

**Tempo:** ~2-4 horas (50 trials por modelo)

**Operações:**
- Teste de 5 algoritmos
- Otimização com Optuna
- Seleção do melhor modelo
- Análise SHAP
- Salvamento de modelos

## Scripts

### 4.3_optuna_shap.py

**Descrição:** Versão script do notebook 4.3.ipynb

**Uso:** Execução em linha de comando ou servidor

```bash
python scripts/4.3_optuna_shap.py
```

**Vantagens sobre notebook:**
- Executável em servidor sem interface gráfica
- Logs para arquivo
- Mais fácil de automatizar
- Pode rodar em background

**Configurações no código:**
```python
# Editar no início do arquivo
N_TRIALS = 100  # Número de trials Optuna
MODELOS = ['xgboost', 'lightgbm']  # Algoritmos a testar
```

## Instalação

### 1. Criar Ambiente Virtual

**Com virtualenv:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

**Com conda:**
```bash
conda create -n ml_empresas python=3.11
conda activate ml_empresas
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

Ver: `requirements.txt` para lista completa

### 3. Verificar Instalação

```python
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import optuna
import shap

print(f"Pandas: {pd.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"Optuna: {optuna.__version__}")
print(f"SHAP: {shap.__version__}")
```

## Ordem de Execução

### Reprodução Completa

Execute notebooks nesta ordem:

1. **0.0.1_limpeza.ipynb** → Limpar dados brutos
2. **0.2.3_juntar_dados.ipynb** → Agregar fontes
3. **3.1.ipynb** → Criar features e targets
4. **EDA_dados_unidos.ipynb** → Análise exploratória (opcional)
5. **4.1.ipynb** → Processar e combinar datasets
6. **4.3.ipynb** ou `4.3_optuna_shap.py` → Otimizar modelos

**Tempo total:** ~4-6 horas (dependendo de N_TRIALS)

### Apenas Modelagem (Dados Prontos)

Se os dados já estão processados:

1. **4.3.ipynb** → Otimização e SHAP
   - Certifique-se que `dataset_unificado.csv` existe

**Tempo:** ~2-4 horas

## Configurações Importantes

### Memória e Performance

**Processamento em Chunks:**
```python
CHUNK_SIZE = 10000  # Reduzir se memória insuficiente
```

**Optuna Trials:**
```python
N_TRIALS = 50  # Padrão: 50-100
# Mais trials = melhor otimização, mas mais lento
# 20 trials: teste rápido (~30min)
# 50 trials: balanceado (~1-2h)
# 200 trials: otimização completa (~6-8h)
```

**SHAP Sample Size:**
```python
SHAP_SAMPLE = 100  # Reduzir se muito lento
# 100: rápido, análise representativa
# 1000: mais detalhado, mais lento
```

### Seeds e Reprodutibilidade

Todos os notebooks usam:
```python
RANDOM_STATE = 42
np.random.seed(42)
```

Isso garante que executar múltiplas vezes produz resultados idênticos.

## Troubleshooting

### Erro: ImportError

**Problema:** Biblioteca não instalada

**Solução:**
```bash
pip install nome_da_biblioteca
```

### Erro: MemoryError

**Problema:** Dataset muito grande para RAM

**Soluções:**
1. Reduzir `CHUNK_SIZE`
2. Usar amostra menor (para testes)
3. Aumentar RAM ou usar servidor

### Erro: Optuna trials muito lentos

**Problema:** Otimização demora muito

**Soluções:**
1. Reduzir `N_TRIALS` (ex: 20)
2. Testar menos modelos
3. Usar servidor com mais CPUs

### Warning: SHAP lento

**Problema:** SHAP demora muito para calcular

**Soluções:**
1. Reduzir `SHAP_SAMPLE` (ex: 50)
2. Usar apenas dataset de teste (não todo treino)
3. Considerar GPU (se disponível)

## Boas Práticas

### 1. Salvar Progresso

Notebooks salvam automaticamente outputs intermediários:
- Sempre salvar após cada etapa
- Manter backups de datasets processados

### 2. Logs

Capture logs para debugging:
```python
import logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 3. Versionamento

Use git para versionar notebooks:
```bash
git add notebooks/4.3.ipynb
git commit -m "Otimização com 100 trials"
```

**Nota:** `.gitignore` deve incluir:
```
*.csv
*.joblib
__pycache__/
```

### 4. Documentação

Adicione markdown cells aos notebooks:
- Explicar cada seção
- Resultados esperados
- Tempo estimado

## Performance Esperada

### Hardware Recomendado

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB |
| GPU | Não necessário | Útil para SHAP |

### Tempos de Execução

| Etapa | Tempo Mínimo | Tempo Esperado |
|-------|--------------|----------------|
| 0-2 | ~30 min | ~1 hora |
| 3 (EDA) | ~20 min | ~30 min |
| 4 (Pipeline) | ~30 min | ~1 hora |
| 5 (Otimização) | ~1 hora | ~3 horas |
| **Total** | **~2.5 horas** | **~5.5 horas** |

## Referências

- **Documentação Metodologia:** `../01_metodologia/`
- **Datasets:** `../06_dados/README_DADOS.md`
- **Modelos:** `../07_modelos/README_MODELOS.md`
- **Guia de Reprodução:** `../09_anexos/como_reproduzir.md`

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Compatibilidade:** Python 3.9+, testado em 3.11

