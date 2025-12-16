# README - Modelos Treinados

## Visão Geral

Esta pasta contém os modelos finais otimizados do pipeline de Machine Learning. Os modelos foram treinados com XGBoost e otimizados com Optuna para predizer sobrevivência empresarial.

## Modelos Disponíveis

### 1. best_dataset_unificado_sobreviveu_pandemia_xgboost.joblib

**Descrição:** Modelo para predizer sobrevivência durante a pandemia

**Performance:**
- **AUC-ROC:** 0.9998
- **Average Precision:** 0.9997
- **F1-Score:** 0.9864

**Target:** `sobreviveu_pandemia` (1 = sobreviveu, 0 = não sobreviveu)

**Período:** Março 2020 - Fevereiro 2022

### 2. best_dataset_unificado_sobreviveu_enchente_xgboost.joblib

**Descrição:** Modelo para predizer sobrevivência durante enchentes

**Performance:**
- **AUC-ROC:** 0.9998
- **Average Precision:** 0.9996
- **F1-Score:** 0.9867

**Target:** `sobreviveu_enchente` (1 = sobreviveu, 0 = não sobreviveu)

**Período:** Maio 2024 - Dezembro 2024

## Conteúdo dos Modelos

Cada arquivo `.joblib` contém um dicionário com:

```python
{
    'model': XGBClassifier,          # Modelo treinado
    'scaler': StandardScaler,         # Normalizador de features
    'imputer': SimpleImputer,         # Imputador de missing values
    'feature_names': list,            # Lista de nomes das features
    'auc': float,                     # AUC-ROC no conjunto de teste
    'ap': float,                      # Average Precision
    'importancia_features': DataFrame # Importância SHAP das features
}
```

## Como Usar

### 1. Carregar Modelo

```python
import joblib
import pandas as pd

# Carregar modelo
modelo_info = joblib.load('07_modelos/best_dataset_unificado_sobreviveu_pandemia_xgboost.joblib')

# Extrair componentes
modelo = modelo_info['model']
scaler = modelo_info['scaler']
imputer = modelo_info['imputer']
feature_names = modelo_info['feature_names']

print(f"AUC-ROC: {modelo_info['auc']:.4f}")
print(f"Features: {len(feature_names)}")
```

### 2. Preparar Dados para Predição

```python
# Carregar novos dados
df_novos = pd.read_csv('dados_novos.csv')

# Selecionar features (mesma ordem do treino)
X_novos = df_novos[feature_names]

# Imputar missing values
X_imputados = imputer.transform(X_novos)

# Normalizar
X_scaled = scaler.transform(X_imputados)
```

### 3. Fazer Predições

```python
# Predição de classes (0 ou 1)
predicoes_classe = modelo.predict(X_scaled)

# Predição de probabilidades
predicoes_proba = modelo.predict_proba(X_scaled)

# Probabilidade da classe positiva (sobrevivência)
prob_sobrevivencia = predicoes_proba[:, 1]

# Adicionar ao DataFrame
df_novos['predicao'] = predicoes_classe
df_novos['prob_sobrevivencia'] = prob_sobrevivencia
```

### 4. Interpretar Resultados

```python
# Empresas com alta probabilidade de sobreviver
sobreviventes_provaveis = df_novos[df_novos['prob_sobrevivencia'] > 0.7]
print(f"Empresas provavelmente sobreviventes: {len(sobreviventes_provaveis)}")

# Empresas em risco
em_risco = df_novos[df_novos['prob_sobrevivencia'] < 0.3]
print(f"Empresas em risco: {len(em_risco)}")

# Empresas incertas
incertas = df_novos[(df_novos['prob_sobrevivencia'] >= 0.3) & 
                     (df_novos['prob_sobrevivencia'] <= 0.7)]
print(f"Empresas incertas: {len(incertas)}")
```

## Features Requeridas

### Features Obrigatórias (40+ no total)

**Dados de Empresas:**
- `idade_empresa_anos`: Idade da empresa em anos
- `tempo_situacao_anos`: Tempo na situação cadastral atual
- `porte_encoded`: Porte codificado (0-4)
- `empresa_ativa`: Indicador binário (1/0)
- `empresa_baixada`: Indicador binário (1/0)
- `empresa_suspensa`: Indicador binário (1/0)
- `cnae_fiscal_principal`: Código CNAE (ou encoded)
- `municipio_codigo`: Código do município
- `cep_3_digitos`: 3 primeiros dígitos do CEP

**Dados de Posts (agregados):**
- `followers_count_mean`, `_max`, `_min`, `_std`
- `like_count_sum`, `_mean`, `_median`, `_std`, `_max`
- `engagement_rate_mean`, `_median`, `_std`, `_max`
- `caption_length_mean`, `_median`, `_std`, `_max`
- `caption_words_mean`, `_median`, `_std`, `_max`
- `total_posts`: Número total de posts

**Nota:** Se empresa não tem posts, features de posts serão imputadas pelo `imputer`.

## Requisitos de Dados

### Dados Mínimos para Predição

Para fazer predições, você precisa de:

1. **CNPJ ou identificador** da empresa
2. **Data de abertura** (para calcular idade)
3. **Porte** da empresa
4. **Situação cadastral** atual
5. **Data da situação** atual
6. **(Opcional) Dados de posts** do Instagram

### Calcular Features Derivadas

```python
from datetime import datetime
import pandas as pd

def preparar_dados_predicao(df):
    """Prepara dados para predição"""
    
    # Data de referência (hoje)
    hoje = pd.Timestamp.now()
    
    # Idade da empresa
    df['data_inicio_atividade'] = pd.to_datetime(df['data_inicio_atividade'])
    df['idade_empresa_anos'] = (hoje - df['data_inicio_atividade']).dt.days / 365.25
    
    # Tempo na situação
    df['data_situacao_cadastral'] = pd.to_datetime(df['data_situacao_cadastral'])
    df['tempo_situacao_anos'] = (hoje - df['data_situacao_cadastral']).dt.days / 365.25
    
    # Indicadores binários
    df['empresa_ativa'] = (df['situacao_cadastral'] == 'ATIVA').astype(int)
    df['empresa_baixada'] = (df['situacao_cadastral'] == 'BAIXADA').astype(int)
    df['empresa_suspensa'] = (df['situacao_cadastral'] == 'SUSPENSA').astype(int)
    
    # Encoding de porte
    porte_map = {'MEI': 0, 'MICRO': 1, 'PEQUENA': 2, 'MEDIA': 3, 'GRANDE': 4}
    df['porte_encoded'] = df['porte'].map(porte_map)
    
    return df
```

## Exemplo Completo

```python
import joblib
import pandas as pd
from datetime import datetime

# 1. Carregar modelo
modelo_info = joblib.load('07_modelos/best_dataset_unificado_sobreviveu_pandemia_xgboost.joblib')
modelo = modelo_info['model']
scaler = modelo_info['scaler']
imputer = modelo_info['imputer']
feature_names = modelo_info['feature_names']

# 2. Carregar dados novos
df = pd.read_csv('../06_dados/processados/6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv')

# 3. Preparar features (exemplo: primeiras 100 empresas)
df_teste = df.head(100).copy()

# 4. Selecionar features
X_teste = df_teste[feature_names]

# 5. Pré-processar
X_imputado = imputer.transform(X_teste)
X_scaled = scaler.transform(X_imputado)

# 6. Predizer
predicoes = modelo.predict_proba(X_scaled)[:, 1]

# 7. Analisar
df_teste['prob_sobrevivencia'] = predicoes
print("\nTop 10 empresas mais resilientes:")
print(df_teste.nlargest(10, 'prob_sobrevivencia')[['cnpj_basico', 'porte', 
                                                      'idade_empresa_anos', 
                                                      'prob_sobrevivencia']])

print("\nTop 10 empresas mais vulneráveis:")
print(df_teste.nsmallest(10, 'prob_sobrevivencia')[['cnpj_basico', 'porte', 
                                                       'idade_empresa_anos', 
                                                       'prob_sobrevivencia']])
```

## Features Mais Importantes

### Top 5 (SHAP)

1. **idade_empresa_anos** (28.45%) - Quanto mais antiga, maior sobrevivência
2. **tempo_situacao_anos** (19.23%) - Estabilidade é protetora
3. **empresa_ativa** (15.67%) - Empresas ativas sobrevivem mais
4. **porte** (8.34%) - Maior porte, maior resiliência
5. **followers_count_mean** (6.21%) - Presença digital relevante

Ver análise completa em: `../04_resultados/analise_shap.md`

## Limitações

### 1. Generalização Temporal

- Modelos treinados em 2020-2024
- Performance em eventos futuros pode variar
- Recomenda-se retreinar periodicamente

### 2. Generalização Geográfica

- Específico para Rio Grande do Sul
- Aplicação em outros estados requer validação

### 3. Dados de Input

- Requer features calculadas corretamente
- Missing values serão imputados (pode afetar precisão)
- Features de posts ausentes são comuns (99.9% das empresas)

## Troubleshooting

### Erro: "Feature names mismatch"

**Problema:** Features fornecidas não correspondem ao treino

**Solução:** Verificar ordem e nomes:
```python
print(f"Features esperadas: {feature_names}")
print(f"Features fornecidas: {list(X_novos.columns)}")
```

### Erro: "ValueError: X has Y features, expecting Z"

**Problema:** Número incorreto de features

**Solução:** Garantir que X tem exatamente 40+ features na ordem correta

### Warning: "X does not have valid feature names"

**Problema:** X é numpy array sem nomes de colunas

**Solução:** Usar pandas DataFrame:
```python
X_novos = pd.DataFrame(X_array, columns=feature_names)
```

## Manutenção dos Modelos

### Quando Retreinar?

- **Anualmente:** Atualizar com dados mais recentes
- **Após eventos críticos:** Incorporar novos padrões
- **Degradação de performance:** Se AUC cai significativamente

### Como Retreinar?

Execute o pipeline completo:
1. Etapa 4: `08_codigo/notebooks/4.1.ipynb`
2. Etapa 5: `08_codigo/notebooks/4.3.ipynb`

## Referências

- **Código de treinamento:** `../08_codigo/notebooks/4.3.ipynb`
- **Documentação:** `../01_metodologia/1.6_otimizacao_shap.md`
- **Resultados:** `../04_resultados/metricas_modelos.md`

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Performance:** AUC-ROC ~0.9998 (quasi-perfeito)

