# README - Datasets

## Visão Geral

Esta pasta contém todos os datasets necessários para reproduzir o pipeline de Machine Learning completo. Os dados estão organizados em **processados** (prontos para uso) e **amostras** (referência).

## Estrutura

```
06_dados/
├── README_DADOS.md (este arquivo)
├── processados/
│   ├── 6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv
│   ├── 7_dados_unidos.csv
│   ├── dataset_unificado.csv
│   ├── dataset_com_posts.csv
│   └── dataset_sem_posts.csv
└── amostras/
    └── amostra_estabelecimentos_rs.csv
```

## Datasets Processados

### 1. 6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv

**Descrição:** Dataset principal com dados de empresas do RS e targets de sobrevivência.

**Uso:** Entrada principal para o pipeline ML (Etapa 4)

**Tamanho:** ~2.6M linhas × 20 colunas

**Colunas Principais:**
- `cnpj_basico`: Identificador único (8 dígitos)
- `porte`: MEI, MICRO, PEQUENA, MEDIA, GRANDE
- `situacao_cadastral`: ATIVA, BAIXADA, SUSPENSA, etc.
- `cnae_fiscal_principal`: Código de atividade econômica
- `municipio_codigo`: Código IBGE do município
- `data_inicio_atividade`: Data de abertura
- `idade_empresa_anos`: Idade em anos
- `tempo_situacao_anos`: Tempo na situação atual
- `empresa_ativa`: Indicador binário (1/0)
- `sobreviveu_pandemia`: **TARGET** (1 = sobreviveu, 0 = não)
- `sobreviveu_enchente`: **TARGET** (1 = sobreviveu, 0 = não)

**Estatísticas:**
- Empresas ativas: ~55-65%
- Taxa sobrevivência pandemia: ~35.76%
- Taxa sobrevivência enchentes: Variável por região

### 2. 7_dados_unidos.csv

**Descrição:** Dados de posts do Instagram de empresas do RS.

**Uso:** Entrada para agregação de features de presença digital

**Tamanho:** Variável (posts de 2.638 empresas)

**Colunas Principais:**
- `cnpj`: CNPJ completo da empresa
- `followers_count`: Número de seguidores
- `media_count`: Número de mídias
- `like_count`: Curtidas no post
- `caption`: Texto da legenda
- `timestamp`: Data/hora da publicação
- `profile_picture_url`: URL da foto de perfil

**Nota:** Apenas 0.098% das empresas têm posts no Instagram.

### 3. dataset_unificado.csv

**Descrição:** Dataset final combinando dados de empresas + features de posts agregadas.

**Uso:** Entrada para otimização de modelos (Etapa 5)

**Tamanho:** 2.685.868 linhas × 43 colunas

**Características:**
- Todas as empresas incluídas
- Features de posts imputadas (NaN → mean) para empresas sem Instagram
- Pronto para modelagem

**Features Agregadas de Posts:**
- `followers_count_mean`, `_max`, `_min`, `_std`
- `like_count_sum`, `_mean`, `_median`, `_std`, `_max`
- `engagement_rate_mean`, `_median`, `_std`, `_max`
- `caption_length_mean`, `_median`, `_std`, `_max`
- `caption_words_mean`, `_median`, `_std`, `_max`
- `total_posts`: Contagem de posts por empresa

### 4. dataset_com_posts.csv

**Descrição:** Subset com apenas empresas que têm posts no Instagram.

**Uso:** Análise específica de empresas digitais, estratégia separate

**Tamanho:** 2.638 linhas × 43 colunas

**Características:**
- Features de posts completas (sem imputação)
- Permite análise de padrões específicos de empresas digitais

### 5. dataset_sem_posts.csv

**Descrição:** Subset com empresas sem presença digital.

**Uso:** Modelagem específica para empresas sem Instagram, estratégia separate

**Tamanho:** 2.683.230 linhas × 16 colunas

**Características:**
- Apenas features de empresas (sem posts)
- Maioria das empresas do RS

## Amostras

### amostra_estabelecimentos_rs.csv

**Descrição:** Amostra de 10.000 linhas dos dados brutos da Receita Federal.

**Uso:** Referência para estrutura dos dados originais, não usado diretamente no pipeline.

**Tamanho:** 10.001 linhas (header + 10k registros)

## Como Usar

### Carregar Datasets

```python
import pandas as pd

# Dataset principal
df_empresas = pd.read_csv('06_dados/processados/6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv')

# Dados de posts
df_posts = pd.read_csv('06_dados/processados/7_dados_unidos.csv')

# Dataset unificado (para modelagem)
df_unificado = pd.read_csv('06_dados/processados/dataset_unificado.csv', low_memory=False)
```

### Verificar Integridade

```python
# Verificar CNPJs únicos
assert not df_unificado['cnpj_basico'].duplicated().any()

# Verificar targets
print(f"Sobreviveu pandemia: {df_unificado['sobreviveu_pandemia'].value_counts()}")
print(f"Sobreviveu enchentes: {df_unificado['sobreviveu_enchente'].value_counts()}")

# Verificar missing values
print(df_unificado.isnull().sum())
```

## Detalhes Técnicos

### Formato
- **Encoding:** UTF-8
- **Separador:** Vírgula (,)
- **Decimal:** Ponto (.)
- **Datas:** ISO format (YYYY-MM-DD) quando aplicável

### Tipos de Dados

**Numéricos:**
- `int64`: CNPJs, contadores, indicadores binários
- `float64`: Médias, desvios, taxas

**Categóricos:**
- `object`: Porte, situação, CNAE, município

**Temporais:**
- `object`: Datas (converter com `pd.to_datetime()`)

### Tamanhos

| Arquivo | Linhas | Colunas | Tamanho (MB) |
|---------|--------|---------|--------------|
| 6_empresas_rs... | 2.685.868 | 20 | ~250 |
| 7_dados_unidos | Variável | 15+ | ~50 |
| dataset_unificado | 2.685.868 | 43 | ~600 |
| dataset_com_posts | 2.638 | 43 | ~1 |
| dataset_sem_posts | 2.683.230 | 16 | ~200 |

## Estatísticas Resumidas

### Distribuição de Porte

| Porte | Proporção |
|-------|-----------|
| MEI | ~65% |
| MICRO | ~25% |
| PEQUENA | ~8% |
| MEDIA | ~1.5% |
| GRANDE | ~0.5% |

### Targets (dataset_unificado)

**sobreviveu_pandemia:**
- 1 (sobreviveu): ~35.76%
- 0 (não sobreviveu): ~64.24%

**sobreviveu_enchente:**
- Distribuição variável por região

## Notas Importantes

### 1. Desbalanceamento

- **Presença digital:** 2.638 vs. 2.683.230 (ratio 1:1018)
- **Target pandemia:** ~35.76% sobreviventes (ratio 1:1.8)

Este desbalanceamento é real e reflete a realidade do mercado.

### 2. Missing Values

- Features de posts estão ausentes para 99.9% das empresas
- Imputação com `SimpleImputer(strategy='mean')` aplicada em dataset_unificado
- Datasets separate evitam imputação artificial

### 3. Reprodutibilidade

- Datasets gerados com seed fixo (42)
- Mesmos dados produzem mesmos resultados
- Ordem de linhas pode variar em algumas operações

## Troubleshooting

### Erro: "low_memory=False" warning

**Problema:** Pandas pode emitir warning sobre tipos mistos

**Solução:** Use `low_memory=False` ao carregar CSVs grandes:
```python
df = pd.read_csv('arquivo.csv', low_memory=False)
```

### Erro: Memória insuficiente

**Problema:** Dataset muito grande para RAM

**Solução:** Carregar em chunks:
```python
chunks = []
for chunk in pd.read_csv('arquivo.csv', chunksize=10000):
    chunks.append(chunk)
df = pd.concat(chunks)
```

### Erro: Encoding issues

**Problema:** Caracteres especiais

**Solução:** Especificar encoding:
```python
df = pd.read_csv('arquivo.csv', encoding='utf-8')
```

## Referências

- **Fonte Receita Federal:** [Dados Abertos](https://www.gov.br/receitafederal/dados-abertos)
- **Documentação Pipeline:** `../README.md`
- **Feature Engineering:** `../01_metodologia/1.3_feature_engineering.md`

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Atualização:** Datasets autocontidos para reprodutibilidade completa

