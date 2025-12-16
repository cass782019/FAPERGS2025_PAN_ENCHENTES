# README - Visualizações

## Visão Geral

Esta pasta contém todas as visualizações geradas pelo pipeline: gráficos SHAP para explicabilidade dos modelos e gráficos EDA para análise exploratória dos dados.

## Estrutura

```
10_visualizacoes/
├── README_VISUALIZACOES.md (este arquivo)
├── shap_plots/
│   ├── dataset_unificado_sobreviveu_pandemia_xgboost_importance_bar.png
│   ├── dataset_unificado_sobreviveu_pandemia_xgboost_importance_summary.png
│   ├── dataset_unificado_sobreviveu_enchente_xgboost_importance_bar.png
│   ├── dataset_unificado_sobreviveu_enchente_xgboost_importance_summary.png
│   └── comparacao_modelos.png
└── eda_plots/
    ├── grafico_*.png
    ├── dashboard_eda_*.png
    └── matriz_correlacao_*.png
```

## Visualizações SHAP

### 1. Importance Bar

**Arquivos:**
- `dataset_unificado_sobreviveu_pandemia_xgboost_importance_bar.png`
- `dataset_unificado_sobreviveu_enchente_xgboost_importance_bar.png`

**Descrição:** Gráfico de barras mostrando importância média absoluta de cada feature.

**Interpretação:**
- Eixo X: Importância média (|SHAP value|)
- Eixo Y: Features ordenadas por importância
- Barras maiores = Features mais importantes

**Uso:** Identificar rapidamente as top features mais impactantes.

### 2. Summary Plot

**Arquivos:**
- `dataset_unificado_sobreviveu_pandemia_xgboost_importance_summary.png`
- `dataset_unificado_sobreviveu_enchente_xgboost_importance_summary.png`

**Descrição:** Dot plot mostrando distribuição de valores SHAP por feature.

**Interpretação:**
- Eixo X: Valor SHAP (impacto na predição)
  - Direita (+): Aumenta probabilidade de sobrevivência
  - Esquerda (-): Diminui probabilidade de sobrevivência
- Eixo Y: Features ordenadas por importância
- Cor: Valor da feature (vermelho = alto, azul = baixo)
- Densidade: Distribuição dos impactos

**Uso:** Entender como e em que direção cada feature afeta as predições.

### 3. Comparação de Modelos

**Arquivo:** `comparacao_modelos.png`

**Descrição:** Comparação de performance (AUC-ROC e Average Precision) entre diferentes modelos.

**Interpretação:**
- Gráficos de barras lado a lado
- Permite comparar XGBoost, LightGBM, CatBoost, etc.

**Uso:** Validar escolha do XGBoost como melhor modelo.

## Visualizações EDA

### Gráficos Principais

**Padrões de Nomenclatura:**
- `grafico_*.png`: Gráficos específicos de distribuições
- `dashboard_eda_*.png`: Dashboards consolidados
- `matriz_correlacao_*.png`: Heatmaps de correlação

**Tipos Comuns:**
- Histogramas de idade e tempo
- Boxplots por porte
- Gráficos de barras por situação cadastral
- Séries temporais de aberturas/fechamentos
- Distribuições geográficas

### Como Visualizar

**No Notebook:**
```python
from IPython.display import Image
Image('10_visualizacoes/shap_plots/dataset_unificado_sobreviveu_pandemia_xgboost_importance_bar.png')
```

**Python Script:**
```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('10_visualizacoes/shap_plots/dataset_unificado_sobreviveu_pandemia_xgboost_importance_bar.png')
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## Insights das Visualizações

### Top 5 Features (SHAP)

1. **idade_empresa_anos** (28.45%)
   - Empresas antigas = mais sobrevivência
   - Threshold crítico: ~5 anos

2. **tempo_situacao_anos** (19.23%)
   - Estabilidade = fator protetor

3. **empresa_ativa** (15.67%)
   - Binário: Ativa = sobrevivência provável

4. **porte** (8.34%)
   - Maior porte = maior resiliência

5. **followers_count_mean** (6.21%)
   - Presença digital = fator positivo

Ver análise completa em: `../04_resultados/analise_shap.md`

## Geração de Visualizações

### SHAP Plots

Gerados automaticamente por `08_codigo/notebooks/4.3.ipynb`:

```python
import shap

# Summary plot (bar)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('10_visualizacoes/shap_plots/importance_bar.png', dpi=300, bbox_inches='tight')

# Summary plot (dot)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('10_visualizacoes/shap_plots/importance_summary.png', dpi=300, bbox_inches='tight')
```

### EDA Plots

Gerados por `08_codigo/notebooks/EDA_dados_unidos.ipynb`:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Exemplo: Distribuição de porte
plt.figure(figsize=(10, 6))
df['porte'].value_counts().plot(kind='bar')
plt.title('Distribuição de Porte das Empresas')
plt.xlabel('Porte')
plt.ylabel('Quantidade')
plt.savefig('10_visualizacoes/eda_plots/grafico_porte.png', dpi=300, bbox_inches='tight')
```

## Formato e Qualidade

### Especificações Técnicas

- **Formato:** PNG
- **Resolução:** 300 DPI (alta qualidade para publicação)
- **Tamanho:** Variável (geralmente 1200-2400 px largura)
- **Compressão:** Otimizada para balanço qualidade/tamanho

### Para Publicação

Visualizações estão prontas para:
- ✅ Artigos científicos
- ✅ Apresentações
- ✅ Relatórios técnicos
- ✅ Documentação

## Customização

### Regenerar com Diferentes Configurações

**Top N Features:**
```python
shap.summary_plot(shap_values, X_test, max_display=10)  # Top 10 ao invés de 20
```

**Tamanho da Figura:**
```python
plt.figure(figsize=(14, 10))  # Maior
shap.summary_plot(shap_values, X_test)
plt.savefig('custom_plot.png', dpi=300)
```

**Estilo:**
```python
plt.style.use('seaborn-v0_8')  # Ou 'ggplot', 'bmh', etc.
```

## Referências

- **Código SHAP:** `../08_codigo/notebooks/4.3.ipynb`
- **Código EDA:** `../08_codigo/notebooks/EDA_dados_unidos.ipynb`
- **Análise:** `../04_resultados/analise_shap.md`
- **Documentação SHAP:** https://shap.readthedocs.io/

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Qualidade:** 300 DPI, pronto para publicação

