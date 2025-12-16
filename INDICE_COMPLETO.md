# Índice Completo da Documentação

## Estrutura da Pasta 0_artigo/

### 📄 Documentos Principais

- **README.md** - Documento principal em formato de artigo científico completo (1000+ linhas)
- __COMO_CONVERTER_PARA_DOCX.md__ - Guia para converter README para formato DOCX
- __INDICE_COMPLETO.md__ - Este arquivo (índice navegável)

### 📁 01_metodologia/ (6 documentos)

Documentação detalhada de cada etapa do pipeline:

1. __1.1_limpeza_dados.md__ - Etapa 0: Limpeza e validação de CNPJs
2. __1.2_agregacao_features.md__ - Etapa 1: Agregação de múltiplas fontes
3. __1.3_feature_engineering.md__ - Etapa 2: Criação de targets e features derivadas
4. __1.4_eda_analise_exploratoria.md__ - Etapa 3: Análise exploratória de dados
5. __1.5_pipeline_ml_base.md__ - Etapa 4: Processamento e combinação de datasets
6. __1.6_otimizacao_shap.md__ - Etapa 5: Otimização com Optuna e análise SHAP

### 📁 02_decisoes_projeto/ (4 documentos)

Justificativas de decisões arquiteturais e de modelagem:

1. __decisoes_arquiteturais.md__ - Chunks, estratégia híbrida, pipeline modular
2. __escolha_algoritmos.md__ - Por que gradient boosting e XGBoost
3. __estrategias_modelagem.md__ - Unified, separate e hybrid
4. __metricas_avaliacao.md__ - AUC-ROC e Average Precision

### 📁 03_limitacoes/ (3 documentos)

Documentação honesta de limitações e trabalhos futuros:

1. __limitacoes_dados.md__ - Viés de seleção, cobertura temporal, qualidade
2. __limitacoes_metodologicas.md__ - Causalidade, generalização, fatores ausentes
3. __trabalhos_futuros.md__ - Curto, médio e longo prazo

### 📁 04_resultados/ (4 documentos)

Resultados detalhados e análises:

1. __metricas_modelos.md__ - Performance, hiperparâmetros ótimos, AUC 0.9998
2. __analise_shap.md__ - Top 20 features, importâncias, insights
3. __comparacao_performance.md__ - Pandemia vs enchentes, análise por porte/setor
4. **graficos/** - Pasta com cópias de visualizações importantes

### 📁 05_diagramas/ (4 documentos mermaid)

Diagramas explicativos do pipeline:

1. __fluxo_pipeline.md__ - Fluxo completo end-to-end
2. __arquitetura_sistema.md__ - Componentes e interações
3. __feature_engineering.md__ - Criação de features e targets
4. __decisoes_modelagem.md__ - Estratégias e otimização

### 📁 06_dados/ (Datasets + README)

Todos os datasets necessários para reprodução:

__README_DADOS.md__ - Documentação completa de todos os datasets

**processados/** (5 arquivos CSV):

- `6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv` (2.6M linhas, dataset principal)
- `7_dados_unidos.csv` (posts Instagram)
- `dataset_unificado.csv` (2.6M linhas, 43 features)
- `dataset_com_posts.csv` (2.6K linhas)
- `dataset_sem_posts.csv` (2.6M linhas)

**amostras/**:

- `amostra_estabelecimentos_rs.csv` (10k linhas, referência)

### 📁 07_modelos/ (Modelos + README)

Modelos treinados prontos para uso:

__README_MODELOS.md__ - Como usar os modelos, fazer predições

**Modelos (.joblib)**:

- `best_dataset_unificado_sobreviveu_pandemia_xgboost.joblib` (AUC 0.9998)
- `best_dataset_unificado_sobreviveu_enchente_xgboost.joblib` (AUC 0.9998)

### 📁 08_codigo/ (Código completo + README)

Todo código necessário para reprodução:

__README_CODIGO.md__ - Guia completo de uso dos notebooks e scripts
__requirements.txt__ - Dependências com versões exatas

**notebooks/** (6 arquivos):

- `0.0.1_limpeza.ipynb` - Limpeza de dados
- `0.2.3_juntar_dados.ipynb` - Agregação
- `3.1.ipynb` - Feature engineering
- `EDA_dados_unidos.ipynb` - Análise exploratória
- `4.1.ipynb` - Pipeline ML base
- `4.3.ipynb` - Otimização e SHAP

**scripts/**:

- `4.3_optuna_shap.py` - Versão script do 4.3.ipynb

### 📁 09_anexos/ (4 documentos)

Material de apoio e referências:

1. **glossario.md** - Glossário completo de termos técnicos
2. __referencias_bibliograficas.md__ - Referências completas (ABNT, APA, BibTeX)
3. __como_reproduzir.md__ - Guia passo a passo de reprodução
4. __requisitos_tecnicos.md__ - Hardware, software, compatibilidade

### 📁 10_visualizacoes/ (Gráficos + README)

Todas as visualizações geradas:

__README_VISUALIZACOES.md__ - Documentação das visualizações

__shap_plots/__ (5+ arquivos):

- Importance bar (pandemia e enchentes)
- Importance summary (pandemia e enchentes)
- Comparação de modelos

__eda_plots/__ (múltiplos arquivos):

- Gráficos de distribuição
- Dashboards EDA
- Matrizes de correlação

## Navegação Rápida

### Para Começar

→ __README.md__ (documento principal)
→ __09_anexos/como_reproduzir.md__

### Para Entender Metodologia

→ __01_metodologia/__ (ler em ordem 1.1 a 1.6)
→ __05_diagramas/fluxo_pipeline.md__

### Para Usar Modelos

→ __07_modelos/README_MODELOS.md__
→ __06_dados/README_DADOS.md__

### Para Reproduzir

→ __08_codigo/README_CODIGO.md__
→ __08_codigo/requirements.txt__
→ __09_anexos/como_reproduzir.md__

### Para Entender Resultados

→ __04_resultados/__ (todos os arquivos)
→ __10_visualizacoes/shap_plots/__

### Para Publicação Científica

→ __README.md__
→ __COMO_CONVERTER_PARA_DOCX.md__
→ __09_anexos/referencias_bibliograficas.md__

## Estatísticas do Projeto

### Documentação

- **README principal:** ~1000 linhas, formato artigo científico
- **Documentos metodologia:** 6 arquivos detalhados
- **Documentos técnicos:** 20+ arquivos markdown
- **Total de documentação:** ~5000+ linhas

### Código

- **Notebooks:** 6 arquivos .ipynb
- **Scripts:** 1 arquivo .py
- **Linhas de código:** ~3000+
- **Dependências:** 15+ bibliotecas principais

### Dados

- **Datasets:** 5 arquivos CSV processados
- **Registros:** 2.685.868 empresas
- **Features:** 40+ por empresa
- **Tamanho total:** ~1 GB

### Modelos

- **Modelos treinados:** 2 (.joblib)
- **Performance:** AUC-ROC 0.9998
- **Algoritmo:** XGBoost otimizado com Optuna

### Visualizações

- **Gráficos SHAP:** 5+ arquivos
- **Gráficos EDA:** Múltiplos
- **Resolução:** 300 DPI (qualidade publicação)

## Checklist de Completude

### ✅ Estrutura

- [x] Pasta 0_artigo criada
- [x] Todas subpastas criadas (10 pastas)
- [x] Estrutura hierárquica clara

### ✅ Dados

- [x] Datasets processados copiados
- [x] Amostra de dados brutos incluída
- [x] README_DADOS.md criado

### ✅ Código

- [x] Notebooks copiados
- [x] Scripts copiados
- [x] requirements.txt criado
- [x] README_CODIGO.md criado

### ✅ Modelos

- [x] Modelos .joblib copiados
- [x] README_MODELOS.md criado

### ✅ Visualizações

- [x] Gráficos SHAP copiados
- [x] Gráficos EDA copiados
- [x] README_VISUALIZACOES.md criado

### ✅ Documentação

- [x] README.md principal completo
- [x] 6 documentos de metodologia
- [x] 4 documentos de decisões
- [x] 3 documentos de limitações
- [x] 3 documentos de resultados
- [x] 4 diagramas mermaid
- [x] 4 documentos de anexos
- [x] Glossário completo
- [x] Referências bibliográficas
- [x] Guia de reprodução

### ⏳ Pendente (Opcional)

- [ ] Conversão para DOCX (usar guia em COMO_CONVERTER_PARA_DOCX.md)
- [ ] Adicionar imagens dos diagramas mermaid (opcional)
- [ ] Revisão ortográfica completa (opcional)

## Como Navegar

1. **Começar pelo README.md** - Visão completa do projeto
2. **Explorar metodologia/** - Entender cada etapa
3. **Verificar resultados/** - Ver performance e análises
4. **Usar código/** - Reproduzir trabalho
5. **Ler anexos/** - Material de apoio

## Tempo de Leitura Estimado

- **README principal:** 60-90 min
- **Metodologia completa:** 2-3 horas
- **Documentação técnica:** 1-2 horas
- **Total para entendimento completo:** ~5-7 horas

## Contato e Suporte

Para dúvidas sobre:

- __Dados:__ Consultar `06_dados/README_DADOS.md`
- __Modelos:__ Consultar `07_modelos/README_MODELOS.md`
- __Código:__ Consultar `08_codigo/README_CODIGO.md`
- __Reprodução:__ Consultar `09_anexos/como_reproduzir.md`

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Status:** Documentação completa e pronta para uso

