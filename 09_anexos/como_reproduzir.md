# Como Reproduzir Este Trabalho

## Pré-requisitos

### Hardware Mínimo
- **CPU:** 4 cores
- **RAM:** 16 GB
- **Storage:** 10 GB livres
- **SO:** Windows 10+, Ubuntu 20.04+, macOS 10.15+

### Software
- **Python:** 3.9 ou superior (recomendado: 3.11)
- **Git:** Para clonar repositório (opcional)
- **Jupyter:** Para executar notebooks

## Passo a Passo Completo

### 1. Preparar Ambiente

#### Opção A: Conda (Recomendado)
```bash
conda create -n ml_empresas python=3.11
conda activate ml_empresas
cd 0_artigo/08_codigo
pip install -r requirements.txt
```

#### Opção B: Virtualenv
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
cd 0_artigo/08_codigo
pip install -r requirements.txt
```

### 2. Verificar Instalação

```python
python -c "import pandas, xgboost, optuna, shap; print('✅ Todas bibliotecas OK')"
```

### 3. Estrutura de Dados

Certifique-se que tem:
```
0_artigo/
├── 06_dados/
│   └── processados/
│       ├── 6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv ✅
│       ├── 7_dados_unidos.csv ✅
│       └── (outros datasets serão gerados)
└── 08_codigo/
    └── notebooks/ (executar notebooks aqui)
```

### 4. Executar Pipeline (Ordem)

#### Passo 4.1: Apenas Modelagem (Dados Prontos)

Se você já tem os dados processados:

```bash
cd 0_artigo/08_codigo/notebooks
jupyter notebook
# Abrir e executar: 4.1.ipynb
# Depois: 4.3.ipynb
```

**Tempo estimado:** 2-4 horas

#### Passo 4.2: Pipeline Completo (Do Zero)

Se quiser reproduzir desde a limpeza:

```bash
cd 0_artigo/08_codigo/notebooks
jupyter notebook
```

Executar notebooks nesta ordem:
1. `0.0.1_limpeza.ipynb` (~10 min)
2. `0.2.3_juntar_dados.ipynb` (~15 min)
3. `3.1.ipynb` (~20 min)
4. `EDA_dados_unidos.ipynb` (~30 min, opcional)
5. `4.1.ipynb` (~1 hora)
6. `4.3.ipynb` (~2-4 horas)

**Tempo total:** 4-6 horas

### 5. Resultados Esperados

Após execução completa, você terá:

**Dados:**
- `06_dados/processados/dataset_unificado.csv`
- `06_dados/processados/dataset_com_posts.csv`
- `06_dados/processados/dataset_sem_posts.csv`

**Modelos:**
- `07_modelos/best_dataset_unificado_sobreviveu_pandemia_xgboost.joblib`
- `07_modelos/best_dataset_unificado_sobreviveu_enchente_xgboost.joblib`

**Visualizações:**
- `10_visualizacoes/shap_plots/*.png`

**Métricas Esperadas:**
- AUC-ROC: ~0.9998
- Average Precision: ~0.9996-0.9997
- F1-Score: ~0.9864-0.9867

### 6. Verificar Reprodutibilidade

```python
import joblib

# Carregar modelo
modelo_info = joblib.load('07_modelos/best_dataset_unificado_sobreviveu_pandemia_xgboost.joblib')

# Verificar AUC
print(f"AUC-ROC: {modelo_info['auc']:.4f}")
# Deve imprimir: AUC-ROC: 0.9998 (ou muito próximo)

# Verificar features
print(f"Número de features: {len(modelo_info['feature_names'])}")
# Deve imprimir: Número de features: 40 (ou próximo)
```

### 7. Testes Rápidos

Para testar rapidamente (sem esperar horas):

**Reduzir trials do Optuna:**

Em `4.3.ipynb`, alterar:
```python
N_TRIALS = 20  # Ao invés de 50-100
```

**Usar amostra menor:**

```python
# No início dos notebooks
SAMPLE_SIZE = 100000  # Ao invés de dataset completo
df = df.sample(SAMPLE_SIZE, random_state=42)
```

## Troubleshooting Comum

### Erro: Memória Insuficiente

**Sintoma:** `MemoryError` ou kernel crash

**Solução:**
```python
# Reduzir chunk size
CHUNK_SIZE = 5000  # Ao invés de 10000

# Ou usar amostra
df = df.sample(500000, random_state=42)
```

### Erro: Optuna Muito Lento

**Sintoma:** Trials demorando > 5 minutos cada

**Solução:**
```python
# Reduzir trials
N_TRIALS = 20

# Testar menos modelos
MODELOS = ['xgboost']  # Ao invés de todos
```

### Erro: SHAP Muito Lento

**Sintoma:** Análise SHAP não termina

**Solução:**
```python
# Reduzir amostra
SHAP_SAMPLE = 50  # Ao invés de 100-1000
```

### Erro: Biblioteca Não Encontrada

**Sintoma:** `ImportError: No module named 'XXX'`

**Solução:**
```bash
pip install XXX
# Ou reinstalar requirements
pip install -r requirements.txt --upgrade
```

## Seeds e Reprodutibilidade

Todos os notebooks usam **seed fixo (42)**:
```python
RANDOM_STATE = 42
np.random.seed(42)
```

Isso garante:
- ✅ Mesmos splits train/test
- ✅ Mesmos resultados de modelos
- ✅ Mesma otimização Optuna

**Nota:** Resultados podem variar ligeiramente entre:
- Diferentes versões de bibliotecas
- Diferentes sistemas operacionais
- Diferentes hardware (GPU vs CPU)

## Checklist de Reprodução

- [ ] Ambiente Python 3.11 criado
- [ ] Dependências instaladas (`requirements.txt`)
- [ ] Dados em `06_dados/processados/`
- [ ] Notebook 4.1 executado com sucesso
- [ ] Notebook 4.3 executado com sucesso
- [ ] Modelos salvos em `07_modelos/`
- [ ] AUC-ROC ~0.9998 confirmado
- [ ] Visualizações SHAP geradas

## Suporte

Se encontrar problemas:
1. Verificar logs de erro
2. Consultar documentação específica:
   - Metodologia: `../01_metodologia/`
   - Dados: `../06_dados/README_DADOS.md`
   - Modelos: `../07_modelos/README_MODELOS.md`
   - Código: `../08_codigo/README_CODIGO.md`
3. Verificar versões de bibliotecas: `pip list`

## Estimativa de Tempo e Recursos

| Tarefa | Tempo | RAM Usada | CPU |
|--------|-------|-----------|-----|
| Instalação | 10 min | - | - |
| Limpeza (Etapa 0-2) | 45 min | 4 GB | 50% |
| EDA (Etapa 3) | 30 min | 6 GB | 30% |
| Pipeline ML (Etapa 4) | 1 hora | 8 GB | 70% |
| Otimização (Etapa 5) | 2-4 horas | 10 GB | 90% |
| **Total** | **5-6 horas** | **Peak: 10 GB** | **Avg: 60%** |

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Testado em:** Windows 11, Ubuntu 22.04, Python 3.11

