# Como Baixar os Datasets

## ⚠️ Importante: Datasets Não Incluídos no Repositório

Os arquivos CSV (datasets) **não estão incluídos** neste repositório Git devido ao seu tamanho (> 1 GB total).

## 📥 Onde Baixar os Dados

### Opção 1: Google Drive (DISPONÍVEL) ✅

**Link direto:** [https://drive.google.com/drive/folders/1j7OiuMJuQ8tu7trlZJ4Zbo5Attu01knM?usp=drive_link](https://drive.google.com/drive/folders/1j7OiuMJuQ8tu7trlZJ4Zbo5Attu01knM?usp=drive_link)

**Instruções detalhadas:** Ver `LEIA_ONDE_ESTAO_OS_DATASETS.md`

### Opção 2: Zenodo / Figshare (Recomendado para Pesquisa)
<!-- TODO: Criar DOI e adicionar link -->
```
[Link e DOI serão adicionados aqui]
```

### Opção 3: Contato Direto
Entre em contato com os autores para solicitar acesso aos dados.

## 📂 Arquivos Necessários

Após download, colocar os arquivos nas seguintes pastas:

### `06_dados/processados/`
- `6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv` (~250 MB)
- `7_dados_unidos.csv` (~50 MB)
- `dataset_unificado.csv` (~600 MB)
- `dataset_com_posts.csv` (~1 MB)
- `dataset_sem_posts.csv` (~200 MB)

### `06_dados/amostras/`
- `amostra_estabelecimentos_rs.csv` (~5 MB)

## ✅ Verificar Integridade

Após baixar, verificar se os arquivos foram extraídos corretamente:

```python
import pandas as pd
import os

# Verificar arquivos
arquivos_necessarios = [
    '06_dados/processados/6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv',
    '06_dados/processados/7_dados_unidos.csv',
    '06_dados/processados/dataset_unificado.csv',
    '06_dados/processados/dataset_com_posts.csv',
    '06_dados/processados/dataset_sem_posts.csv',
    '06_dados/amostras/amostra_estabelecimentos_rs.csv'
]

for arquivo in arquivos_necessarios:
    if os.path.exists(arquivo):
        df = pd.read_csv(arquivo, nrows=5)
        print(f"✅ {arquivo}: {len(df)} linhas (amostra)")
    else:
        print(f"❌ {arquivo}: AUSENTE")
```

## 📊 Estrutura dos Datasets

Ver documentação completa em: `06_dados/README_DADOS.md`

## 🔐 Licença dos Dados

- **Dados da Receita Federal:** Dados públicos, uso livre
- **Dados do Instagram:** Anonimizados, apenas para pesquisa

## 📧 Suporte

Para questões sobre acesso aos dados:
- Abrir issue no GitHub
- Contatar autores diretamente

---

**Nota:** Os modelos treinados (.joblib) **estão incluídos** no repositório e podem ser usados sem necessidade de baixar os datasets completos.

