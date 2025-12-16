# 📥 ONDE ESTÃO OS DATASETS

## ⚠️ IMPORTANTE

Os arquivos CSV (datasets) **NÃO estão incluídos** neste repositório GitHub devido ao tamanho total superior a 1 GB.

## 📂 Download dos Dados

### Google Drive (Principal)

**Link de acesso:**
👉 [https://drive.google.com/drive/folders/1j7OiuMJuQ8tu7trlZJ4Zbo5Attu01knM?usp=drive_link](https://drive.google.com/drive/folders/1j7OiuMJuQ8tu7trlZJ4Zbo5Attu01knM?usp=drive_link)

### Arquivos Disponíveis

O Google Drive contém todos os datasets processados necessários para reproduzir o pipeline:

#### Pasta `processados/`
- ✅ `6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv` (~250 MB)
- ✅ `7_dados_unidos.csv` (~50 MB)
- ✅ `dataset_unificado.csv` (~600 MB)
- ✅ `dataset_com_posts.csv` (~1 MB)
- ✅ `dataset_sem_posts.csv` (~200 MB)

#### Pasta `amostras/`
- ✅ `amostra_estabelecimentos_rs.csv` (~5 MB)

**Total:** ~1.1 GB

## 📋 Como Usar

### 1. Baixar os Arquivos

1. Acesse o link do Google Drive acima
2. Faça download de todos os arquivos
3. Organize-os na estrutura abaixo

### 2. Estrutura de Pastas

Após o download, organize os arquivos assim:

```
0_artigo/
└── 06_dados/
    ├── processados/
    │   ├── 6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv
    │   ├── 7_dados_unidos.csv
    │   ├── dataset_unificado.csv
    │   ├── dataset_com_posts.csv
    │   └── dataset_sem_posts.csv
    └── amostras/
        └── amostra_estabelecimentos_rs.csv
```

### 3. Verificar Integridade

Execute este script Python para verificar se todos os arquivos foram baixados corretamente:

```python
import pandas as pd
import os

arquivos = {
    '06_dados/processados/6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv': 2685868,
    '06_dados/processados/7_dados_unidos.csv': None,  # Variável
    '06_dados/processados/dataset_unificado.csv': 2685868,
    '06_dados/processados/dataset_com_posts.csv': 2638,
    '06_dados/processados/dataset_sem_posts.csv': 2683230,
    '06_dados/amostras/amostra_estabelecimentos_rs.csv': 10001
}

print("Verificando integridade dos datasets...\n")
tudo_ok = True

for arquivo, linhas_esperadas in arquivos.items():
    if os.path.exists(arquivo):
        try:
            df = pd.read_csv(arquivo)
            tamanho = len(df)
            tamanho_mb = os.path.getsize(arquivo) / (1024 * 1024)
            
            status = "✅"
            if linhas_esperadas and abs(tamanho - linhas_esperadas) > 10:
                status = "⚠️"
                tudo_ok = False
            
            print(f"{status} {arquivo}")
            print(f"   Linhas: {tamanho:,} | Tamanho: {tamanho_mb:.1f} MB")
        except Exception as e:
            print(f"❌ {arquivo}: ERRO ao ler - {e}")
            tudo_ok = False
    else:
        print(f"❌ {arquivo}: ARQUIVO NÃO ENCONTRADO")
        tudo_ok = False
    print()

if tudo_ok:
    print("🎉 Todos os datasets estão OK! Você pode prosseguir com o pipeline.")
else:
    print("⚠️ Alguns problemas foram encontrados. Verifique os arquivos.")
```

## 📊 Descrição dos Datasets

### 6_empresas_rs_porte_sobreviveu_pandemia_enchente.csv

**Descrição:** Dataset principal com dados de empresas do RS e targets de sobrevivência

**Linhas:** 2.685.868  
**Colunas:** 20  
**Tamanho:** ~250 MB

**Principais colunas:**
- `cnpj_basico`: Identificador único (8 dígitos)
- `porte`: MEI, MICRO, PEQUENA, MEDIA, GRANDE
- `sobreviveu_pandemia`: Target 1 (1/0)
- `sobreviveu_enchente`: Target 2 (1/0)
- `idade_empresa_anos`: Feature temporal
- `situacao_cadastral`: ATIVA, BAIXADA, etc.

### 7_dados_unidos.csv

**Descrição:** Posts do Instagram de empresas do RS

**Empresas únicas:** 2.638  
**Tamanho:** ~50 MB

**Principais colunas:**
- `cnpj`: CNPJ da empresa
- `followers_count`: Seguidores
- `like_count`: Curtidas
- `caption`: Texto do post
- `timestamp`: Data/hora

### dataset_unificado.csv

**Descrição:** Dataset final combinando dados de empresas + features de posts agregadas

**Linhas:** 2.685.868  
**Colunas:** 43  
**Tamanho:** ~600 MB

**Uso:** Entrada principal para modelagem (Etapa 5)

### dataset_com_posts.csv / dataset_sem_posts.csv

**Descrição:** Subsets para estratégia "separate"

**Com posts:** 2.638 empresas  
**Sem posts:** 2.683.230 empresas

## 🔐 Licença e Uso dos Dados

### Dados da Receita Federal
- **Fonte:** Cadastro Nacional da Pessoa Jurídica (CNPJ) - Dados Públicos
- **Licença:** Uso livre, dados públicos
- **URL:** https://www.gov.br/receitafederal/dados-abertos

### Dados do Instagram
- **Anonimização:** CNPJs mantidos, conteúdo de posts anônimo
- **Uso:** Apenas para pesquisa acadêmica e científica
- **Restrições:** Não redistribuir comercialmente

## 📧 Suporte

### Problemas com Download?

- **Link não funciona:** Abra uma issue no GitHub
- **Arquivos corrompidos:** Tente baixar novamente
- **Acesso negado:** Verifique se o link está correto

### Perguntas?

- **GitHub Issues:** [https://github.com/cass782019/FAPERGS2025_PAN_ENCHENTES/issues](https://github.com/cass782019/FAPERGS2025_PAN_ENCHENTES/issues)
- **Documentação completa:** `06_dados/README_DADOS.md`

## ✅ Checklist

Antes de executar o pipeline, confirme:

- [ ] Baixei todos os 6 arquivos CSV do Google Drive
- [ ] Organizei nas pastas `processados/` e `amostras/`
- [ ] Executei o script de verificação de integridade
- [ ] Li a documentação em `README_DADOS.md`
- [ ] Instalei as dependências (`08_codigo/requirements.txt`)

## 🚀 Próximos Passos

Após baixar os dados:

1. **Verificar integridade** (script acima)
2. **Ler documentação:** `06_dados/README_DADOS.md`
3. **Instalar dependências:** `pip install -r 08_codigo/requirements.txt`
4. **Executar pipeline:** Seguir `09_anexos/como_reproduzir.md`

---

**Última atualização:** Dezembro 2024  
**Tamanho total dos dados:** ~1.1 GB  
**Tempo estimado de download:** 5-15 minutos (depende da conexão)

