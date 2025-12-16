# Requisitos Técnicos

## Sistema Operacional

### Suportados
- **Windows:** 10, 11
- **Linux:** Ubuntu 20.04+, Debian 11+, CentOS 8+
- **macOS:** 10.15 (Catalina) ou superior

## Python

### Versão
- **Mínimo:** Python 3.9
- **Recomendado:** Python 3.11 ou 3.12
- **Não suportado:** Python 3.8 ou inferior

### Verificar Versão
```bash
python --version
# Deve mostrar: Python 3.11.x ou superior
```

## Hardware

### Configuração Mínima
- **CPU:** 4 cores, 2.5 GHz
- **RAM:** 16 GB
- **Storage:** 10 GB livres
- **GPU:** Não necessária

### Configuração Recomendada
- **CPU:** 8+ cores, 3.0+ GHz (Intel i7/AMD Ryzen 7 ou superior)
- **RAM:** 32 GB
- **Storage:** 50 GB livres (SSD preferível)
- **GPU:** Opcional (ajuda com SHAP e deep learning futuro)

### Configuração Ideal (Servidor)
- **CPU:** 16+ cores (Intel Xeon/AMD EPYC)
- **RAM:** 64 GB
- **Storage:** 100 GB SSD
- **GPU:** NVIDIA com CUDA (opcional)

## Dependências Python

Ver: `08_codigo/requirements.txt`

### Core (Obrigatórias)
```
pandas >= 2.3.0
numpy >= 1.26.4
scikit-learn >= 1.7.0
xgboost >= 2.1.3
lightgbm >= 4.5.0
optuna >= 4.1.0
matplotlib >= 3.9.3
seaborn >= 0.13.2
joblib >= 1.4.2
```

### Opcionais (Recomendadas)
```
catboost >= 1.2.7
shap >= 0.49.1
jupyterlab >= 4.3.3
```

## Instalação de Dependências

### Via pip (Recomendado)
```bash
pip install -r 08_codigo/requirements.txt
```

### Via conda (Alternativo)
```bash
conda install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm optuna shap catboost
```

## Ferramentas Auxiliares

### Para Execução
- **Jupyter Notebook** ou **JupyterLab**
- **IDE:** VSCode, PyCharm, ou similar (opcional)

### Para Desenvolvimento (Opcional)
- **Git:** Versionamento
- **Black:** Formatação de código
- **Pytest:** Testes

## Configuração GPU (Opcional)

### Para SHAP Acelerado

**NVIDIA GPU + CUDA:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Verificar CUDA:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

## Testes de Sistema

### Teste de RAM
```python
import numpy as np

# Tentar criar array grande
try:
    arr = np.zeros((10000, 5000))  # ~400 MB
    print("✅ RAM suficiente")
except MemoryError:
    print("❌ RAM insuficiente")
```

### Teste de CPU
```python
import multiprocessing

cores = multiprocessing.cpu_count()
print(f"CPUs disponíveis: {cores}")
if cores >= 4:
    print("✅ CPU adequada")
else:
    print("⚠️ CPU pode ser limitante")
```

### Teste de Storage
```python
import os

free_space_gb = os.statvfs('/').f_bavail * os.statvfs('/').f_frsize / (1024**3)
print(f"Espaço livre: {free_space_gb:.1f} GB")
if free_space_gb >= 10:
    print("✅ Storage suficiente")
else:
    print("❌ Storage insuficiente")
```

## Performance Esperada

### Tempos de Execução (Hardware Recomendado)

| Etapa | Tempo |
|-------|-------|
| Limpeza | 5-10 min |
| Agregação | 10-15 min |
| Feature Engineering | 15-20 min |
| EDA | 20-30 min |
| Pipeline ML | 30-60 min |
| Otimização (50 trials) | 1-2 horas |
| **Total** | **~3-4 horas** |

### Hardware Mínimo (16 GB RAM, 4 cores)

| Etapa | Tempo |
|-------|-------|
| Total | **~5-6 horas** |

## Limitações Conhecidas

### Por Hardware

**16 GB RAM:**
- Usar chunk_size = 5000
- Processar datasets em partes
- SHAP limitado a 50-100 amostras

**4 Cores CPU:**
- Otimização Optuna mais lenta
- Paralelização limitada
- ~2x mais tempo que 8 cores

**SSD vs HDD:**
- HDD: +30% tempo de I/O
- SSD recomendado para datasets grandes

## Compatibilidade

### Versões de Bibliotecas

**Testado e Funcionando:**
- pandas 2.3.0
- scikit-learn 1.7.0
- xgboost 2.1.3
- optuna 4.1.0

**Compatibilidade com Versões Antigas:**
- pandas >= 1.5.0 (pode funcionar)
- scikit-learn >= 1.3.0 (pode funcionar)
- xgboost >= 1.7.0 (pode funcionar)

**Aviso:** Versões mais antigas podem ter APIs diferentes.

### Sistemas Operacionais

**100% Compatível:**
- Ubuntu 22.04 LTS
- Windows 11
- macOS 12+ (Monterey)

**Compatível com Ajustes:**
- CentOS 8 (ajustar instalação pip)
- Debian 11 (ajustar python-dev)

**Não Testado:**
- Windows 7/8
- macOS < 10.15
- Distribuições Linux exóticas

## Troubleshooting

### Erro: "Python version too old"

**Problema:** Python < 3.9

**Solução:**
```bash
# Atualizar Python
# Windows: Download do python.org
# Ubuntu: 
sudo apt update
sudo apt install python3.11
# macOS:
brew install python@3.11
```

### Erro: "Insufficient memory"

**Problema:** RAM < 16 GB

**Solução:** Reduzir cargas ou usar servidor

### Erro: "Disk full"

**Problema:** Storage < 10 GB

**Solução:** Liberar espaço ou usar drive externo

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Hardware Testado:** Intel i7-12700, 32GB RAM, SSD 1TB

