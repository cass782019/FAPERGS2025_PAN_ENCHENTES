# Limitações dos Dados

## 1. Viés de Seleção

**Problema:** Apenas 0.098% das empresas têm Instagram

**Implicações:**
- Empresas com Instagram são sistematicamente diferentes
- Generalização limitada
- Causalidade não pode ser inferida

**Mitigação:** Estratégia separate permite análise específica

## 2. Cobertura Temporal

**Problema:** Dados de posts com cobertura variável

**Implicações:**
- Features agregadas não comparáveis
- Sazonalidade não capturada

**Mitigação:** Features normalizadas (médias)

## 3. Dados Ausentes

**Missing Values:**
- CEP: ~15%
- CNAE: ~10%
- Features de posts: 99.9%

**Mitigação:** SimpleImputer, tree-based models

## 4. Qualidade da Fonte

**Problema:** Dependência da Receita Federal

**Implicações:**
- Empresas "zombie"
- Defasagem temporal
- Qualidade fundamental não controlável

**Mitigação:** Validação, cruzamento com posts

---

**Versão:** 1.0  
**Data:** Dezembro 2024

