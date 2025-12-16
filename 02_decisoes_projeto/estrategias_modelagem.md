# Estratégias de Modelagem

## Três Estratégias Implementadas

### 1. UNIFIED (Unificado)

**Descrição:** Dataset único com imputação

**Quando Usar:**
- Desbalanceamento moderado
- Simplicidade desejada
- Todos os dados devem ser aproveitados

**Vantagens:**
- Simples de implementar
- Usa todos os dados
- Modelo único

**Desvantagens:**
- Imputação pode distorcer
- Não especializado

### 2. SEPARATE (Separado)

**Descrição:** Datasets separados (com/sem posts)

**Quando Usar:**
- Padrões muito distintos
- Desbalanceamento extremo
- Análise comparativa necessária

**Vantagens:**
- Modelos especializados
- Capturam padrões específicos
- Sem imputação artificial

**Desvantagens:**
- Menos dados por modelo
- Mais complexo
- Dois modelos para manter

### 3. HYBRID (Híbrido) ⭐ RECOMENDADO

**Descrição:** Gera os 3 datasets

**Quando Usar:**
- Desbalanceamento extremo
- Robustez necessária
- Ensemble desejado

**Vantagens:**
- Melhor dos dois mundos
- Permite ensemble
- Análise comparativa

**Desvantagens:**
- Mais complexo
- Mais recursos computacionais

## Resultado no Projeto

**Estratégia Unified** teve performance excepcional (AUC 0.9998), tornando separate desnecessário para produção. Porém, hybrid foi mantido para análises e robustez.

---

**Versão:** 1.0  
**Data:** Dezembro 2024

