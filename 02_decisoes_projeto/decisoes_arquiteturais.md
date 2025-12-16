# Decisões Arquiteturais

## 1. Processamento em Chunks

**Decisão:** Implementar leitura em chunks para arquivos > 500MB

**Justificativa:**
- Dataset com 2.6M+ linhas pode exceder memória RAM
- Escalabilidade para qualquer tamanho
- Mantém performance aceitável

**Trade-offs:**
- ✅ Escalável
- ✅ Menor uso de memória
- ❌ Tempo ligeiramente maior
- ❌ Complexidade adicional

## 2. Estratégia Híbrida

**Decisão:** Implementar estratégia híbrida (unified + separate)

**Justificativa:**
- Desbalanceamento extremo (1:1018)
- Modelos especializados capturam padrões específicos
- Dataset unificado aproveita todo volume

**Resultado:** Performance excelente (AUC 0.9998) com unified

## 3. Pipeline Modular

**Decisão:** Dividir em 6 etapas independentes

**Benefícios:**
- Facilita reprodução
- Permite debugging específico
- Reutilização de resultados
- Manutenção facilitada

## 4. Seed Fixo (42)

**Decisão:** random_state=42 em todas operações aleatórias

**Justificativa:** Reprodutibilidade científica

---

**Versão:** 1.0  
**Data:** Dezembro 2024

