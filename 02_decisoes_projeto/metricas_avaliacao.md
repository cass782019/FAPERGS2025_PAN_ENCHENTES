# Métricas de Avaliação

## Métrica Principal: AUC-ROC

### Por Que AUC-ROC?

1. **Invariante a threshold:** Não precisa escolher ponto de corte
2. **Boa para desbalanceamento moderado:** Target 1:1.8
3. **Interpretação intuitiva:** P(ranquear positivo > negativo)
4. **Padrão da indústria:** Facilita comparação
5. **Otimização direta:** Optuna pode otimizar AUC

### Limitações

- Não indica performance em threshold específico
- Pode ser otimista em desbalanceamento extremo

## Métrica Secundária: Average Precision

### Por Que AP?

1. **Foco em precision-recall:** Mais informativa
2. **Melhor para desbalanceamento extremo:** Se aplicável
3. **Complementa AUC-ROC:** Perspectiva diferente

## Métricas Reportadas (Não Principais)

### F1-Score
- Depende de threshold (0.5 padrão)
- Reportado mas não usado para seleção

### Accuracy
- ❌ Inadequada para desbalanceamento
- Não usada

### Confusion Matrix
- Útil para análise
- Requer threshold definido

## Conclusão

AUC-ROC como principal, AP como secundária. Outras métricas reportadas para contexto.

---

**Versão:** 1.0  
**Data:** Dezembro 2024

