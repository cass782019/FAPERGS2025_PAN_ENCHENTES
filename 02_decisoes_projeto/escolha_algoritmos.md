# Escolha dos Algoritmos

## Por Que Gradient Boosting?

### Justificativas

1. **Performance Superior:** State-of-the-art para dados tabulares
2. **Tratamento de Missing:** Tree-based lidam bem com NaN
3. **Não-Linearidade:** Capturam relações complexas
4. **Interpretabilidade:** Feature importance + SHAP
5. **Escalabilidade:** XGBoost e LightGBM otimizados

## Por Que XGBoost Como Final?

### Comparação

| Aspecto | XGBoost | LightGBM | CatBoost | Random Forest |
|---------|---------|----------|----------|---------------|
| AUC | 0.9998 ⭐ | 0.9998 | 0.9997 | 0.9998 |
| Velocidade | Moderada | Rápida ⭐ | Lenta | Moderada |
| Memória | Moderada | Baixa ⭐ | Alta | Alta |
| Maturidade | Muito madura ⭐ | Madura | Recente | Muito madura |
| Ecossistema | Rico ⭐ | Bom | Médio | Rico |

**Conclusão:** XGBoost por maturidade e ecossistema, apesar de performance equivalente.

## Por Que Não Redes Neurais?

1. Tree-based superiores para tabulares
2. Interpretabilidade necessária
3. 2.6M registros suficiente para trees, não ideal para DL
4. Custo computacional maior
5. Performance já excelente

---

**Versão:** 1.0  
**Data:** Dezembro 2024

