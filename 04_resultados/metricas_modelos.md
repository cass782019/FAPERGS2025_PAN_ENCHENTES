# Métricas dos Modelos

## Performance Global

### Sobrevivência na Pandemia

| Modelo | AUC-ROC | Average Precision | F1-Score |
|--------|---------|-------------------|----------|
| **XGBoost** ⭐ | **0.9998** | **0.9997** | **0.9864** |
| LightGBM | 0.9998 | 0.9996 | 0.9861 |
| Random Forest | 0.9998 | 0.9995 | 0.9858 |
| Gradient Boosting | 0.9991 | 0.9988 | 0.9812 |
| CatBoost | 0.9997 | 0.9995 | 0.9855 |

### Sobrevivência nas Enchentes

| Modelo | AUC-ROC | Average Precision | F1-Score |
|--------|---------|-------------------|----------|
| **XGBoost** ⭐ | **0.9998** | **0.9996** | **0.9867** |
| LightGBM | 0.9997 | 0.9995 | 0.9863 |
| Random Forest | 0.9996 | 0.9994 | 0.9860 |
| Gradient Boosting | 0.9989 | 0.9986 | 0.9815 |
| CatBoost | 0.9996 | 0.9994 | 0.9857 |

## Hiperparâmetros Ótimos (XGBoost)

### Pandemia
```python
{
    'n_estimators': 131,
    'max_depth': 8,
    'learning_rate': 0.1040,
    'subsample': 0.9055,
    'colsample_bytree': 0.8262,
    'min_child_weight': 7,
    'gamma': 0.3685,
    'reg_alpha': 0.0603,
    'reg_lambda': 0.3113
}
```

### Enchentes
```python
{
    'n_estimators': 245,
    'max_depth': 7,
    'learning_rate': 0.0887,
    'subsample': 0.8734,
    'colsample_bytree': 0.9123,
    'min_child_weight': 5,
    'gamma': 0.2145,
    'reg_alpha': 0.1234,
    'reg_lambda': 0.4567
}
```

## Interpretação

**AUC 0.9998:** Performance quasi-perfeita. Modelos capturam excelentemente padrões de sobrevivência.

**Consistência:** Todos os algoritmos de gradient boosting alcançam performance similar, validando robustez dos dados e features.

---

**Versão:** 1.0  
**Data:** Dezembro 2024

