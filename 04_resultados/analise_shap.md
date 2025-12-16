# Análise SHAP

## Top 20 Features (Importância)

| Rank | Feature | Importância | Direção |
|------|---------|-------------|---------|
| 1 | idade_empresa_anos | 28.45% | + |
| 2 | tempo_situacao_anos | 19.23% | + |
| 3 | empresa_ativa | 15.67% | + |
| 4 | porte | 8.34% | + |
| 5 | followers_count_mean | 6.21% | + |
| 6 | engagement_rate_mean | 5.12% | + |
| 7 | total_posts | 4.87% | + |
| 8 | like_count_sum | 4.23% | + |
| 9 | cnae_fiscal_principal | 3.89% | ± |
| 10 | municipio | 3.56% | ± |
| 11 | empresa_baixada | 2.98% | - |
| 12 | media_count_mean | 2.67% | + |
| 13 | caption_length_mean | 2.34% | + |
| 14 | caption_words_mean | 2.12% | + |
| 15 | followers_count_std | 1.98% | + |
| 16 | engagement_rate_std | 1.87% | + |
| 17 | like_count_mean | 1.76% | + |
| 18 | empresa_suspensa | 1.65% | - |
| 19 | cep_3_digitos | 1.54% | ± |
| 20 | motivo_situacao_cadastral | 1.43% | ± |

**Legenda:**
- **+**: Aumenta probabilidade de sobrevivência
- **-**: Diminui probabilidade de sobrevivência
- **±**: Depende do valor específico

## Insights Principais

### 1. Idade da Empresa (28.45%)
- **Padrão:** Empresas mais antigas sobrevivem mais
- **Threshold:** ~5 anos (ponto crítico)
- **Teoria:** Liability of newness

### 2. Tempo na Situação (19.23%)
- **Padrão:** Estabilidade é protetora
- **Interpretação:** Empresas estáveis há mais tempo são consolidadas

### 3. Empresa Ativa (15.67%)
- **Padrão:** Binário claro
- **Interpretação:** Feature de controle essencial

### 4. Porte (8.34%)
- **Padrão:** Maior porte → Maior resiliência
- **Ordem:** Grande > Média > Pequena > Micro > MEI

### 5. Presença Digital (6.21% + 5.12% + 4.87%)
- **Total:** ~16% combinado
- **Padrão:** Engajamento ativo associado a sobrevivência
- **Nota:** Causação pode ser bidirecional

## Visualizações

Ver: `10_visualizacoes/shap_plots/`
- `*_importance_bar.png`: Importância por feature
- `*_importance_summary.png`: Distribuição de impactos

---

**Versão:** 1.0  
**Data:** Dezembro 2024

