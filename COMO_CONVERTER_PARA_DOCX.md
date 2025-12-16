# Como Converter README.md para DOCX Acadêmico

## Opção 1: Usando Pandoc (Recomendado)

### Instalar Pandoc

**Windows:**
```bash
# Via winget
winget install JohnMacFarlane.Pandoc

# Ou download: https://pandoc.org/installing.html
```

**Linux:**
```bash
sudo apt install pandoc  # Ubuntu/Debian
sudo yum install pandoc  # CentOS/RHEL
```

**macOS:**
```bash
brew install pandoc
```

### Converter para DOCX

**Comando Básico:**
```bash
cd 0_artigo
pandoc README.md -o README.docx
```

**Com Formatação Acadêmica:**
```bash
pandoc README.md -o README.docx \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --reference-doc=template_academico.docx \
  --highlight-style=tango
```

**Parâmetros:**
- `--toc`: Gera índice automático
- `--toc-depth=3`: Índice até 3 níveis
- `--number-sections`: Numera seções automaticamente
- `--reference-doc`: Usa template DOCX customizado (opcional)
- `--highlight-style`: Estilo de código

### Template Acadêmico (Opcional)

Criar `template_academico.docx` com:
- Fonte: Times New Roman 12pt
- Espaçamento: 1.5
- Margens: 2.5cm
- Numeração de páginas
- Cabeçalho e rodapé

## Opção 2: Usando Python (Programático)

### Instalar pypandoc

```bash
pip install pypandoc
```

### Script de Conversão

```python
import pypandoc

# Converter
output = pypandoc.convert_file(
    'README.md',
    'docx',
    outputfile='README.docx',
    extra_args=[
        '--toc',
        '--number-sections',
        '--highlight-style=tango'
    ]
)

print("✅ Conversão concluída: README.docx")
```

## Opção 3: Online (Sem Instalação)

### Ferramentas Web

1. **Pandoc Online:** https://pandoc.org/try/
   - Cole conteúdo do README.md
   - Selecione output: DOCX
   - Download

2. **CloudConvert:** https://cloudconvert.com/md-to-docx
   - Upload README.md
   - Converter para DOCX
   - Download

3. **Dillinger:** https://dillinger.io/
   - Cole markdown
   - Export → Word

## Opção 4: Microsoft Word

1. Abrir Microsoft Word
2. File → Open → Selecionar README.md
3. Word abrirá em modo markdown
4. File → Save As → Format: Word Document (.docx)

**Nota:** Formatação pode precisar ajustes manuais.

## Pós-Processamento

Após conversão, ajustar no Word:

### 1. Índice
- Atualizar índice: Clique direito → Atualizar Campo

### 2. Imagens dos Diagramas

Converter diagramas mermaid em imagens:

**Online:**
- https://mermaid.live/
- Cole código mermaid
- Export PNG
- Inserir no DOCX

**VS Code:**
- Instalar extensão "Markdown Preview Mermaid Support"
- Capturar screenshot
- Inserir no DOCX

### 3. Formatação de Código

```
Fonte: Courier New ou Consolas
Tamanho: 10pt
Fundo: Cinza claro
```

### 4. Tabelas

- Verificar alinhamento
- Adicionar bordas se necessário
- Ajustar largura de colunas

### 5. Referências

- Converter para formato ABNT ou APA
- Adicionar DOIs clicáveis
- Formatar bibliografia

## Template DOCX Exemplo

### Estrutura Recomendada

```
Página de Rosto
├── Título
├── Autores
├── Instituição
├── Data
└── Palavras-chave

Resumo
├── Português
└── English (Abstract)

Índice (automático)

Conteúdo
├── Introdução
├── Metodologia
├── Resultados
├── Discussão
└── Conclusões

Referências Bibliográficas

Apêndices
```

### Formatação

```
Título: Times New Roman 16pt, Negrito, Centralizado
Seção: Times New Roman 14pt, Negrito, Alinhado à esquerda
Subseção: Times New Roman 12pt, Negrito, Alinhado à esquerda
Texto: Times New Roman 12pt, Justificado
Espaçamento: 1.5 linhas
Margens: 2.5cm (todas)
Numeração: Rodapé, centralizado, a partir da introdução
```

## Verificação Final

### Checklist

- [ ] Índice atualizado e correto
- [ ] Seções numeradas automaticamente
- [ ] Tabelas formatadas
- [ ] Código com destaque de sintaxe
- [ ] Diagramas como imagens
- [ ] Referências formatadas (ABNT/APA)
- [ ] Páginas numeradas
- [ ] Cabeçalhos e rodapés
- [ ] Espaçamento 1.5
- [ ] Margens corretas
- [ ] Sem erros de formatação

## Troubleshooting

### Erro: Pandoc não encontrado

**Solução:** Adicionar pandoc ao PATH
```bash
# Windows: Adicionar C:\Program Files\Pandoc\ ao PATH
# Linux/Mac: Reinstalar via package manager
```

### Diagramas Mermaid Não Aparecem

**Solução:** Converter manualmente para PNG e inserir no markdown antes da conversão:
```markdown
![Diagrama](05_diagramas/fluxo_pipeline.png)
```

### Tabelas Quebradas

**Solução:** Usar formato de tabela mais simples ou converter manualmente no Word após.

### Código Sem Formatação

**Solução:** Adicionar `--highlight-style=tango` ao comando pandoc.

## Resultado Final

Após conversão e ajustes, você terá:

✅ **README.docx** pronto para submissão em journals  
✅ Formatação acadêmica profissional  
✅ Índice automático navegável  
✅ Seções numeradas  
✅ Código com destaque  
✅ Tabelas e figuras bem formatadas  
✅ Referências adequadas  

---

**Versão:** 1.0  
**Data:** Dezembro 2024  
**Ferramenta Recomendada:** Pandoc 3.x

