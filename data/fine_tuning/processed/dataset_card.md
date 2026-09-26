# Dataset Card — Fine-tuning do assistente médico (Fase 3)

## Composição

| Split | Exemplos | Por tipo |
|---|---|---|
| train | 1123 | {'cancer_geral': 535, 'recusa_segura': 200, 'protocolo': 204, 'faq': 96, 'modelo_documento': 28, 'mama': 60} |
| val | 139 | {'recusa_segura': 25, 'protocolo': 24, 'mama': 8, 'cancer_geral': 67, 'faq': 12, 'modelo_documento': 3} |
| test | 136 | {'cancer_geral': 66, 'modelo_documento': 3, 'protocolo': 24, 'faq': 12, 'recusa_segura': 24, 'mama': 7} |

## Origem dos dados

1. **MedQuAD** (perguntas e respostas de sites do NIH) — filtrado para câncer de
   mama e para a fonte CancerGov; traduzido/resumido para português com
   Claude Haiku. Licença **CC BY 4.0**.
   Citação obrigatória: Ben Abacha A., Demner-Fushman D. *A Question-Entailment
   Approach to Question Answering.* BMC Bioinformatics, 2019.
2. **Dados sintéticos do hospital (fictícios)** — protocolos internos, FAQ de
   médicos, modelos de laudo/receita e exemplos de recusa segura, gerados
   com Claude. Não contêm dados de pessoas ou instituições reais.

## Processamento

- Curadoria: {'linhas_iniciais': 772, 'apos_remover_respostas_curtas': 771, 'apos_remover_respostas_repetidas': 743, 'respostas_longas_a_resumir': 344}
- Traduções que falharam e foram descartadas: 0
- Anonimização (itens mascarados): nenhum encontrado
- Split 80/10/10 feito por grupo (respostas idênticas / mesmo protocolo ficam
  sempre no mesmo split) e verificado automaticamente contra vazamento.

## Limitações

- Traduções e textos sintéticos foram gerados por IA e não passaram por revisão
  médica completa; o assistente é apoio à decisão e não substitui o médico.
- O MedQuAD é conteúdo público de educação em saúde, não protocolo real de hospital.
