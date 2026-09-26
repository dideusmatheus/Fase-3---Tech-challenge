# PROT-21 — Protocolo Interno: Uso de Modelo de IA de Apoio ao Diagnóstico (Breast Cancer Wisconsin) com Validação Humana

# Protocolo Interno: Uso de Modelo de IA de Apoio ao Diagnóstico (Breast Cancer Wisconsin) com Validação Humana

## Objetivo
Estabelecer diretrizes para a utilização segura e eficaz do modelo de inteligência artificial Breast Cancer Wisconsin como ferramenta de apoio ao diagnóstico diferencial em casos de câncer de mama, garantindo sempre a validação humana por profissional qualificado e mantendo a responsabilidade clínica exclusiva do médico assistente.

## Indicações
O modelo de IA deve ser utilizado como auxiliar diagnóstico nos seguintes cenários:
- Análise de biópsias de mama com diagnóstico histopatológico indeterminado ou complexo
- Casos de diagnóstico diferencial entre lesões benignas e malignas
- Suporte em segunda opinião diagnóstica interna
- Auxílio em revisão de casos com alta relevância clínica
- Situações em que há dúvida diagnóstica entre profissionais

Nunca substitui a avaliação clínica, exame físico, imagiologia ou análise histopatológica por patologista credenciado.

## Fluxo de Trabalho
1. **Identificação do caso candidato**: O médico oncologista ou patologista identifica caso potencialmente adequado para análise pela IA.
2. **Preparação de dados**: Dados clínicos e resultados preliminares são inseridos no sistema de forma anônima, utilizando marcadores como [CASO_ID].
3. **Processamento pela IA**: O modelo Breast Cancer Wisconsin processa as informações e gera probabilidades de diagnóstico.
4. **Revisão e validação humana**: Médico especialista (oncologista ou patologista) revisa criticamente todo parecer da IA, comparando com dados clínicos, imagiológicos e histopatológicos.
5. **Documentação**: Resultado final (apenas a conclusão do médico) é documentado no prontuário do paciente com registro de que foi utilizado apoio diagnóstico por IA.
6. **Comunicação com paciente**: Toda informação ao paciente refere-se exclusivamente ao diagnóstico validado pelo médico responsável.

## Responsabilidades
- **Médico solicitante**: Selecionar casos apropriados e garantir que todas as informações clínicas estejam disponíveis.
- **Médico revisor (validação)**: Responsável pela análise crítica do resultado da IA, decisão diagnóstica final e assinatura eletrônica do parecer.
- **Equipe de TI/Compliance**: Monitorar segurança de dados, anonymização correta e manutenção do sistema.
- **Direção médica**: Supervisionar implantação, garantir conformidade regulatória e revisar resultados periodicamente.

## Alertas e Limitações Importantes
- O modelo de IA é auxiliar; decisão diagnóstica final é sempre responsabilidade exclusiva do médico.
- Não use resultado da IA como único critério para diagnóstico ou conduta terapêutica.
- Sistema não substitui correlação clínico-patológica rigorosa.
- Em caso de discordância entre IA e avaliação clínica, priorize sempre o julgamento do médico assistente.
- Dados devem ser completamente anonimizados; nenhuma informação de identificação pessoal é permitida no sistema.
- Resultados da IA devem ser documentados em seção separada do prontuário, claramente identificados como "apoio diagnóstico".
- Treinamento obrigatório para todos os usuários antes de primeiro acesso ao sistema.
- Auditoria mensal de concordância entre diagnóstico final do médico e sugestões da IA.
