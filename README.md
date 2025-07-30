# LunarisCodex-MoC

*Um Modelo de Linguagem Experimental com um Módulo de Experts Colaborativos.*

---

## Visão Geral (Overview)

`LunarisCodex-MoC` é um modelo de linguagem grande (LLM) funcional, no estilo Llama, que serve como uma base de testes para arquiteturas de redes neurais de ponta. O projeto substitui a camada Feed-Forward Network (FFN) padrão dos blocos Transformer por uma implementação inovadora de **Mixture of Collaborative Experts (MoC)**. Esta arquitetura visa superar as limitações dos modelos Mixture-of-Experts (MoE) tradicionais, permitindo que os experts interajam e refinem suas saídas coletivamente antes da fusão final.

## Entendendo o Conceito: A Analogia dos Ministros

Para compreender melhor a diferença entre MoE tradicional e o novo MoC, imagine um governo lidando com um problema complexo:

### O Método Antigo: MoE Padrão (O Presidente Sobrecarregado)

Imagine um presidente que precisa lidar com um relatório sobre "Economia vs. Meio Ambiente":

1. **Decisão Rápida**: O presidente olha apenas o título e pensa: "80% Economia, 20% Meio Ambiente"
2. **Seleção Simples**: Chama apenas a Ministra da Economia e o Ministro do Meio Ambiente
3. **Trabalho Isolado**: Cada ministro vai para sua sala e trabalha sozinho, sem conversar
4. **Combinação Mecânica**: O presidente pega 80% das ideias da Economia e 20% do Meio Ambiente
5. **O Problema**: A solução final é incoerente - as políticas econômicas podem contradizer diretamente as ambientais

### O Novo Método: MoC (O Presidente Sábio)

O mesmo problema, mas com uma abordagem muito mais sofisticada:

#### Fase 1: Roteamento Contextualizado (A Reunião Preliminar)
1. **Pré-Análise**: O presidente envia o relatório para TODOS os 8 ministros e pede um resumo inicial
2. **Reunião de Contexto**: Todos se reúnem em uma sala:
   - A Ministra da Economia ouve o Ministro de Relações Exteriores falar sobre acordos internacionais
   - O Ministro do Meio Ambiente ouve sobre campanhas educacionais
   - Eles não estão resolvendo ainda, apenas entendendo as conexões
3. **Decisão Informada**: Com essa visão holística, o presidente percebe que precisa também do Ministro de Relações Exteriores (acordos climáticos) - escolhe 3 ministros em vez de 2

#### Fase 2: Fusão Colaborativa (A Força-Tarefa)
1. **Reunião de Trabalho**: Os 3 ministros escolhidos vão para uma sala de reuniões
2. **Discussão e Refinamento**:
   - Economia: "Proponho incentivos fiscais para empresas verdes"
   - Relações Exteriores: "Ótimo, mas o Acordo de Paris exige metas mais rígidas"
   - Meio Ambiente: "E se usarmos esses incentivos para subsidiar pesquisa em captura de carbono?"
3. **Relatório Integrado**: Em vez de 3 relatórios separados, produzem uma solução coesa onde cada aspecto considera os outros

**Resultado**: Uma política que é economicamente viável, ambientalmente responsável e internacionalmente alinhada.

## A Evolução: Do MoE ao MoC

O projeto começou como uma exploração dos modelos MoE padrão, que utilizam um roteador para enviar cada token para um único expert (`k=1`), seguindo a lógica do Switch Transformer. Embora eficientes, os MoEs tradicionais têm uma limitação fundamental: **os experts trabalham em completo isolamento**. A saída final é simplesmente o resultado do expert escolhido, sem qualquer sinergia ou refinamento entre eles.

A questão que impulsionou este projeto foi: "E se os experts pudessem colaborar?". O MoC (Mixture of Collaborative Experts) é a resposta a essa pergunta.

## Inovações Arquiteturais do MoC

O `CollaborativeExpertsModule` é o coração do `LunarisCodex-MoC`. Ele introduz duas inovações principais que o distinguem dos MoEs convencionais.

### 1. Router Contextualization (Roteamento Contextualizado)

> **O Problema:** Roteadores MoE padrão tomam decisões com base apenas na representação do token de entrada, sem qualquer conhecimento do que os diferentes experts "pensam" sobre esse token. A decisão é cega.
>
> **A Solução MoC:** O roteador do MoC é muito mais sofisticado. O fluxo de decisão é o seguinte:
> 1. O token de entrada é enviado para **todos os experts** em paralelo (como enviar o relatório para todos os ministros)
> 2. Uma camada de **auto-atenção (self-attention)** é aplicada sobre esses outputs dos experts (a "reunião preliminar" onde todos escutam uns aos outros)
> 3. Com base nesse resumo contextual rico, o roteador toma uma decisão de roteamento top-k muito mais informada (o presidente escolhe a equipe certa com base na discussão completa)

### 2. Collaborative Fusion (Fusão Colaborativa)

> **O Problema:** Em MoEs padrão que usam `k > 1`, as saídas dos experts selecionados são simplesmente combinadas através de uma soma ponderada. Não há colaboração real.
>
> **A Solução MoC:** Uma vez que os `k` experts são selecionados, a colaboração começa:
> 1. As saídas dos `k` experts escolhidos são empilhadas (os ministros escolhidos vão para a sala de reuniões)
> 2. Essas saídas são alimentadas em uma camada de **atenção cruzada (cross-attention)** (cada ministro escuta e responde aos outros)
> 3. Isso permite que cada expert refine sua própria saída com base nas perspectivas de seus pares (refinamento colaborativo das propostas)
> 4. Apenas após esse processo de refinamento colaborativo, as saídas são combinadas de forma ponderada para produzir o resultado final (relatório integrado)

## A Perda Auxiliar Avançada

Para garantir que o sistema MoC treine de forma estável e que a colaboração seja eficaz, uma perda auxiliar multifacetada é crucial. Ela é composta por dois elementos:

1. **Perda de Balanceamento (Balance Loss):** Semelhante aos MoEs padrão, esta perda incentiva o roteador a distribuir os tokens de forma relativamente uniforme entre todos os experts, evitando o colapso onde apenas alguns experts são utilizados.

2. **Perda de Diversidade (Diversity Loss):** Esta é a inovação principal da perda auxiliar do MoC. Ela é calculada sobre a **entropia dos pesos da atenção cruzada** durante a etapa de fusão colaborativa. Ao maximizar essa entropia, a perda incentiva a criação de padrões de colaboração ricos e diversificados, impedindo que os experts "viciem" em atender sempre aos mesmos pares da mesma maneira.

## Como Treinar

O treinamento do `LunarisCodex-MoC` é gerenciado através do script `train_moe.py` e um arquivo de configuração YAML.

1. **Crie um arquivo de configuração**, por exemplo `config_moc.yaml`. Destaque os parâmetros específicos do MoC:

    ```yaml
    # config_moc.yaml
    model:
      d_model: 768
      n_layers: 12
      n_heads: 12
      n_kv_heads: 12
      vocab_size: 50257
      max_seq_len: 1024
      # --- Parâmetros do MoC ---
      n_experts: 8          # Número total de experts a serem criados
      top_k: 2              # Número de experts a serem selecionados para colaboração
      aux_loss_weight: 0.01 # Peso da perda auxiliar (balanceamento + diversidade)

    data_dir: "data/"
    out_dir: "checkpoints/lunaris-moc-8e-2k"
    learning_rate: 3.0e-4
    max_steps: 600000
    batch_size: 16
    gradient_accumulation_steps: 4
    wandb_project: "lunaris-codex-moc"
    wandb_run_name: "moc-8-experts-2-topk"
    ```

2. **Inicie o treinamento** usando o seguinte comando:

    ```bash
    python train_moe.py config_moc.yaml
    ```

3. **Monitore no Weights & Biases:** Durante o treinamento, preste atenção especial às seguintes métricas:
    - `loss/main`: A perda de cross-entropy padrão. Ela mede o quão bem o modelo está prevendo o próximo token.
    - `loss/aux`: A perda auxiliar do MoC. Um valor estável aqui indica que o roteador está balanceando a carga e que a colaboração entre os experts é diversificada.
    - `perplexity`: Calculada apenas a partir de `loss/main`, é a principal métrica de desempenho do modelo de linguagem.

## Limitações e Considerações

Como um projeto experimental, o MoC tem trade-offs importantes:

- **Custo Computacional**: Processar todos os experts + atenção cruzada é mais caro que MoE padrão
- **Complexidade**: A arquitetura é mais complexa, com mais hiperparâmetros para ajustar
- **Experimentação**: Este é um protótipo conceitual - a eficácia prática ainda precisa ser validada

## Resumo da Arquitetura

**Conceito Central**: O LunarisCodex-MoC transforma os experts de "consultores isolados" em uma "equipe colaborativa".

**Arquitetura**:
- **Router Contextualization (Auto-Atenção)**: A "reunião preliminar" com todos os ministros, que permite ao presidente fazer uma escolha muito mais inteligente sobre quem deve formar a equipe
- **Collaborative Fusion (Atenção Cruzada)**: A "força-tarefa" onde os membros da equipe escolhida trabalham juntos, discutem e integram suas ideias para criar uma solução final muito mais forte e coesa

É por isso que essa arquitetura tem potencial. Ela imita um processo de resolução de problemas muito mais sofisticado e realista, onde a colaboração e o contexto são fundamentais para chegar a uma boa solução.
