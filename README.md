# LunarisCodex-MoC

*Um Modelo de Linguagem Experimental com um Módulo de Experts Colaborativos.*

---

## Visão Geral (Overview)

`LunarisCodex-MoC` é um modelo de linguagem grande (LLM) funcional, no estilo Llama, que serve como uma base de testes para arquiteturas de redes neurais de ponta. O projeto substitui a camada Feed-Forward Network (FFN) padrão dos blocos Transformer por uma implementação inovadora de **Mixture of Collaborative Experts (MoC)**. Esta arquitetura visa superar as limitações dos modelos Mixture-of-Experts (MoE) tradicionais, permitindo que os experts interajam e refinem suas saídas coletivamente antes da fusão final.

## A Evolução: Do MoE ao MoC

O projeto começou como uma exploração dos modelos MoE padrão, que utilizam um roteador para enviar cada token para um único expert (`k=1`), seguindo a lógica do Switch Transformer. Embora eficientes, os MoEs tradicionais têm uma limitação fundamental: **os experts trabalham em completo isolamento**. A saída final é simplesmente o resultado do expert escolhido, sem qualquer sinergia ou refinamento entre eles.

A questão que impulsionou este projeto foi: "E se os experts pudessem colaborar?". O MoC (Mixture of Collaborative Experts) é a resposta a essa pergunta. A arquitetura foi descoberta durante uma série de experimentos com IAs de ponta, onde uma implementação de referência gerada por um modelo misterioso e altamente avançado (apelidado de "ryo / GPT-5-Alpha") demonstrou uma abordagem fundamentalmente nova. O MoC move-se para além do simples roteamento, introduzindo mecanismos para que os experts interajam de forma significativa, tornando a camada de FFN um processo dinâmico e colaborativo.

## Inovações Arquiteturais do MoC

O `CollaborativeExpertsModule` é o coração do `LunarisCodex-MoC`. Ele introduz duas inovações principais que o distinguem dos MoEs convencionais.

### 1. Router Contextualization (Roteamento Contextualizado)

> **O Problema:** Roteadores MoE padrão tomam decisões com base apenas na representação do token de entrada, sem qualquer conhecimento do que os diferentes experts "pensam" sobre esse token. A decisão é cega.
>
> **A Solução MoC:** O roteador do MoC é muito mais sofisticado. O fluxo de decisão é o seguinte:
> 1.  O token de entrada é enviado para **todos os experts** em paralelo, e a saída de cada um é calculada.
> 2.  Uma camada de **auto-atenção (self-attention)** é aplicada sobre esses *outputs* dos experts. Isso permite que o roteador entenda as relações e os acordos/desacordos entre as "opiniões" dos experts.
> 3.  Com base nesse resumo contextual rico, o roteador toma uma decisão de roteamento top-k muito mais informada, selecionando os experts mais relevantes com base em uma compreensão holística.

### 2. Collaborative Fusion (Fusão Colaborativa)

> **O Problema:** Em MoEs padrão que usam `k > 1`, as saídas dos experts selecionados são simplesmente combinadas através de uma soma ponderada. Não há colaboração real.
>
> **A Solução MoC:** Uma vez que os `k` experts são selecionados, a colaboração começa:
> 1.  As saídas dos `k` experts escolhidos são empilhadas.
> 2.  Essas saídas são alimentadas em uma camada de **atenção cruzada (cross-attention)**, onde cada expert selecionado "atende" aos outputs dos outros experts selecionados.
> 3.  Isso permite que cada expert refine sua própria saída com base nas perspectivas de seus pares.
> 4.  Apenas após esse processo de refinamento colaborativo, as saídas são combinadas de forma ponderada para produzir o resultado final.

## A Perda Auxiliar Avançada

Para garantir que o sistema MoC treine de forma estável e que a colaboração seja eficaz, uma perda auxiliar multifacetada é crucial. Ela é composta por dois elementos:

1.  **Perda de Balanceamento (Balance Loss):** Semelhante aos MoEs padrão, esta perda incentiva o roteador a distribuir os tokens de forma relativamente uniforme entre todos os experts, evitando o colapso onde apenas alguns experts são utilizados. Ela é calculada sobre a variância do uso dos experts.

2.  **Perda de Diversidade (Diversity Loss):** Esta é a inovação principal da perda auxiliar do MoC. Ela é calculada sobre a **entropia dos pesos da atenção cruzada** durante a etapa de fusão colaborativa. Ao maximizar essa entropia, a perda incentiva a criação de padrões de colaboração ricos e diversificados, impedindo que os experts "viciem" em atender sempre aos mesmos pares da mesma maneira.

## Como Treinar

O treinamento do `LunarisCodex-MoC` é gerenciado através do script `train_moe.py` e um arquivo de configuração YAML.

1.  **Crie um arquivo de configuração**, por exemplo `config_moc.yaml`. Destaque os parâmetros específicos do MoC:

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

2.  **Inicie o treinamento** usando o seguinte comando:

    ```bash
    python train_moe.py config_moc.yaml
    ```

3.  **Monitore no Weights & Biases:** Durante o treinamento, preste atenção especial às seguintes métricas:
    *   `loss/main`: A perda de cross-entropy padrão. Ela mede o quão bem o modelo está prevendo o próximo token.
    *   `loss/aux`: A perda auxiliar do MoC. Um valor estável aqui indica que o roteador está balanceando a carga e que a colaboração entre os experts é diversificada.
    *   `perplexity`: Calculada apenas a partir de `loss/main`, é a principal métrica de desempenho do modelo de linguagem.

## Jornada do Projeto e Agradecimentos

Este projeto representa uma jornada de descoberta. A busca por arquiteturas MoE mais eficazes começou com experimentos na LLM Arena, comparando as saídas de vários modelos de ponta. A inspiração para a arquitetura MoC não foi teórica, mas sim empírica, derivada da análise de uma implementação de referência gerada por um modelo de IA de próxima geração, "GPT-5".
