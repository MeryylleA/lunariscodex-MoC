# Mixture of Collaborative Experts (MoC)

Uma arquitetura avançada de Mixture of Experts com colaboração entre especialistas

---

## 📋 Visão Geral

A Mixture of Collaborative Experts (MoC) é uma evolução do tradicional Mixture of Experts (MoE) que introduz colaboração inteligente entre os especialistas selecionados. Ao invés de simplesmente combinar as saídas dos experts de forma independente, a MoC permite que os experts colaborem através de cross-attention antes da fusão final.

### Diferenças Fundamentais do MoE Tradicional

| Aspecto | MoE Tradicional | MoC (Nossa Implementação) |
|---------|----------------|---------------------------|
| Seleção de Experts | Router simples com softmax | Router contextualizado com self-attention |
| Colaboração | Nenhuma - experts independentes | Cross-attention entre experts selecionados |
| Auxiliary Loss | Load balancing básico | Diversity + Balance combinados |
| Estabilidade | Problemas de renormalização | Softmax direto nos logits |

---

## 🏗️ Arquitetura Detalhada

### Componentes Principais

#### 1. Router Contextualization

```python
# Self-attention sobre TODOS os expert outputs
contextualized_experts, _ = self.router_self_attn(
    expert_flat, expert_flat, expert_flat
)
```

**Por que isso é importante:**
- O router não decide baseado apenas no input token
- Considera as saídas de todos os experts para fazer uma seleção mais informada
- Permite routing adaptativo baseado no contexto completo

#### 2. Top-K Selection com Temperature

```python
# Temperature scaling para controle de distribuição
routing_logits = routing_logits / self.router_temperature

# Seleção estável dos top-k experts
topk_logits, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
topk_probs = F.softmax(topk_logits, dim=-1)
```

**Vantagens:**
- Numericamente mais estável que renormalização manual
- Temperature permite controlar sharpness da distribuição
- Evita problemas de divisão por zero

#### 3. Collaborative Fusion

```python
# Cross-attention entre experts selecionados
collaborative_outputs, collab_attn = self.collab_cross_attn(
    selected_flat, selected_flat, selected_flat
)

# Refinement adicional
refined_outputs = self.collab_ffn(collaborative_outputs) + collaborative_outputs
```

**O diferencial:**
- Experts selecionados "conversam" entre si via cross-attention
- Refinement FFN adicional para melhorar a colaboração
- Residual connections para estabilidade de treino

#### 4. Advanced Auxiliary Loss

```python
def compute_diversity_loss(self, cross_attn_weights, routing_probs):
    # Diversity: maximizar entropia da cross-attention
    attn_entropy = -torch.sum(
        cross_attn_weights * torch.log(cross_attn_weights + 1e-8), dim=-1
    ).mean()
    diversity_loss = -attn_entropy
    
    # Balance: minimizar variância do uso dos experts
    expert_usage = routing_probs.mean(dim=[0, 1])
    balance_loss = torch.var(expert_usage)
    
    return 0.01 * diversity_loss + 0.01 * balance_loss
```

---

## 🔄 Fluxo de Processamento

### Step-by-Step

1. **Input Processing**
   - Recebe tokens: (batch_size, seq_len, d_model)
   - Processa através de TODOS os experts em paralelo

2. **Router Contextualization**
   - Self-attention sobre todas as saídas dos experts
   - Gera representação contextualizada para routing

3. **Expert Selection**
   - Aplica temperature scaling nos routing logits
   - Seleciona top-k experts via torch.topk()
   - Calcula probabilidades com softmax estável

4. **Collaborative Fusion**
   - Cross-attention entre experts selecionados
   - Refinement via FFN adicional
   - Residual connections para estabilidade

5. **Final Output**
   - Combinação ponderada das saídas colaborativas
   - Projeção final + auxiliary loss

### Dimensões dos Tensors

```
Input: (B, S, d_model)
Expert Outputs: (B, S, n_experts, d_model)
Contextualized: (B*S, n_experts, d_model)
Top-K Selection: (B, S, top_k, d_model)
Final Output: (B, S, d_model)
```

---

## ⚙️ Configuração

### Parâmetros Principais

```python
@dataclass
class LunarisCodexConfig:
    # MoC Specific Parameters
    n_experts: int = 8                    # Número total de experts
    top_k: int = 2                        # Quantos experts selecionar
    aux_loss_weight: float = 1e-2         # Peso da auxiliary loss
    router_temperature: float = 1.0       # Temperature para routing
```

### Recomendações de Configuração

| Parâmetro | Valor Recomendado | Justificativa |
|-----------|------------------|---------------|
| n_experts | 8-16 | Balance entre capacidade e eficiência |
| top_k | 2-4 | Permite colaboração sem overhead excessivo |
| aux_loss_weight | 1e-2 a 1e-3 | Suficiente para regularização |
| router_temperature | 0.5-2.0 | 1.0 = neutro, <1.0 = mais sharp, >1.0 = mais suave |

---

## 🧮 Análise Matemática

### Complexidade Computacional

**Forward Pass:**
- Expert computation: O(B × S × n_experts × d_model²)
- Router self-attention: O(B × S × n_experts² × d_model)
- Cross-attention: O(B × S × top_k² × d_model)
- **Total: O(B × S × n_experts × d_model²) (dominante)**

**Comparação com MoE tradicional:**
- MoE: O(B × S × top_k × d_model²)
- MoC: O(B × S × n_experts × d_model²) (durante treino)
- **Trade-off: Maior custo computacional por melhor qualidade**

### Auxiliary Loss Breakdown

**Diversity Loss: Encoraja padrões diversos na cross-attention**
- Previne collapse dos experts
- Maximiza entropia das attention weights

**Balance Loss: Garante uso equilibrado dos experts**
- Minimiza variância do expert usage
- Evita que alguns experts sejam ignorados

---

## 🚀 Vantagens da MoC

### 1. Melhor Especialização
- Experts colaboram ao invés de competir
- Cada expert pode focar em aspectos específicos
- Combinação inteligente de conhecimentos

### 2. Routing Contextualizado
- Decisões de routing mais informadas
- Considera output de todos os experts
- Adaptativo ao contexto atual

### 3. Estabilidade de Treino
- Auxiliary loss bem balanceada
- Softmax numericamente estável
- Residual connections para gradientes

### 4. Flexibilidade
- Temperature permite tuning fino
- Configurável para diferentes tarefas
- Escalável para mais experts

---

## 🔧 Implementação

### Integração no Transformer

```python
class Block(nn.Module):
    def __init__(self, config):
        # ... attention layers ...
        
        if config.n_experts is not None and config.n_experts > 0:
            self.feed_forward = CollaborativeExpertsModule(config)
            self.is_moe = True
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False
```

### Training Loop Considerations

```python
# Durante o forward pass
logits, loss, past_key_values = model(idx, targets=targets)

if loss is not None:
    total_loss, main_loss, aux_loss = loss
    # total_loss já inclui auxiliary loss ponderada
    total_loss.backward()
```

---

## 📊 Monitoramento e Debug

### Métricas Importantes

**Expert Usage Distribution**
```python
expert_usage = routing_probs.mean(dim=[0, 1])
print(f"Expert usage variance: {torch.var(expert_usage):.4f}")
```

**Auxiliary Loss Components**
```python
print(f"Diversity loss: {diversity_loss:.4f}")
print(f"Balance loss: {balance_loss:.4f}")
```

**Routing Entropy**
```python
routing_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1).mean()
print(f"Routing entropy: {routing_entropy:.4f}")
```

---

## 🎯 Casos de Uso

### Quando Usar MoC

✅ **Ideal para:**
- Tarefas que requerem diferentes tipos de raciocínio
- Modelos grandes onde eficiência é importante
- Cenários com dados diversificados
- Quando você quer melhor interpretabilidade

❌ **Evitar quando:**
- Modelos muito pequenos (overhead não compensa)
- Tarefas muito específicas/homogêneas
- Recursos computacionais muito limitados
- Prototipagem rápida (use FFN padrão primeiro)

---

## 🔬 Experimentos e Tuning

### Hyperparameter Sweep Sugerido

```python
# Configurações para testar
configs = [
    {"n_experts": 8, "top_k": 2, "router_temperature": 1.0},
    {"n_experts": 8, "top_k": 3, "router_temperature": 0.7},
    {"n_experts": 16, "top_k": 4, "router_temperature": 1.2},
]
```

### Ablation Studies
- **Sem Router Contextualization:** Remove self-attention do router
- **Sem Collaborative Fusion:** Remove cross-attention entre experts
- **Auxiliary Loss Components:** Teste diversity vs balance separadamente

---

## 📚 Referências e Inspirações

### Papers Relacionados
- **Switch Transformer:** Fundação do MoE moderno
- **GLaM:** Scaling MoE para modelos gigantes
- **Expert Choice:** Routing improvements

### Diferenças da Nossa Implementação
- Router contextualization com self-attention
- Collaborative fusion via cross-attention
- Auxiliary loss combinada (diversity + balance)
- Integração limpa com arquitetura Llama-style

---

## 🐛 Troubleshooting

### Problemas Comuns

**Auxiliary Loss Muito Alta**
- Reduza aux_loss_weight
- Verifique se diversity e balance estão balanceados

**Experts Não Sendo Usados**
- Aumente router_temperature
- Verifique inicialização dos pesos

**Instabilidade de Treino**
- Reduza learning rate
- Verifique gradient clipping

**Overfitting**
- Aumente dropout
- Reduza número de experts ou top_k

---

## 💡 Ideias para Extensões Futuras

- **Dynamic Top-K:** Ajustar top_k baseado no contexto
- **Hierarchical Experts:** Experts especializados em diferentes níveis
- **Memory-Augmented Routing:** Router com memória de decisões passadas
- **Multi-Scale Collaboration:** Cross-attention em diferentes escalas

---

**Criado por:** Francisco  
**Data:** Julho 2025  
**Versão:** 1.0

> "A ideia é simples: ao invés de experts competindo, eles colaboram. E essa colaboração acontece através de cross-attention, permitindo que cada expert refine sua saída baseado no que os outros experts estão 'pensando'."
