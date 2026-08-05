# CORPUS ANALYSIS - Rubidium Transformer (500M)

> Comprehensive corpus size analysis for the Rubidium CUDA/C++ transformer training pipeline.
> Model: 505M params | Hardware: P100 16GB | Task: Spanish chatbot

---

## 1. Model Parameter Count Breakdown

### 1.1 Configuration

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Vocab size | V | 256 | Character-level (ASCII bytes) |
| Context length | T | 256 | Max sequence length (block_size) |
| Model dimension | D | 2048 | Embedding / hidden dimension |
| Attention heads | H | 32 | Number of attention heads |
| Head dimension | hd | 64 | D / H = 2048 / 32 |
| Transformer layers | L | 10 | Number of decoder blocks |
| FFN dimension | FF | 8192 | Feed-forward hidden dim (4x D) |

### 1.2 Parameter Breakdown (exact count from source: `main.cpp:109-113`)

```
COMPONENT                           FORMULA                     COUNT
─────────────────────────────────────────────────────────────────────
EMBEDDINGS
  Token Embedding                   V x D                       524,288
  Position Embedding                T x D                       524,288
  Embedding subtotal                                            1,048,576

TRANSFORMER LAYER (per layer, x10)
  LayerNorm 1 (weight + bias)       2 x D                       4,096
  Q projection (weight + bias)      D x D + D               4,198,404
  K projection (weight + bias)      D x D + D               4,198,404
  V projection (weight + bias)      D x D + D               4,198,404
  O projection (weight + bias)      D x D + D               4,198,404
  LayerNorm 2 (weight + bias)       2 x D                       4,096
  FFN W1 (weight + bias)            D x FF + FF             16,785,408
  FFN W2 (weight + bias)            FF x D + D             16,781,312
  Per-layer subtotal                                            50,368,512
  x10 layers                                                   503,685,120

OUTPUT HEAD
  Final LayerNorm (weight + bias)   2 x D                       4,096
  LM Head (weight + bias)           V x D + V                 524,544
  Output subtotal                                               528,640
─────────────────────────────────────────────────────────────────────
TOTAL PARAMETERS                                                ~505,262,336
```

**Exact total: 505,262,336 parameters (~505.3M)**

### 1.3 Parameter Distribution

| Component | Parameters | % of Total | VRAM (float32) |
|-----------|-----------|------------|-----------------|
| Token Embedding | 524,288 | 0.10% | 2.0 MB |
| Position Embedding | 524,288 | 0.10% | 2.0 MB |
| Attention (Q/K/V/O) x10 | 167,936,000 | 33.24% | 640.6 MB |
| FFN (W1/W2) x10 | 335,667,200 | 66.43% | 1,280.3 MB |
| LayerNorm (all) | 81,920 | 0.02% | 0.3 MB |
| LM Head | 524,544 | 0.10% | 2.0 MB |
| **Total Weights** | **505,258,240** | **100%** | **1,928.6 MB** |

> Note: The parameter count from `main.cpp` formula:
> `V*D*2 + T*D + L*(4*D*D + 8*D + 2*D*FF + 2*FF) + 2*D + V*D + V`

---

## 2. Memory Analysis (P100 16GB Constraints)

### 2.1 VRAM Budget Breakdown

| Component | Calculation | Size |
|-----------|-------------|------|
| **Model Weights** | 505M x 4 bytes | **1,928 MB (1.88 GB)** |
| **Gradients** | 505M x 4 bytes | **1,928 MB (1.88 GB)** |
| **Adam Moment m** | 505M x 4 bytes | **1,928 MB (1.88 GB)** |
| **Adam Moment v** | 505M x 4 bytes | **1,928 MB (1.88 GB)** |
| **Subtotal (Optimizer State)** | | **7,712 MB (7.53 GB)** |

### 2.2 Activation Memory (BS=2, T=256, D=2048)

| Activation | Size per layer | 10 layers |
|------------|---------------|-----------|
| x_emb (input) | 2 x 256 x 2048 = 1.0 MB | - |
| h1, q, k, v (each) | 2 x 256 x 2048 = 1.0 MB | 4 x 10 = 40 MB |
| att, att_p (each) | 2 x 32 x 256 x 256 = 4.0 MB | 2 x 10 = 80 MB |
| ao, x1, h2 (each) | 2 x 256 x 2048 = 1.0 MB | 3 x 10 = 30 MB |
| fi (FFN) | 2 x 256 x 8192 = 4.0 MB | 10 x 4 = 40 MB |
| ln_mean, ln_inv_std | 2 x 256 = 2 KB | 4 x 10 = 80 KB |
| **Per-layer subtotal** | | **~19 MB** |
| **All layers subtotal** | | **~190 MB** |
| hf, logits | 2 x 256 x 2048 + 2 x 256 x 256 = 1.13 MB | 1.13 MB |
| **Total Activations** | | **~191 MB** |

### 2.3 Backward Pass Temporary Allocations

The backward pass (`model.cu:240-336`) allocates temporary buffers per layer:

| Temporary | Size | Notes |
|-----------|------|-------|
| d_dlogits | 2 x 256 x 256 = 0.5 MB | CE backward |
| d_dx | 2 x 256 x 2048 = 4.0 MB | LM head backward |
| d_ln_f | 2 x 256 x 2048 = 4.0 MB | Final LN backward |
| Per layer: d_fi, d_h2, d_ln2, d_ao, d_h1, d_ln1 | ~24 MB | Allocated/freed per layer |

**Peak backward allocation: ~40 MB** (d_dlogits + d_dx + d_ln_f + 1 layer temps)

### 2.4 Total VRAM Summary

| Category | Size |
|----------|------|
| Model weights | 1,928 MB |
| Gradients | 1,928 MB |
| Adam m + v | 3,856 MB |
| Activations (forward) | 191 MB |
| Backward temporaries | 40 MB |
| cuBLAS workspace | ~50 MB |
| CUDA context + misc | ~300 MB |
| **TOTAL ESTIMATED** | **~8,293 MB (8.1 GB)** |
| **Available on P100** | **16,384 MB (16 GB)** |
| **Remaining headroom** | **~8,091 MB (7.9 GB)** |

**VERDICT: The model fits comfortably within P100 16GB with ~8GB headroom.**

### 2.5 Why BS=2 is Necessary

The batch size of 2 is dictated by activation memory scaling:

| BS | Activation Memory | Total VRAM | Fits P100? |
|----|-------------------|------------|------------|
| 1 | ~96 MB | ~8.0 GB | YES |
| 2 | ~191 MB | ~8.1 GB | YES |
| 4 | ~382 MB | ~8.3 GB | YES |
| 8 | ~764 MB | ~8.7 GB | YES |
| 16 | ~1,528 MB | ~9.5 GB | YES |
| 32 | ~3,056 MB | ~11.0 GB | YES |
| 64 | ~6,112 MB | ~14.1 GB | TIGHT |

> BS=2 is conservative. BS=8 or even BS=16 would fit. The gradient accumulation (GA=16) compensates to achieve effective batch=32.

---

## 3. Corpus Size Analysis

### 3.1 Current Corpus Statistics

Based on analysis of `D:\Inteligente\rubidium-api\corpus\`:

| Metric | Value |
|--------|-------|
| Total files | 15 (CorpusAgent1-15.txt) |
| Non-empty files | 11 |
| Empty files | 4 (Agent3, 4, 13, 15) |
| Total characters | ~984,044 |
| Total bytes | ~984 KB |
| U:/B: pair format | `U: question / B: answer` |
| Estimated pairs | ~2,500-3,741 |
| Avg pair length | ~250-350 chars |

**File size distribution:**

```
CorpusAgent1.txt   119,466 bytes
CorpusAgent2.txt   253,555 bytes  (largest)
CorpusAgent5.txt    90,501 bytes
CorpusAgent6.txt   217,474 bytes
CorpusAgent7.txt    31,321 bytes
CorpusAgent8.txt    12,868 bytes
CorpusAgent9.txt    23,288 bytes
CorpusAgent10.txt   22,514 bytes
CorpusAgent11.txt  135,465 bytes
CorpusAgent12.txt   60,934 bytes
CorpusAgent14.txt   52,568 bytes
```

### 3.2 Token Count Analysis

With character-level tokenization (V=256, 1 char = 1 token):

```
Current corpus:     ~984,044 tokens
Available windows:  984,044 - 256 = ~983,788 unique starting positions
Training sequences per epoch: ~983,788
```

### 3.3 Training Data Requirements

#### Effective Tokens per Step

```
BS = 2, GA = 16, T = 256
Tokens per step = BS x GA x T = 2 x 16 x 256 = 8,192 tokens
```

#### Total Tokens Processed

```
200,000 steps x 8,192 tokens/step = 1,638,400,000 tokens (~1.64B)
```

#### Epochs (Repeats)

```
Total tokens processed / Unique tokens = 1,638,400,000 / 984,044 = ~1,665 epochs
```

**CRITICAL: With the current corpus, the model would see each character ~1,665 times during training. This is extreme overfitting territory.**

---

## 4. Chinchilla Scaling Laws Analysis

### 4.1 Theoretical Optimal (Chinchilla, 2022)

The Chinchilla scaling law states optimal training uses:

```
Tokens_optimal = 20 x Parameters
Tokens_optimal = 20 x 505,000,000 = 10,100,000,000 (~10.1B tokens)
```

For a 505M parameter model, the Chinchilla-optimal corpus size is **~10 billion tokens**.

### 4.2 Practical Scaling (Industry Practice)

Real-world models often train on more tokens than Chinchilla-optimal:

| Model | Params | Tokens | Tokens/Param | Source |
|-------|--------|--------|--------------|--------|
| GPT-2 Small | 117M | 40B | 342x | OpenAI |
| GPT-2 Medium | 345M | 40B | 116x | OpenAI |
| GPT-2 Large | 762M | 40B | 52x | OpenAI |
| LLaMA-7B | 7B | 1T | 143x | Meta |
| LLaMA-13B | 13B | 1T | 77x | Meta |
| Phi-2 | 2.7B | 1.4T | 519x | Microsoft |

**For a 505M model, practical training targets range from 1B to 40B tokens.**

### 4.3 Minimum Viable Corpus

Based on empirical evidence from similar-scale models:

```
ABSOLUTE MINIMUM:     100M tokens  (10x Chinchilla, prevents total memorization)
RECOMMENDED MINIMUM:  500M tokens  (50x Chinchilla, reasonable generalization)
RECOMMENDED:            2B tokens  (200x Chinchilla, good quality)
IDEAL:                  5B tokens  (500x Chinchilla, excellent quality)
CHINCHILLA OPTIMAL:   10B tokens  (theoretical optimum)
```

### 4.4 Current vs Required

```
Current corpus:       ~984,000 tokens
Absolute minimum:     ~100,000,000 tokens (100M)
Deficit:              ~99x short of minimum
Recommended:          ~2,000,000,000 tokens (2B)
Deficit:              ~2,032x short of recommended
```

---

## 5. Corpus Size Recommendations

### 5.1 Three-Tier Recommendation

#### TIER 1: MINIMUM VIABLE (Quick Prototype)

| Metric | Value |
|--------|-------|
| **Characters** | 10,000,000 |
| **Tokens** | 10,000,000 |
| **U:/B: Pairs** | ~30,000 - 40,000 |
| **Files (10-file structure)** | 10 files x ~1M chars each |
| **Total corpus size** | ~10 MB |
| **Epochs at 200K steps** | ~166 epochs |
| **Expected quality** | Basic pattern recognition, limited generalization |

**Rationale:** 10M tokens is 100x the current corpus. The model can learn basic Spanish patterns, common phrases, and simple Q&A structures. However, it will still overfit significantly and memorize training examples.

**How to achieve:** Generate or collect 10x more U:/B: pairs. At ~300 chars/pair, this requires ~33,333 pairs.

#### TIER 2: RECOMMENDED (Good Quality)

| Metric | Value |
|--------|-------|
| **Characters** | 200,000,000 |
| **Tokens** | 200,000,000 |
| **U:/B: Pairs** | ~600,000 - 800,000 |
| **Files (10-file structure)** | 10 files x ~20M chars each |
| **Total corpus size** | ~200 MB |
| **Epochs at 200K steps** | ~8.3 epochs |
| **Expected quality** | Good conversational ability, reasonable generalization |

**Rationale:** 200M tokens allows the model to see each unique training example ~8 times during 200K steps. This is in the sweet spot for learning without extreme memorization. The model can learn diverse conversation patterns, domain knowledge, and Spanish language nuances.

**How to achieve:** Combine the current corpus with synthetic data generation, web scraping of Spanish conversational data, and curated datasets.

#### TIER 3: IDEAL (Excellent Quality)

| Metric | Value |
|--------|-------|
| **Characters** | 2,000,000,000 |
| **Tokens** | 2,000,000,000 |
| **U:/B: Pairs** | ~6,000,000 - 8,000,000 |
| **Files (10-file structure)** | 10 files x ~200M chars each |
| **Total corpus size** | ~2 GB |
| **Epochs at 200K steps** | ~0.83 epochs (under-trained) |
| **Expected quality** | Excellent conversational ability, strong generalization |

**Rationale:** 2B tokens approaches the Chinchilla-optimal regime. The model sees each example less than once, forcing it to learn generalizable patterns rather than memorize. This requires reducing training steps or increasing corpus to match.

**How to achieve:** Large-scale data collection from Spanish internet, books, conversations, and synthetic augmentation.

### 5.2 Corpus Size vs Training Quality Matrix

```
Corpus Size    Tokens      Epochs@200K    Quality Rating    Use Case
──────────────────────────────────────────────────────────────────────
984K (current) 984K        1,665x         POOR              Overfitting demo
5M             5M          333x           POOR              Still heavy overfitting
10M            10M         166x           BASIC             Proof of concept
50M            50M         33x            FAIR              Basic chatbot
100M           100M        16x            GOOD              Functional chatbot
200M           200M        8x             GOOD+             Production prototype
500M           500M        3.3x           VERY GOOD         Near-optimal
1B             1B          1.6x           EXCELLENT         High quality
2B             2B          0.8x           EXCELLENT+        Needs more steps
5B             5B          0.3x           OPTIMAL           Needs 600K+ steps
10B            10B         0.16x          CHINCHILLA        Needs 2M+ steps
```

---

## 6. Training Time Estimates

### 6.1 Speed Benchmark (P100)

Based on the architecture and code analysis:

| Metric | Estimate |
|--------|----------|
| Forward pass (BS=2, T=256) | ~0.3-0.5 seconds |
| Backward pass (BS=2, T=256) | ~0.6-1.0 seconds |
| Optimizer step | ~0.1-0.2 seconds |
| **Total per step** | **~1.0-1.7 seconds** |
| **Estimated steps/second** | **0.6-1.0 steps/s** |

> Note: The actual speed depends on GPU utilization, memory bandwidth, and kernel efficiency. The P100 has 9.3 TFLOPS FP32 and 732 GB/s memory bandwidth.

### 6.2 Training Duration by Corpus Size

| Corpus | Tokens | Steps Needed | Duration (at 0.8 sps) | Kaggle Sessions |
|--------|--------|-------------|----------------------|-----------------|
| 10M | 10M | 200K | ~69 hours | 4.3 sessions |
| 100M | 100M | 200K | ~69 hours | 4.3 sessions |
| 200M | 200M | 200K | ~69 hours | 4.3 sessions |
| 2B | 2B | 200K | ~69 hours | 4.3 sessions |
| 2B | 2B | 600K (optimal) | ~208 hours | 13 sessions |

> Note: Training duration is fixed at 200K steps regardless of corpus size. More data means fewer epochs, not more training time.

### 6.3 Kaggle Session Constraints

| Constraint | Value |
|------------|-------|
| Max session length | ~12 hours (36,000 seconds) |
| Max sessions/day | ~2 (GPU quota dependent) |
| Checkpoint saving | Every 5,000 steps (~1.7 hours) |
| Steps per session (at 0.8 sps) | ~28,800 steps |
| Sessions for 200K steps | ~7 sessions |
| Days to complete | ~3.5 days |

---

## 7. Quality vs Quantity Trade-offs

### 7.1 The Overfitting Problem

With the current corpus (984K tokens), the model will:

1. **Memorize training examples**: The model has 505M parameters but only 984K tokens. The parameter-to-token ratio is 515:1, meaning each parameter has ~515 tokens to learn from. This guarantees memorization.

2. **Generate verbatim outputs**: The model will reproduce training examples almost exactly, especially for common patterns.

3. **Fail to generalize**: New questions not in the training set will produce incoherent or memorized responses.

4. **High training loss plateau**: The loss will decrease rapidly as the model memorizes, then plateau as it runs out of unique patterns to learn.

### 7.2 The Quality-Quantity Curve

```
Quality
  ^
  |                              ___________  <-- Diminishing returns
  |                         ____/
  |                    ____/
  |               ____/
  |          ____/
  |     ____/
  |____/
  +-----------------------------------------> Corpus Size
  984K  10M  50M  100M  200M  500M  1B  2B

  Current: 984K (far left, poor quality)
  Minimum: 10M (basic functionality)
  Recommended: 200M (good quality)
  Ideal: 2B (excellent quality)
```

### 7.3 Data Quality Considerations

For a Spanish chatbot, corpus quality matters as much as quantity:

| Factor | Impact | Recommendation |
|--------|--------|----------------|
| **Diversity of topics** | HIGH | Cover daily life, knowledge, opinions, emotions |
| **Conversation naturalness** | HIGH | Use real conversational data, not robotic Q&A |
| **Spanish dialect variety** | MEDIUM | Include Latin American and Spain variants |
| **Response length variety** | MEDIUM | Mix short and long answers |
| **Formal/informal register** | MEDIUM | Include both "tú" and "usted" forms |
| **Domain coverage** | HIGH | General knowledge, not just one topic |
| **Deduplication** | HIGH | Remove near-duplicate pairs |
| **Consistency** | MEDIUM | Same persona throughout |

### 7.4 Synthetic Data Generation Strategy

To reach the recommended 200M tokens from the current 984K:

```
Current corpus:           984,044 tokens
Target:                 200,000,000 tokens
Gap:                    199,015,956 tokens

Generation strategy:
1. Template expansion:    ~20M tokens (paraphrase existing pairs)
2. Topic expansion:       ~50M tokens (new topics from seed patterns)
3. Web scraping:          ~80M tokens (Spanish forums, Wikipedia, books)
4. LLM-assisted:          ~50M tokens (generate new Q&A pairs using GPT-4/Claude)
```

---

## 8. 10-File Structure Recommendations

### 8.1 Current Structure Analysis

The current 15-file structure (CorpusAgent1-15.txt) has issues:
- 4 files are empty (0 bytes)
- Inconsistent file sizes (12KB to 253KB)
- No clear organization by topic or quality

### 8.2 Recommended 10-File Structure

For optimal training with the 10-file target:

```
corpus/
├── 01_daily_conversations.txt    (~20% of corpus)
├── 02_knowledge_facts.txt        (~15% of corpus)
├── 03_emotional_support.txt      (~10% of corpus)
├── 04_creative_writing.txt       (~10% of corpus)
├── 05_technology_science.txt     (~15% of corpus)
├── 06_health_wellness.txt        (~10% of corpus)
├── 07_culture_entertainment.txt  (~10% of corpus)
├── 08_education_learning.txt     (~5% of corpus)
├── 09_opinions_debate.txt        (~5% of corpus)
└── 10_meta_conversation.txt      (~5% of corpus)
```

### 8.3 File Size Targets

| File | Target Size | Tokens | Pairs | Purpose |
|------|-------------|--------|-------|---------|
| 01_daily_conversations.txt | 40 MB | 40M | ~120K | Everyday chat patterns |
| 02_knowledge_facts.txt | 30 MB | 30M | ~90K | Factual Q&A |
| 03_emotional_support.txt | 20 MB | 20M | ~60K | Empathetic responses |
| 04_creative_writing.txt | 20 MB | 20M | ~60K | Stories, poetry, humor |
| 05_technology_science.txt | 30 MB | 30M | ~90K | Tech and science topics |
| 06_health_wellness.txt | 20 MB | 20M | ~60K | Health advice |
| 07_culture_entertainment.txt | 20 MB | 20M | ~60K | Movies, music, culture |
| 08_education_learning.txt | 10 MB | 10M | ~30K | Educational content |
| 09_opinions_debate.txt | 10 MB | 10M | ~30K | Opinions and discussions |
| 10_meta_conversation.txt | 10 MB | 10M | ~30K | Meta-chat, greetings |
| **TOTAL** | **210 MB** | **210M** | **~630K** | |

### 8.4 File Naming Convention

Use descriptive, numbered names for clarity:
- `01_` through `10_` prefix for load order
- Descriptive topic names
- Consistent `.txt` extension

### 8.5 Content Format

Maintain the existing U:/B: format for compatibility:

```
U: ¿Qué es la fotosíntesis?
B: La fotosíntesis es el proceso por el cual las plantas convierten la luz solar, agua y dióxido de carbono en glucosa y oxígeno. Es fundamental para la vida en la Tierra.

U: ¿Por qué el cielo es azul?
B: El cielo es azul debido a la dispersión de Rayleigh. La luz solar se dispersa más en longitudes de onda cortas (azul) al interactsuar con las moléculas de la atmósfera.
```

---

## 9. Implementation Roadmap

### Phase 1: Minimum Viable (Week 1-2)

```
Target: 10M tokens (10x current)
Actions:
1. Generate 30,000 new Q&A pairs using template expansion
2. Scrape 5M tokens from Spanish Wikipedia
3. Clean and deduplicate all data
4. Organize into 10-file structure
5. Train and evaluate
```

### Phase 2: Recommended (Week 3-6)

```
Target: 200M tokens (200x current)
Actions:
1. Generate 200,000 Q&A pairs using LLM assistance (GPT-4/Claude)
2. Scrape 100M tokens from Spanish forums and Reddit
3. Add 50M tokens from Spanish books (public domain)
4. Quality filtering and deduplication
5. Train and evaluate with validation split
```

### Phase 3: Ideal (Week 7-12)

```
Target: 2B tokens (2000x current)
Actions:
1. Large-scale web scraping (Common Crawl Spanish subset)
2. Book corpus digitization
3. Conversation dataset collection
4. Synthetic data generation at scale
5. Quality filtering pipeline
6. Train with extended steps (500K-1M)
```

---

## 10. Key Metrics to Track

### 10.1 During Training

| Metric | Target | Warning Sign |
|--------|--------|--------------|
| Training loss | Decreasing steadily | Plateau before 50K steps |
| Loss variance | Low variance | High oscillation |
| Perplexity | < 10 at end | > 50 at end |
| Gradient norm | < 1.0 (clipped) | Frequent clipping |

### 10.2 During Evaluation

| Metric | Target | Warning Sign |
|--------|--------|--------------|
| Response coherence | Grammatically correct | Gibberish output |
| Topic relevance | Stays on topic | Random topic switching |
| Factual accuracy | Mostly correct | Frequent hallucination |
| Conversational flow | Natural exchanges | Robotic responses |
| Diversity | Varied responses | Repetitive outputs |

### 10.3 Corpus Quality Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Deduplication rate | < 5% duplicates | Exact and fuzzy matching |
| Avg response length | 50-300 chars | Statistics |
| Topic diversity | > 20 topics | Topic modeling |
| Language quality | > 90% grammatical | Grammar checker |
| Persona consistency | Same voice throughout | Manual review |

---

## 11. Summary and Recommendations

### 11.1 Critical Findings

1. **Current corpus is severely inadequate**: 984K tokens vs. 200M+ recommended = 200x deficit
2. **Model will overfit massively**: 505M params on 984K tokens guarantees memorization
3. **Training is wasteful**: 200K steps on tiny corpus = 1,665 epochs of repetition
4. **Hardware is sufficient**: P100 16GB has ~8GB headroom for this model
5. **Architecture is sound**: The transformer implementation is functional (with noted backward pass issues)

### 11.2 Priority Recommendations

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Expand corpus to 100M+ tokens | CRITICAL | High |
| **P0** | Fix attention backward pass | CRITICAL | Medium |
| **P1** | Add validation split (5-10%) | HIGH | Low |
| **P1** | Implement dropout (p=0.1) | HIGH | Low |
| **P2** | Organize 10-file structure | MEDIUM | Low |
| **P2** | Add data quality filtering | MEDIUM | Medium |
| **P3** | Mixed precision training | MEDIUM | Medium |
| **P3** | Flash Attention integration | LOW | High |

### 11.3 Corpus Size Decision Matrix

```
IF your goal is:                    THEN target corpus size:
─────────────────────────────────────────────────────────────
Quick prototype / demo              10M tokens (10x current)
Functional chatbot                  100M tokens (100x current)
Production-quality chatbot          200M tokens (200x current)
Competitive with GPT-2              2B tokens (2000x current)
Chinchilla-optimal training         10B tokens (10,000x current)
```

### 11.4 Final Recommendation

**Expand the corpus to at least 200M tokens (210 MB, ~630,000 U:/B: pairs) organized in 10 topic-based files.** This represents a 200x increase from the current corpus and will enable the model to learn generalizable conversation patterns rather than memorizing training examples.

The most efficient path to reach this target:
1. Use LLM assistance (GPT-4/Claude) to generate 150M tokens of high-quality Spanish Q&A
2. Supplement with 50M tokens from web-scraped Spanish conversational data
3. Quality filter and deduplicate
4. Organize into the recommended 10-file structure

---

*Document generated from source code analysis of `D:\Inteligente\rubidium-train\src\`*
*Model config: V=256, T=256, D=2048, H=32, L=10, FF=8192*
*Parameters: 505,262,336 (~505.3M)*
*Hardware target: NVIDIA P100 16GB (Kaggle)*
