# NBA Point Differential Predictions with Attention Mechanisms

**Can transformer-style attention mechanisms predict NBA game outcomes better than simple averaging, by capturing lineup synergies and matchup effects?**

## Key Finding

**No.** Attention mechanisms did not improve predictions. **XGBoost with simple features outperformed all neural network approaches.**

| Model | Features/Params | Test R² | Win Accuracy |
|-------|-----------------|---------|--------------|
| **XGBoost (concat)** | 160 features | **0.289** | 65.2% |
| Neural Net (Baseline) | 46K params | 0.203 | 66.2% |
| Neural Net (Attention) | 55K params | 0.200 | 66.7% |

This suggests that non-linear individual player effects exist, but pairwise synergy was not detectable when using the attention mechanism with this data and feature set.

---

## Motivation

I was curious to see if vector representations for NBA players could be learned through deep learning, and moreover if player chemistry and matchup advantages could be modeled through attention. In searching for prior work on this topic, I found the [NBA2Vec paper (2023)](https://arxiv.org/abs/2302.13386), which showed that player embeddings could capture positional and stylistic similarities. Their approach averaged embeddings across lineups, losing pairwise interaction information.

I hypothesized that attention mechanisms could do better by:
- Using **self-attention** to model within-team synergies (e.g., pick-and-roll partners)
- Using **cross-attention** to model offensive vs. defensive matchups

This project tests that hypothesis.

---

## Data

- **Source**: [NBA Play-by-Play Dataset](https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019) (Kaggle)
- **Scope**: 2015-2021 seasons (2022 partial)
- **Size**: ~4.4 million plays across 9,456 games
- **Players**: 1,181 with sufficient playing time (>100 possessions)
- **Target**: Home team point differential

### Feature Engineering

For each player, computed from play-by-play:

| Feature | Description |
|---------|-------------|
| `off_rating` | Points scored per 100 possessions |
| `def_rating` | Points allowed per 100 possessions |
| `net_rating` | Offensive - Defensive rating |
| `pts_per100` | Individual scoring rate |
| `reb_per100` | Individual rebounding rate |
| `possessions` | Total possessions played (volume) |
| `games_played` | Games appeared in (experience) |

All features normalized to zero mean, unit variance.

Note that `pts_per100` and `reb_per100` were only included in the 'rich' models.

---

## Methodology

### Approach Evolution

1. **Learned Embeddings Only** → R² ≈ 0 (failed completely)
   - Random embeddings have no signal without pretraining

2. **Stats as Features** → R² ≈ 0.20
   - Player quality encoded directly in stats
   - Tested at both team-level (pre-aggregated) and player-level (with attention)
   - Attention doesn't help at either level

3. **Hybrid: Stats + Learned Residuals** → R² ≈ 0.20
   - Small learned embeddings capture what stats miss

4. **Richer Stats (7 features)** → R² ≈ 0.20
   - Included `pts_per100` and `reb_per100` to see if individual stats make a difference
   - No meaningful improvement from additional features

### Model Architectures

**HybridBaseline**
```
Player Stats → Linear Projection → Add Residual Embedding
                                          ↓
                        Possession-Weighted Average per Team
                                          ↓
                              Concat [Away, Home] → MLP → Prediction
```

**HybridAttention**
```
Player Stats → Linear Projection → Add Residual Embedding
                                          ↓
                              Self-Attention (within team)
                                          ↓
                              Cross-Attention (offense vs defense)
                                          ↓
                        Possession-Weighted Average per Team
                                          ↓
                              Concat [Away, Home] → MLP → Prediction
```

---

## Results

### Neural Network Comparison

| Model | Parameters | Test R² | Test RMSE | Win Accuracy |
|-------|------------|---------|-----------|--------------|
| EmbeddingsBaseline | 83K | 0.001 | 26.7 | 54.9% |
| EmbeddingsAttention | 176K | 0.000 | 26.7 | 56.0% |
| StatsBaseline (team-level) | 5K | 0.202 | 12.7 | 66.5% |
| StatsAttention (team-level) | 42K | 0.193 | 12.7 | 66.2% |
| StatsPlayerBaseline | 9K | 0.201 | 12.7 | 65.0% |
| StatsPlayerAttention | 17K | 0.191 | 12.8 | 66.1% |
| HybridBaseline | 46K | 0.203 | 12.7 | 65.8% |
| HybridAttention | 55K | 0.199 | 12.7 | 65.6% |
| HybridBaseline (rich) | 46K | 0.203 | 12.7 | 66.2% |
| HybridAttention (rich) | 55K | 0.200 | 12.7 | 66.7% |

### XGBoost Comparison

As a sanity check, I compared against XGBoost with different feature representations:

| Model | Features | Test R² | Win Accuracy |
|-------|----------|---------|--------------|
| XGBoost (averaged) | 14 | 0.195 | 65.3% |
| XGBoost (summary) | 56 | 0.196 | 65.2% |
| **XGBoost (concat)** | 160 | **0.289** | 65.2% |

**XGBoost with concatenated player features achieves the best R².** The three feature modes:

- **Averaged**: Weighted mean of player stats per team (14 features)
- **Summary**: Mean, std, min, max of player stats per team (56 features)
- **Concat**: Top-10 players' stats concatenated (160 features)

The concat approach lets XGBoost learn non-linear rules like "if the best player has net rating > +8, add 3 points" — something the averaging-based neural networks cannot capture.

### What This Tells Us

| Model Type | What It Can Learn |
|------------|-------------------|
| Averaged (neural net) | Linear combination of player quality |
| XGBoost (concat) | Non-linear effects of player quality (superstars, roster depth) |
| Attention | Pairwise synergy between specific players |

The improvement from XGBoost concat shows that non-linear individual effects exist. However, attention didn't help, suggesting that pairwise synergy is either weak, requires different features to detect, or is washed out by game-level aggregation.

### Limitations

It's hard to conclude that lineup chemistry doesn't exist. But this approach didn't detect it. Possible explanations:

- **Feature limitations**: Efficiency stats may not capture the right signal (e.g., play-type compatibility)
- **Aggregation level**: Game-level averaging may obscure lineup-specific effects
- **Sample size**: ~6,600 games across 1,181 players may be insufficient for learning pairwise interactions
- **Architecture choices**: Different attention mechanisms might perform better
- **Synergy is real but small**: Effects may exist but be overwhelmed by individual player quality and noise

---

## Lineup-Level Experiment

To test whether game-level aggregation was hiding interaction effects, I ran a separate experiment at the **lineup level**:

- **Unit**: Each unique 5-player combination
- **Target**: Net rating (points scored - allowed per 100 possessions) across all minutes played together
- **Dataset**: 35,195 lineups with 50+ possessions each

### Lineup Results

| Model | Test R² | Sign Accuracy |
|-------|---------|---------------|
| LineupBaseline | 0.0225 | 54.6% |
| LineupAttention | 0.0200 | 54.6% |

**Both models have near-zero predictive power.** Knowing which 5 players are on the court explains only ~2% of their net rating variance.

### Interpretation

The low R² at lineup level could be due to:

1. **Uncontrolled opponent strength**: Each lineup's net rating averages across all opponents faced
2. **High inherent variance**: Basketball outcomes are noisy even controlling for personnel
3. **Chemistry effects may be real but small**: Overwhelmed by individual player quality and randomness

The game-level approach (R² ≈ 0.21, 66% win accuracy) works better because aggregating to full games averages out opponent effects and reduces noise. However, this aggregation may also wash out the lineup-specific chemistry effects I was trying to detect.

---

## Project Structure

```
nba-lineup-transformer/
├── README.md
├── requirements.txt
├── data/                                # Not included in repo (see Data Setup)
├── src/
│   │
│   │── # Original approach (embeddings only) - FAILED, R² ≈ 0
│   ├── models.py                        # Learned embeddings + attention
│   ├── train.py                         # Training script for original approach
│   │
│   │── # Stats at team level (pre-aggregated) - R² ≈ 0.20
│   ├── models_stats.py                  # StatsBaseline, StatsAttention
│   ├── train_stats.py                   # Training script
│   │
│   │── # Stats at player level (no learned embeddings) - R² ≈ 0.20
│   ├── models_stats_player.py           # StatsPlayerBaseline, StatsPlayerAttention
│   ├── train_stats_player.py            # Training script
│   │
│   │── # Final approach (stats + embeddings) - R² ≈ 0.20
│   ├── models_hybrid.py                 # HybridBaseline, HybridAttention
│   ├── train_hybrid.py                  # Game-level training
│   ├── train_hybrid_rich.py             # Game-level training (7 features)
│   │
│   │── # XGBoost challenger - BEST R² ≈ 0.29
│   ├── train_xgboost.py                 # XGBoost with multiple feature modes
│   │
│   │── # Lineup-level experiment - R² ≈ 0.02
│   ├── models_lineup.py                 # LineupBaseline, LineupAttention
│   ├── train_lineup.py                  # Lineup-level training
│   │
│   │── # Data pipelines
│   ├── compute_player_stats.py          # Compute 5 features per player
│   ├── compute_player_stats_rich.py     # Compute 7 features per player
│   ├── process_games_hybrid.py          # Game-level data processing
│   ├── process_games_hybrid_rich.py     # Game-level processing (7 features)
│   └── process_lineups.py               # Lineup-level data processing
│
└── checkpoints/                         # Saved models (not in repo)
```

### Data Setup

The raw data is too large for GitHub. To reproduce:

1. Download [NBA Play-by-Play Data](https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019) from Kaggle
2. Place `all_games.csv` in the `data/` directory
3. Run the processing scripts (see Usage below)

---

## Usage

### Requirements

```bash
pip install torch pandas numpy tqdm xgboost scikit-learn
```

### Training

```bash
# Baseline model (recommended)
python src/train_hybrid_rich.py --model baseline

# Attention model
python src/train_hybrid_rich.py --model attention

# With embedding regularization
python src/train_hybrid_rich.py --model attention --emb-reg 0.5
```

### Reproducing from Scratch

```bash
# 1. Compute player statistics
python src/compute_player_stats_rich.py

# 2. Process games into train/val/test splits
python src/process_games_hybrid_rich.py

# 3. Train game-level models
python src/train_hybrid_rich.py --model baseline
python src/train_hybrid_rich.py --model attention
```

### XGBoost Baseline

```bash
# Averaged features (14 features)
python src/train_xgboost.py --feature-mode averaged

# Concatenated features (160 features) - BEST PERFORMANCE
python src/train_xgboost.py --feature-mode concat

# Summary statistics (56 features)
python src/train_xgboost.py --feature-mode summary
```

### Lineup-Level Experiment

```bash
# 1. Process lineups (aggregates by 5-player combinations)
python src/process_lineups.py

# 2. Train lineup-level models
python src/train_lineup.py --model baseline
python src/train_lineup.py --model attention
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Top-10 players per team | Captures full rotation, not just starters |
| Possession-weighted averaging | Minutes played → influence on outcome |
| Small embedding initialization (`std=0.01`) | Embeddings start as residuals to stats |
| Normalized targets | Stabilizes gradient flow |
| Early stopping (patience=15) | Prevents overfitting |
| AdamW + weight decay (0.01) | Standard regularization |

---

## Future Directions

1. **Opponent-adjusted lineup ratings**: Control for opponent strength when computing lineup net ratings, potentially revealing chemistry effects hidden by strength-of-schedule variance

2. **Embedding visualization**: Cluster learned embeddings to find player archetypes and analyze which players have large residual embeddings (undervalued by stats)

3. **External features**: Add injuries, rest days, travel distance, home court advantage to improve game-level predictions

4. **Real-time prediction**: Use the model to predict live game outcomes as lineups change throughout a game

---

## Acknowledgments

- [NBA Play-by-Play Dataset](https://www.kaggle.com/datasets/schmadam97/nba-playbyplay-data-20182019)
- [NBA2Vec Paper](https://arxiv.org/abs/2302.13386) for inspiration on player embeddings
- Built with assistance from Claude Code
