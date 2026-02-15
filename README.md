# CryptoEvolution

[![Run tests](https://github.com/rochfedorowicz/CryptoEvolution/actions/workflows/test.yml/badge.svg)](https://github.com/rochfedorowicz/CryptoEvolution/actions/workflows/test.yml)

**A comprehensive research platform for algorithmic trading using machine learning**

Main documentation & reports: https://rochfedorowicz.github.io/CryptoEvolution/

---

## Overview

CryptoEvolution is a research environment designed for conducting reproducible experiments in algorithmic trading. The platform provides a complete pipeline for data processing, market simulation, and model training/evaluation using both supervised learning and reinforcement learning paradigms.

The system supports:

- Collecting and preprocessing financial market data with technical indicators
- Simulating trading environments with realistic order management
- Training and evaluating ML models using different learning strategies
- Generating comprehensive performance reports with financial metrics

---

## Architecture

![Architecture Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/rochfedorowicz/CryptoEvolution/gh-pages/diagrams/architecture.puml)

The platform integrates three main services:

- **GitHub** - Source code repository, CI/CD automation, and interactive documentation hosting
- **AWS S3** - Data storage for datasets, configuration files, and generated reports
- **Paperspace** - GPU compute resources for computationally intensive experiments

---

## Core Components

### Data Processing Pipeline

The data handling system processes market data through several stages:

1. **Data Collection** - Fetch historical price data from Coinbase or Yahoo Finance APIs
2. **Feature Engineering** - Extend raw OHLCV data with technical indicators:
   - Volatility (always included)
   - Optional indicators: ATR, Bollinger Bands, Donchian Channels, EMA, MACD, MFI, Moving Volume Profile, OBV, RSI, Stochastic Oscillator
3. **Labeling** (for supervised learning) - Dynamic threshold-based trend classification using volatility-adjusted thresholds
4. **Normalization** - Local normalization within sliding time windows to maintain relative relationships

### Trading Environment Simulation

The `TradingEnvironment` simulates market conditions with:

- **Order Management** - Support for long and short positions with configurable stop-loss and take-profit thresholds
- **Trading Modes**:
  - _Partial Autonomy_ - Agent opens positions; automatic closure via SL/TP thresholds
  - _Full Autonomy_ - Agent controls both opening and closing of positions
- **Budget Control** - Constraints on maximum concurrent positions and capital allocation
- **Reward Function** - Normalized returns from closed positions with penalties for excessive inactivity

### Learning Strategies

The platform supports two most popular paradigms:

**Supervised Learning (Classification)**

- Models predict price trend direction (up/down/neutral)
- Uses labeled historical data with class balancing options
- Evaluation via standard classification metrics (accuracy, precision, recall, F1, ROC curves)

**Reinforcement Learning**

- Models learn optimal trading policies through environment interaction
- Compatible with keras-rl2 agents (DQN)
- Evaluation via financial performance metrics (Sharpe ratio, annual returns)

Both approaches can utilize the same neural network architectures (RNN, LSTM, CNN, etc.) built with TensorFlow/Keras.

![Class Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/rochfedorowicz/CryptoEvolution/gh-pages/diagrams/classes.puml)

---

## Key Technologies

- **Python** - Core implementation language
- **TensorFlow/Keras** - Deep learning model construction
- **scikit-learn** - Classical ML algorithms and metrics
- **keras-rl2** - Reinforcement learning agents
- **pandas/numpy** - Data manipulation and numerical computing
- **Gym** - RL environment interface

---

## Use Cases

### 1. Dataset Preparation

Create preprocessed datasets with technical indicators:

```bash
python scripts/create_dataset.py \
  --symbol BTC-USD \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --granularity 1h \
  --indicators RSI MACD EMA
```

**Parameters:**

- Trading symbol or input CSV file
- Date range for data collection
- Time granularity (1m, 5m, 1h, 1d, etc.)
- List of technical indicators to include

Datasets are stored on AWS S3 for use in experiments.

![Use case 1 Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/rochfedorowicz/CryptoEvolution/gh-pages/diagrams/use_case_1.puml)

### 2. Running Experiments

Execute training and evaluation using configuration files.

First, create a configuration file (e.g., `experiment_config.json`):

```json
{
  "training_config": {
    "model_blue_print": {
      "class_name": "RnnBluePrint"
    },
    "initial_budget": 1000,
    "max_amount_of_trades": 5,
    "window_size": 24,
    "learning_strategy_handler": {
      "class_name": "ReinforcementLearningStrategyHandler"
    },
    "testing_strategy_handlers": [
      {
        "class_name": "PerformanceTestingStrategyHandler"
      }
    ]
  },
  "data_set_name": "s3://bucket/path/data.csv"
}
```

Then run the experiment:

```bash
python scripts/train_model.py --config_path experiment_config.json
```

**Key Configuration Options:**

- Dataset location (S3 path)
- Model architecture blueprint
- Learning strategy (supervised/reinforcement)
- Testing strategies (classification metrics/financial performance)
- Trading parameters (budget, window size, position limits)

Experiments are executed on Paperspace with GPU acceleration, then results are uploaded to AWS S3.

![Use case 2 Diagram](http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/rochfedorowicz/CryptoEvolution/gh-pages/diagrams/use_case_2.puml)

---

## Generated Reports

Each experiment produces a comprehensive PDF report containing:

### Asset Summary Visualizations

- **Price Movement** - Logarithmic scale chart showing asset price evolution
- **Volatility Analysis** - Time series of market volatility
- **Combined View** - Price movements with volatility overlay

### Classification Reports (Supervised Learning)

- **Trend Labels** - Annotated price chart with trend classifications
- **Class Distribution** - Bar chart showing label balance
- **Confusion Matrix** - Multi-class prediction accuracy breakdown
- **Performance Metrics** - Precision, recall, F1-score per class and averaged
- **ROC Curves** - True positive vs false positive rates

### Financial Performance Reports

- **Portfolio Value** - Agent's capital vs buy-and-hold baseline
- **Annualized Sharpe Ratio** - Risk-adjusted return visualization
- **Annual Returns** - Yearly return projections over testing period

All metrics are computed on held-out test sets and compared against passive buy-and-hold strategies.

---

## Development

The project maintains **80%+ code coverage** through continuous integration with GitHub Actions.

### Project Structure

```
source/
├── agent/          # Agent adapters for different learning paradigms
├── data_handling/  # API collectors and data preprocessing
├── environment/    # Trading simulation and reward functions
├── indicators/     # Technical indicator implementations
├── model/          # Neural network blueprints and building blocks
├── plotting/       # Report generation
├── training/       # Experiment orchestration
└── utils/          # AWS/cloud integration utilities
```

---

## License

See LICENSE file for details.

---

## Citations

If you use this platform in your research, please cite the original work and reference the repository.
