# Multi-Agent LLM Trading Framework

**"Virtual Trading Firm"** - A modular, multi-agent AI trading system where each LLM agent performs a specific trading role, working together like a real trading desk to produce explainable trading decisions.

## Overview

Most AI trading scripts treat an LLM as a single giant oracle: _"Here's price data. Should I buy or sell?"_

This approach is **unrealistic and unstable**.

Real trading firms divide responsibilities across specialized teams:
- Research
- Market analysis
- Execution
- Risk management
- Compliance
- Logging & oversight

This system replicates that structure using **multiple LLM agents**, each with a role, tools, and communication protocol, forming a **virtual trading firm**.

## Architecture

### Agent Pipeline

```
Data → Market Analyst → News Analyst → Strategy → CIO → Risk Manager → Execution → Environment → Logger
```

### Agents

| Agent | Responsibilities | Outputs |
|-------|-----------------|---------|
| **Market Analyst** | Short-term trend, volatility, TA indicators | Trend, momentum, confidence |
| **News & Events Analyst** | News summaries, sentiment, catalysts | Event impact assessment |
| **Research/Strategy** | Pattern recognition, historical context | Proposed trade idea |
| **CIO / Decision** | Final directional call | BUY/SELL/HOLD |
| **Risk Manager** | Check size, limits, exposure, stops | Approved size, veto power |
| **Execution Trader** | Convert decision → order | Action dict |
| **Trade Journal / Logger** | Log rationales, PnL, reasons | JSON/CSV logs |

## Features

- **Multi-Agent Architecture**: Specialized agents with clear roles
- **Multiple LLM Providers**: OpenAI, Anthropic Claude, Ollama (local models)
- **Structured Communication**: JSON-based agent communication
- **Risk Management**: Hard-coded failsafes and LLM-based risk assessment
- **Paper Trading**: Pandas-based simulation environment
- **Complete Logging**: All decisions, rationales, and outcomes logged
- **Transparent Reasoning**: Every decision is explained
- **Modular & Extensible**: Easy to add new agents or features

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/trading-agents.git
cd trading-agents
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install as package:

```bash
pip install -e .
```

### 4. Set up API keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Quick Start

### Basic Example

```python
from trading_agents.main import TradingFirm
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
price = 100.0
prices = []
for i in range(len(dates)):
    price += np.random.normal(0.5, 2)
    prices.append(price)

data = pd.DataFrame({
    'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
    'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
    'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
    'close': prices,
    'volume': [np.random.randint(1000000, 5000000) for _ in prices]
}, index=dates)

# Initialize trading firm
firm = TradingFirm(
    llm_provider="openai",  # or "anthropic" or "ollama"
    initial_capital=100000.0,
    max_position_size=5.0,
    verbose=True
)

# Run backtest
firm.run_backtest(data=data)

# Get results
results = firm.get_results()
print(results['metrics'])
```

### Run the demo

```bash
python trading_agents/main.py
```

## Configuration

Edit `trading_agents/config/config.yaml` to customize:

```yaml
# LLM Settings
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.5

# Trading Parameters
trading:
  initial_capital: 100000.0
  max_position_size: 10.0
  max_drawdown_pct: 0.15

# Risk Management
risk:
  conservative_mode: false
  min_confidence_threshold: 0.6
```

Or load configuration programmatically:

```python
from trading_agents.config.config_loader import load_config

config = load_config('path/to/config.yaml')
```

## Project Structure

```
trading_agents/
├── agents/              # Agent implementations
│   ├── base_agent.py    # Base agent class
│   ├── market_analyst.py
│   ├── news_analyst.py
│   ├── strategy.py
│   ├── cio.py
│   ├── risk_manager.py
│   ├── execution.py
│   └── logger.py
├── core/                # Core components
│   ├── llm.py          # LLM interface layer
│   └── market_env.py   # Trading environment simulator
├── data/                # Data processing
│   └── indicators.py   # Technical indicators
├── config/              # Configuration
│   ├── config.yaml
│   └── config_loader.py
└── main.py             # Main orchestrator

examples/                # Example scripts
logs/                    # Trade logs (generated)
tests/                   # Unit tests
```

## Advanced Usage

### Using Different LLM Providers

#### OpenAI (GPT-4)

```python
firm = TradingFirm(
    llm_provider="openai",
    llm_model="gpt-4o-mini"
)
```

#### Anthropic Claude

```python
firm = TradingFirm(
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022"
)
```

#### Local Models (Ollama)

```bash
# First, start Ollama with your model
ollama pull llama3.1
ollama serve
```

```python
firm = TradingFirm(
    llm_provider="ollama",
    llm_model="llama3.1"
)
```

### Custom Risk Parameters

```python
from trading_agents.agents import RiskManagerAgent
from trading_agents.core.llm import get_llm

llm = get_llm("openai")

risk_manager = RiskManagerAgent(
    llm=llm,
    max_position_size=5.0,
    max_exposure_pct=0.3,  # Max 30% exposure
    max_drawdown_pct=0.10,  # Max 10% drawdown
    stop_loss_pct=0.03      # 3% stop loss
)
```

### Adding News Data

```python
news_data = [
    {
        'headline': 'Fed announces rate cut',
        'source': 'Reuters',
        'sentiment': 'positive'
    },
    # ... more news
]

firm.run_backtest(
    data=data,
    news_data=news_data
)
```

### Logging and Analysis

All decisions are logged to `logs/` directory:

- `{timestamp}_decisions.json`: Complete decision log with all agent outputs
- `{timestamp}_trades.csv`: Executed trades with P&L
- `{timestamp}_equity.csv`: Equity curve

```python
# Access logs programmatically
summary = firm.logger.get_summary()
print(summary)

# Save logs
firm.logger.save_logs(prefix="my_backtest")

# Print summary
firm.logger.print_summary()
```

## Performance Metrics

The system tracks:

- **Total Return %**
- **Max Drawdown %**
- **Sharpe Ratio**
- **Sortino Ratio**
- **Win Rate**
- **Profit Factor**
- **Total Trades**

```python
metrics = firm.env.get_performance_metrics()
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
```

## Roadmap

### Phase 1 - MVP ✅
- [x] Basic agent system
- [x] Price-only simulator
- [x] Logging
- [x] Multi-provider LLM support

### Phase 2 - Full Framework (Next)
- [ ] Real news integration (RSS, APIs)
- [ ] Sentiment analysis
- [ ] Advanced position sizing (Kelly criterion, ATR-based)
- [ ] Multi-asset support
- [ ] Portfolio management

### Phase 3 - Advanced (Future)
- [ ] Long-term memory (vector DB)
- [ ] Agent debate mode
- [ ] Quant hybrid (ML models as agents)
- [ ] Live trading integration (Alpaca, IBKR)
- [ ] Web dashboard
- [ ] Backtesting optimization

## Extensions

### Adding New Agents

```python
from trading_agents.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            name="Custom Agent",
            role="My Custom Role",
            llm=llm
        )

    def get_system_prompt(self):
        return "You are a custom agent..."

    def process(self, context):
        return self.generate_response(context, use_json=True)
```

### Adding New Indicators

Edit `trading_agents/data/indicators.py`:

```python
def calculate_custom_indicator(data: pd.Series) -> pd.Series:
    # Your indicator logic
    return result
```

## Safety & Disclaimers

⚠️ **Important**: This is a research/educational framework.

- **NOT** production-ready for real money trading
- **NOT** financial advice
- **NO** guarantee of profitability
- LLMs can hallucinate and make errors
- Always test thoroughly in paper trading first
- Use proper risk management
- Consider regulatory requirements

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## License

MIT License - see LICENSE file

## Citation

If you use this framework in research:

```bibtex
@software{trading_agents_2024,
  author = {Michał},
  title = {Multi-Agent LLM Trading Framework},
  year = {2024},
  url = {https://github.com/yourusername/trading-agents}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/trading-agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/trading-agents/discussions)

## Acknowledgments

Built with:
- OpenAI GPT models
- Anthropic Claude
- Pandas for data processing
- Inspired by real trading firm structures

---

**Built by traders, for traders 📈🤖**
