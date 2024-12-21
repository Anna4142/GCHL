# Goal-Conditioned Hierarchical Learning (GCHL)

This repository is forked from [OGBench](https://github.com/seohongpark/ogbench) and contains implementations of additional agents for goal-conditioned hierarchical reinforcement learning.

## Added Agents
We extend OGBench with the following new agent implementations:

- **GCHBC**: Goal-Conditioned Hierarchical Behavioral Cloning
 - Enhances standard BC with subgoal extraction and weighting
 - Integrates seamlessly with standard goal-conditioned frameworks

- **HCHIQL**: Hierarchical Conditioned Implicit Q-Learning 
 - Combines hierarchical learning with IQL
 - Modified architecture for improved performance

- **HIQL**: Original Hierarchical Implicit Q-Learning implementation
 - High/low level policy decomposition
 - Goal-conditioned value estimation

## Base Repository
This is built on [OGBench](https://github.com/seohongpark/ogbench), a benchmark for offline goal-conditioned RL. Please refer to the original repository for:
- Environment details
- Dataset information
- Base agent implementations 
- Evaluation protocols

## Running Experiments

Train individual agents:
```bash
# Train HBC agent on cube environment
python main.py --env_name=cube-single-play-v0 --agent=agents/hbc.py

# Train GCHBC on navigation
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gchbc.py 

# Train HCHIQL on scene manipulation
python main.py --env_name=scene-play-v0 --agent=agents/hchiql.py

# Train HIQL
python main.py --env_name=cube-single-play-v0 --agent=agents/hiql.py