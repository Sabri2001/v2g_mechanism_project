# Scheduling the Charge of Temporally Flexible Electric Vehicles: A Game-Theoretic Approach

## Abstract

With the rise of intermittent renewable energy sources and the massive electrification of our activities, the grid is coming under increasing pressure. Through vehicle-to-grid (V2G) services, electric vehicles (EVs) offer a promising approach for mitigating infrastructure constraints at EV charging stations (EVCS) while also serving as backup capacity for the grid. In this work, we explore how the flexibility in the departure time of EV drivers can be exploited to amplify these benefits.

We start by formulating a mixed-integer quadratic charge scheduling problem that accounts for the EVs' temporal flexibility, and we efficiently approximate its solution using the Alternating Direction Method of Multipliers (ADMM). We then show that a charging station coordinator needs to know the charging preferences of its users, including their flexibility, to maximize the satisfaction of its customers. Considering the possibility of strategic behavior from EV drivers, we subsequently design a Vickrey–Clarke–Groves mechanism to elicit truthful preference reporting.

We conclude with numerical simulations using real data to study the value of deferring EVs' desired departure times. We find conditions under which these delays help the charging station reduce costs while still meeting all charging requests in spite of its limited capacity.

## Project Structure

- **analyses/**: Notebooks for preliminary analysis of the datasets used in the simulations
- **config/**: Configuration files for running the experiments
- **run_experiments/**: Bash files for running the experiments
- **experiments_runners/**: Helper for running experiments
- **experiments/**: Charge schedulers
- **preprocessing/**: Data preprocessing and formatting script
- **results/**: Result and plot handlers
- **plot_experiments/**: Scripts for individual experiment plots
- **log_experiments/**: Scripts for individual experiment logs

## Key Features

- Mixed-integer quadratic charge scheduling optimization
- ADMM-based solution approximation for computational efficiency
- VCG mechanism implementation for truthful user preference elicitation
- Temporal flexibility analysis of EV charging scenarios
- Simulation framework using real-world EV charging data

## Experiments legend
- **Experiment 1/**: Effect of battery wear costs on charge schedule
- **Experiment 2/**: Cost savings from bidirectional charging as a function of battery wear costs
- **Experiment 3/**: Effect of station congestion on charge schedule
- **Experiment 4/**: Cost savings from temporal flexibility as a function of station congestion
- **Experiment 5/**: VCG tax as a function of the charging request
- **Experiment 6/**: Indidividual cost savings from optimal charging with VCG taxes
- **Experiment 7/**: Costs saved from reporting temporal flexibilities to the scheduler
- **Experiment 8/**: Optimality gap with ADMM
- **Experiment 9/**: Utility of EVs as a function of their bids in the absence of taxes
- **Experiment 10/**: Occurence of a negative VCG tax


## Getting Started

### Prerequisites
```
# Install dependencies from environment.yml
conda env create -f environment.yml
conda activate v2g
```

### Running Experiments
```
# Basic experiment execution
python main.py --config configs/config.yaml
```

## Results
Our findings demonstrate that leveraging the temporal flexibility of EV users can significantly reduce charging station operational costs while maintaining service quality. The game-theoretic mechanism ensures truthful reporting of user preferences, creating a fair and efficient charging ecosystem.

## License
This project is available under the [MIT License](LICENSE).

## Contact

For questions or collaborations, please open an issue or contact the repository owner.
