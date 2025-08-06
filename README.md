# Scheduling the Charge of Temporally Flexible Electric Vehicles: A Game-Theoretic Approach

## Abstract

With the rise of intermittent renewable energy sources and the massive electrification of our activities, the grid is coming under increasing pressure. Through vehicle-to-grid (V2G) services, electric vehicles (EVs) offer a promising approach for mitigating infrastructure constraints at EV charging stations (EVCS) while also serving as backup capacity for the grid. In this work, we explore how the flexibility in the departure time of EV drivers can be exploited to amplify these benefits.

We start by formulating a mixed-integer quadratic charge scheduling problem that accounts for the EVs' temporal flexibility, and we efficiently approximate its solution using the Alternating Direction Method of Multipliers (ADMM). We then show that a charging station coordinator needs to know the charging preferences of its users, including their flexibility, to maximize the satisfaction of its customers. Considering the possibility of strategic behavior from EV drivers, we subsequently design a Vickrey–Clarke–Groves mechanism to elicit truthful preference reporting.

We conclude with numerical simulations using real data to study the value of deferring EVs' desired departure times. We find conditions under which these delays help the charging station reduce costs while still meeting all charging requests in spite of its limited capacity.

## Project Structure

- **analyses/**: Energy efficiency and stochastic initial/final state of charge analyses
- **config/**: Configuration files for running experiments
- **experiment_mains/**: Main experiment execution files
- **experiments/**: Experiment specification files
- **preprocessing/**: Data preprocessing and formatting scripts
- **results/**: Output data and analysis results

## Key Features

- Mixed-integer quadratic charge scheduling optimization
- ADMM-based solution approximation for computational efficiency
- VCG mechanism implementation for truthful user preference elicitation
- Temporal flexibility analysis of EV charging scenarios
- Simulation framework using real-world EV charging data

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
