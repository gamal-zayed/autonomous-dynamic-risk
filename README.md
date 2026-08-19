# autonomous-dynamic-risk
Modeling the Dynamic Risk Assessment for Autonomous Systems

# Modeling Dynamic Risk Assessment for Autonomous-Systems Conflict Resolution: Constraint-Aware and Distribution-Sensitive Decision-Making Framework

Official implementation and canonical research artifacts for **"Modeling Dynamic Risk Assessment for Autonomous-Systems Conflict Resolution: Constraint-Aware and Distribution-Sensitive Decision-Making Framework"**.

## Project Overview
The Dynamic Risk Assessment (DRA) framework evaluates candidate maneuvers in a road-aligned local coordinate system using planar rotation, incorporating hard admissibility constraints alongside distribution-sensitive and ethical risk terms to resolve critical pedestrian-oncoming conflicts.

## Funding & Acknowledgments
This research is supported and funded under the Afretec Research Grant (African Engineering and Technology Network, involving Carnegie Mellon University Africa, The American University in Cairo, and partner institutions)[cite: 8, 9].

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/gamal-zayed/autonomous-dynamic-risk/blob/main/LICENSE) file for details.

## Setup
Bash
```text
git clone https://github.com/gamal-zayed/autonomous-dynamic-risk.git
cd autonomous-dynamic-risk
pip install -r requirements.txt
```

## Running the Canonical Benchmark
Start the CARLA server instance on Town03.

Run the main evaluation script:

Bash
```text
python src/DRA_v4.3.2_Resume_State_Fix.py --seed 1000 --scenario S1
```
Logs will be generated under results/.

## Repository Structure
```text
DRA-autonomous-vehicle/
├── README.md
├── LICENSE (MIT License)
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/
│   └── DRA_v4.3.2_Resume_State_Fix.py
├── experiments/
│   └── run_instructions.md
├── results/
│   └── canonical/
│       ├── episode_results_20260813_143431.csv
│       ├── timestep_log_20260813_143431.csv
│       └── DRA_Debug_v4.3.2.2.txt
└── docs/
    └── Supporting_Information.pdf
