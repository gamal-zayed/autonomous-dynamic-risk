# Execution & Reproducibility Guide

To reproduce the benchmark performance metrics presented in the manuscript:

1. Launch CARLA Simulator with graphics acceleration:
   ```bash
   ./CarlaUE4.sh -quality-level=Epic -benchmark -fps=20

2. Run the main experiment execution command:
  ```bash
python src/DRA_v4.3.2_Resume_State_Fix.py --seed 1000 --scenario S1
```
3. Verification:
Compare generated output CSV logs against canonical references stored in results/canonical/
