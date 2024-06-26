# Code for the Bayesian Behaviors framework

The current repository contains the source code for generating the simulation results for the paper "**Synergizing habits and goals with variational Bayes**" by Dongqi Han, Kenji Doya, Dongsheng Li and Jun Tani, published on *Nature Communications*.  [[Link]](https://www.nature.com/articles/s41467-024-48577-7)

## Installation

Tested using Python 3.7.7 on Ubuntu 20.04 and Windows 11

### Install Requirements (typically takes a few minutes)

```bash
pip install -r requirements.txt 
```

And you also need to install PyTorch. Please install PyTorch >= 1.11 that matches your CUDA version according to <https://pytorch.org/>.

## Demo: play with trained agent for customized goal-directed planning

You can try to play with trained agent model for goal-directed planning (the agent used for Figure.5 in the [paper](https://www.nature.com/articles/s41467-024-48577-7)) in PyBullet GUI.

- You can customize the goal and see how the agent performs!
- Check the ipython notebook `goal-directed-planning-demo.ipynb` and get started!

![image info](./GUI.png)

## How to train and inference (Python, PyTorch)

### Habitization Experiment (Results for Figures 2, 3, 4)

```bash
python run_habitization_experiment.py --seed 42 --verbose 1 --gui 0
```

Set `--gui 1` if you want to see the visualized environment.

The default arguments (hyperparameters) are the same as used in the paper. For the information of the arguments in training the habitual behavior, see `run_habitization_experiment.py`

To run the models with different training steps in stage 2 (Figure 3), use the `--stage_3_start_step` argument.

### Flexible Goal-Directed Planning Experiment (Results for Figure 5)

```bash
python run_planning_experiment.py --seed 42 --verbose 1 --gui 0
```

### Data format

Either program takes less than 1 day with a descent GPU, the result data will be saved at `./data/` and `./details/` (and at `./planning/` for the planning experiment) in .mat files, for which you can load using MATLAB or scipy:

```python
import scipy.io as sio
data = sio.loadmat("xxx.mat")
```

The PyTorch model of the trained agent will also be saved at `./data/`, which can be loaded by `torch.load()`.



## Tutorial on plotting the quantitative results in the article (MATLAB)

To replicate the plots, please ensure you have MATLAB version R2022b or later, and download the simulated result data from TODO.
(You may also train your own models using the guideline above).

The start, change the MATLAB working directory to ./data_analysis

### Figure 2b

```matlab
plot_adaptation_readaptation_progress("DATAPATH/BB_habit_automaticity/search_mpz_0.1_s3s_420000/details/")
```

Please modify DATAPATH to the data folder you downloaded.

### Figure 2c-h

```matlab
fig2_habitization_analysis("DATAPATH/BB_habit_automaticity/search_mpz_0.1_s3s_420000/data/")
```

### Figure 3

```matlab
fig3_extinction_analysis("DATAPATH/BB_habit_automaticity/")
```

### Figure 4

```matlab
fig4_devaluation_analysis("DATAPATH/BB_habitization/")
```

### Figure 5b

```matlab
plot_adaptation_progress("DATAPATH/BB_planning/search_mpz_0.1/details/")
```

### Figure 5c

```matlab
plot_diversity_statistics("DATAPATH/BB_planning/search_mpz_0.1/details/")
```

### Figure 5d,e

```matlab
plot_planning_details("DATAPATH/BB_planning/search_mpz_0.1/planning/")
```

## Citation

Han, D., Doya, K., Li, D. et al. Synergizing habits and goals with variational Bayes. Nat Commun 15, 4461 (2024). https://doi.org/10.1038/s41467-024-48577-7

### BibTeX

```
@article{han2024synergizing,
  title={Synergizing habits and goals with variational Bayes},
  author={Han, Dongqi and Doya, Kenji and Li, Dongsheng and Tani, Jun},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={4461},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
