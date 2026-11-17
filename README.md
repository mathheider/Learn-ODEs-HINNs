# Learning ODEs with Hamiltonian-inspired neural networks

Code from my NSF REU at Emory University (Computational Mathematics for Data Science, 2022 to 2023), with Dr. Deepanshu Verma and Dr. Lars Ruthotto.

A residual network learns a system's motion directly. A Hamiltonian-inspired neural network (HINN) learns the system's Hamiltonian, its energy function, and the motion follows from it. These notebooks build both kinds of network in PyTorch and test them on a spring-mass system and on the two-body problem.

**Main result.** Stepping the learned system with symplectic Verlet integration matched the accuracy of fourth-order Runge-Kutta (RK4) at about 4.5 times lower compute cost.

Presented as *Predicting Dynamics of ODEs using Residual and Hamiltonian-Inspired Neural Networks* at the Joint Mathematics Meetings, 2023.

## Notebooks

| Notebook | What it holds |
|---|---|
| `All_Neural_Nets.ipynb` | The four models side by side on the spring-mass system: a residual network, a residual network with RK4 steps, and HINNs stepped with RK4 and with Verlet |
| `HINN_Finding_H.ipynb` | A HINN that learns the Hamiltonian of the spring-mass system |
| `HINN_Finding_H_finding_TimeSteps.ipynb` | The same model with the number of time steps varied |
| `2_Body_Problem.ipynb` | Two-body orbits: data generation, training and energy plots |
| `2_Body_Problem_RK4.ipynb` | The two-body problem with RK4 steps |
| `combined_rk4_and_verlet.ipynb` | RK4 and Verlet compared on the two-body problem |
| `3_Body_Problem.ipynb` | An exploratory three-body setup |
| `Copy_of_2_Body_Problem.ipynb` | A working copy of the two-body notebook |

`sho_data.npz` is the spring-mass data the first three notebooks load.

## Running them

The notebooks were written for Google Colab. They need PyTorch, NumPy, SciPy, Matplotlib and [hessQuik](https://github.com/elizabethnewman/hessQuik) (`pip install hessquik`), which supplies the network layers with fast gradients and Hessians.

## Credit

The two-body data generation is adapted from the Hamiltonian Neural Networks code of Greydanus, Dzamba and Yosinski ([greydanus/hamiltonian-nn](https://github.com/greydanus/hamiltonian-nn)).
