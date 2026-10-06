Charged H3O+ SCCS/PCC regression.

The full cavity adds a 1 Bohr oxygen core Gaussian; hydrogen is excluded by Z=zv. SCCS starts after the vacuum density residual drops below 1e-5 (or at iteration 50).

Uses epsilon=1.01 for a short 20 Ry CI regression. Higher-dielectric force and cell convergence are validated separately. No explicit ecutrho. The integration harness checks total energy, force magnitudes, solvation energies and PCC energy where enabled. References use 2 MPI ranks and OMP_NUM_THREADS=1.
