# Replicating Toy Models of Superposition

## Project Overview
This project is a replication of the core geometric phenomenon described in Anthropic's "Toy Models of Superposition" paper. It demonstrates how neural networks use polysemanticity to store more features than they have neurons by utilizing high-dimensional geometry.

## The Math
We train a sparse autoencoder to compress $N=5$ sparse features into $M=2$ dimensions. The model minimizes the reconstruction loss: $\mathcal{L} = ||x - x'||^2$

## Results
As predicted by the theory, the model does not collapse. Instead, it learns to arrange the feature vectors into a regular pentagon within the 2D embedding space, maximizing the interference distance between features.

<img width="800" height="800" alt="superposition_plot" src="https://github.com/user-attachments/assets/c06f490f-04ef-4c0d-8151-4fc4263db23f" />
