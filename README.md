[![Workflow Status](https://github.com/illumination-k/markov-clustering-rs/workflows/Rust/badge.svg)](https://github.com/illumination-k/markov-clustering-rs/actions?query=workflow%3A%22Rust%22)

# markov-clustering-rs

This crates implements of the MCL algorithm in rust

The MCL argorithm was developed by Stijn van Dongen at the University of Utrecht. Details of the algorithm can be found on the MCL homepage.

## Example Usage

```rust
use markov_clustering_rs::mcl::*;
use ndarray::Array2;

// set parameters
let expantion = 2;
let inflation = 2.;
let loop_value = 1.;
let iterations = 100;
let pruning_threshold = 0.0001;
let pruning_frequency = 1;
let convergence_check_frequency = 1;

let input: Array2<f64> = array![[1., 1., 1., 0., 0., 0., 0.],
                                [1., 1., 1., 0., 0., 0., 0.],
                                [1., 1., 1., 1., 0., 0., 0.],
                                [0., 0., 1., 1., 1., 0., 1.],
                                [0., 0., 0., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 1., 1., 1.],
                                [0., 0., 0., 1., 1., 1., 1.]];
let output: Array2<f64> = array![[0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0.],
                                [1., 1., 1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0.5, 0.5, 0.5, 0.5],
                                [0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0.5, 0.5, 0.5, 0.5]];
assert_abs_diff_eq!(input.mcl(expantion, inflation, loop_value, iterations, pruning_threshold, pruning_frequency, convergence_check_frequency).unwrap(), output)
```

Please see the [API documentation](https://illumination-k.github.io/markov-clustering-rs/markov_clustering_rs/) for more details.

License: MIT
