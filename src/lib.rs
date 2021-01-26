/*!
This crates implements of the graph clustering algorithm in rust

Please see the [API documentation](https://illumination-k.github.io/markov-clustering-rs/markov_clustering_rs/) for more details.
*/


extern crate anyhow;

#[cfg_attr(test, macro_use)]
extern crate ndarray;

extern crate approx;

extern crate num_traits;

pub mod mcl;
pub mod utils;
