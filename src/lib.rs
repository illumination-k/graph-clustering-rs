/*!
This crates implements of the graph clustering algorithm in rust

Please see the [API documentation](https://illumination-k.github.io/graph-clustering-rs/graph_clustering_rs/) for more details.

## RoadMap

- [x] Markov Clustering
- [ ] louvain
- [ ] HCCA
- [ ] MCODE
- [ ] DPClus
- [ ] IPCA
- [ ] CoAch
- [ ] Graph Entropy Clustering
*/


extern crate anyhow;

#[cfg_attr(test, macro_use)]
extern crate ndarray;

extern crate approx;

extern crate num_traits;

extern crate petgraph;

pub mod mcl;
pub mod utils;
pub mod mcode;