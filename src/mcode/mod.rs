use std::{collections::{HashSet}, iter::Sum};
use petgraph::{EdgeType, graph::{Graph, IndexType, NodeIndex}};
use num_traits::{Float};

trait McodeExt<W, Ty, Ix> {
    fn vertex_weighting(&mut self);
}

impl<W, Ty, Ix> McodeExt<W, Ty, Ix> for Graph<W, W, Ty, Ix>
where
    W: Float + Sum,
    Ty: EdgeType,
    Ix: IndexType
{
    fn vertex_weighting(&mut self) {
        _get_node_weight_from_edges(self);

        for node in self.node_indices() {
            let mut neighborhood: HashSet<NodeIndex<Ix>> = std::iter::once(node).chain(self.neighbors(node)).collect();

            if neighborhood.len() <= 2 { continue; }

            let mut k = 2;

            loop {
                neighborhood = _update_neighborhood(&neighborhood, k, self);

                if neighborhood.is_empty() { break; }

                let new_weight = _make_new_weight(W::from(k).unwrap(), node, &neighborhood, self);

                match self.node_weight_mut(node) {
                    Some(w) => {
                        if *w > new_weight {
                            *w = new_weight;
                        }
                    },
                    None => {}
                }

                k += 1;
            }
        }
    }
}


/// vertex weight = k-core number * density of k-core
fn _make_new_weight<W, Ty, Ix>(k: W, node: NodeIndex<Ix>, neighborhood: &HashSet<NodeIndex<Ix>>, graph: &Graph<W, W, Ty, Ix>) -> W
where
    W: Float + Sum,
    Ty: EdgeType,
    Ix: IndexType
{
    k * neighborhood.iter()
        .filter_map(|&x| graph.find_edge(node, x))
        .filter_map(|e| graph.edge_weight(e))
        .map(|x| *x)
        .sum::<W>() / W::from(neighborhood.len().pow(2u32)).unwrap()
}

fn _update_neighborhood<W, Ty, Ix>(neighborhood: &HashSet<NodeIndex<Ix>>, k: usize, graph: &Graph<W, W, Ty, Ix>) -> HashSet<NodeIndex<Ix>>
where
    W: Float + Sum,
    Ty: EdgeType,
    Ix: IndexType
{
    let mut flag = true;
    let mut neighborhood = neighborhood.clone();
    let mut invalid_nodes = HashSet::new();
    while flag && !neighborhood.is_empty() {
        for &n in neighborhood.clone().iter() {
            let n_neighborhood: HashSet<NodeIndex<Ix>> = graph.neighbors(n).collect();
            // dbg!(&n_neighborhood);

            if n_neighborhood.intersection(&neighborhood).count() < k {
                invalid_nodes.insert(n);
            }
        }

        // dbg!(&invalid_nodes);
        neighborhood = neighborhood.difference(&invalid_nodes).map(|x| *x).collect();
        flag = !invalid_nodes.is_empty();
    }

    neighborhood
}

fn _get_node_weight_from_edges<W, Ty, Ix>(graph: &mut Graph<W, W, Ty, Ix>)
where
    W: Float + Sum,
    Ty: EdgeType,
    Ix: IndexType
{
    for node in graph.node_indices() {
        let weight: W = graph.edges(node).map(|e| *e.weight()).sum::<W>() / W::from(graph.neighbors(node).count().pow(2u32)).unwrap();
        match graph.node_weight_mut(node) {
            Some(w) => { *w = weight},
            None => {}
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use petgraph::{Undirected};

    fn graph1() -> Graph::<f64, f64, Undirected, usize> {
        Graph::<f64, f64, Undirected, usize>::from_edges(&[
            (0, 1, 1.), (0, 2, 0.7), (0, 3, 0.7),
            (1, 2, 1.), (1, 3, 0.7),
            (2, 3, 0.8),
        ])
    }

    #[test]
    fn test_get_node_weight_from_edges_1() {
        let mut gr1 = graph1();
        _get_node_weight_from_edges(&mut gr1);
        assert_abs_diff_eq!(*gr1.node_weight(NodeIndex::new(0)).unwrap(), 0.26666666666666666);
        assert_abs_diff_eq!(*gr1.node_weight(NodeIndex::new(1)).unwrap(), 0.3);
        assert_abs_diff_eq!(*gr1.node_weight(NodeIndex::new(2)).unwrap(), 0.2777777777777777);
        assert_abs_diff_eq!(*gr1.node_weight(NodeIndex::new(3)).unwrap(), 0.24444444444444444);
    }

    #[test]
    fn test_update_neighbors() {
        let gr1 = graph1();
        let node: NodeIndex<usize> = NodeIndex::new(0);

        let mut neighborhood: HashSet<NodeIndex<usize>> = std::iter::once(node).chain(gr1.neighbors(node)).collect();
        dbg!(1, &neighborhood);
        neighborhood = _update_neighborhood(&neighborhood, 2, &gr1);
        dbg!(2, &neighborhood);
    }
}