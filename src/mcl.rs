use anyhow::Result;
use ndarray::{Array2, Axis};
use ndarray_linalg::*;
use ndarray_stats::*;

/// normalize matrix by L1 normalization method
fn normalize(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let mut new_matrix = matrix.clone();

    for (i, row) in matrix.axis_iter(Axis(1)).enumerate() {
        let norms = row.norm_l1();
        // TODO! row / norms

    }
    Ok(new_matrix)
}

fn inflate(matrix: &Array2<f64>, power: i32) -> Result<Array2<f64>> {
    let norm_mat = normalize(matrix)?;
    let inflate_mat = norm_mat.mapv(|f| f.powi(power));
    Ok(inflate_mat)
}

fn expand(matrix: &Array2<f64>, power: i32) -> Result<Array2<f64>> {
    let mut new_mat = matrix.clone();

    for _ in 0..power-1 {
        new_mat = new_mat.dot(matrix)
    }

    Ok(new_mat)
}
 
fn add_self_loop(matrix: &Array2<f64>, loop_value: f64) -> Result<Array2<f64>> {
    let shape = matrix.shape();
    assert_eq!(shape[0], shape[1]);
    let mut new_matrix = matrix.clone();
    for i in 0..shape[0] {
        new_matrix[(i, i)] = loop_value;
    }
    Ok(new_matrix)
}

fn prune(matrix: &Array2<f64>, threshold: f64) -> Result<Array2<f64>> {
    let mut pruned = matrix.clone().mapv(|x| { if x < threshold { 0. } else { x }});

    for (i, row) in matrix.axis_iter(Axis(1)).enumerate() {
        let c: usize = row.argmax()?;

        pruned[(c, i)] = matrix[(c, i)];
    }

    Ok(pruned)
}

fn iterate(matrix: &Array2<f64>, expansion: i32, inflation: i32) -> Result<Array2<f64>>{
    inflate(
        &expand(matrix, expansion)?,
        inflation
    )
}

pub fn mcl(
    matrix: Array2<f64>,
    expansion: i32,
    inflation: i32,
    loop_value: f64,
    iterations: usize,
    pruning_threshold: f64,
    pruning_frequency: usize,
    convergence_check_frequency: usize
) -> Result<Array2<f64>> {
    let mut new_mat = matrix.clone();

    if loop_value > 0. {
        new_mat = add_self_loop(&new_mat, loop_value)?;
    }

    new_mat = normalize(&new_mat)?;

    for i in 0..iterations {
        let last_mat = new_mat.clone();

        new_mat = iterate(&new_mat, expansion, inflation)?;

        if i % pruning_frequency == pruning_frequency - 1 {
            new_mat = prune(&new_mat, pruning_threshold)?;
        }

        if i % convergence_check_frequency % i == convergence_check_frequency - 1 {
            // TODO!
            // check abs_diff_eq(last_mat, new_mat)
            // break;
        }
    }

    Ok(new_mat)
} 

#[cfg(test)]
mod test {
    use super::*;
    use approx::AbsDiffEq;

    #[test]
    fn test_normalize_1() {
        let matrix = array![[-0.02630925,  0.34560928], [ 0.49153899,  0.37912572]];
        let normed_matrix = array![[-0.05080494,  0.47687676], [ 0.94919506,  0.52312324]];
        dbg!(normalize(&matrix).unwrap());
        assert!(normalize(&matrix).unwrap().abs_diff_eq(&normed_matrix, 1e-8))
    }

    #[test]
    fn test_pruned() {
        let threshold = 2.5;
        let matrix = array![[1., 2., 3.], [3., 1., 4.]];
        let pruned_mat = array![[0., 2., 3.,], [3., 0., 4.]];

        assert!(prune(&matrix, threshold).unwrap().abs_diff_eq(&pruned_mat, 1e-8))
    }
}