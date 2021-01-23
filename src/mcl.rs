use std::iter::Sum;
use anyhow::Result;

use ndarray::{Array2, ArrayBase, Axis, Data, Dimension};

use approx::{AbsDiffEq};

use num_traits::{Float, zero, one};

pub trait PartiqlArgMaxExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: PartialOrd
{
    fn argmax(&self) -> Result<D::Pattern>;
}

impl<A, S, D> PartiqlArgMaxExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: PartialOrd
{
    fn argmax(&self) -> Result<D::Pattern> {
        let mut current_max = self.first().unwrap();
        let mut current_pattern_max = D::zeros(self.ndim()).into_pattern();
        
        for (pattern, elem) in self.indexed_iter() {
            if elem.partial_cmp(current_max).unwrap() == std::cmp::Ordering::Greater {
                current_pattern_max = pattern;
                current_max = elem;
            }
        }
        Ok(current_pattern_max)
    }
}

pub trait MclExt<A>
where 
    A: Float,
{   
    /// Normalize the columns of the given matrix by L1 normalization
    ///
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// # #[macro_use] extern crate approx;
    /// use markov_clustering_rs::mcl::*;
    /// use ndarray::Array2;
    /// let input: Array2<f64> = array![[1., 1., 0.],
    ///                                 [0., 1., 1.],
    ///                                 [0., 0., 1.]];
    /// let output: Array2<f64> = array![[1., 0.5, 0.],
    ///                                 [0., 0.5, 0.5],
    ///                                 [0., 0., 0.5]];
    /// assert_abs_diff_eq!(input.normalize().unwrap(), output)
    /// ```
    fn normalize(&self) -> Result<Array2<A>>;

    /// Apply cluster expansion to the given matrix with given power
    ///
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// # #[macro_use] extern crate approx;
    /// use markov_clustering_rs::mcl::*;
    /// use ndarray::Array2;
    /// let input = array![[1., 0.5, 0.],
    ///                 [0., 0.5, 0.5],
    ///                 [0., 0., 0.5]];
    /// let output = array![[1., 0.75, 0.25],
    ///                 [0., 0.25, 0.5 ],
    ///                 [0., 0., 0.25]];
    /// assert_abs_diff_eq!(input.expand(2).unwrap(), output) 
    /// ```
    fn expand(&self, power: i32) -> Result<Array2<A>>;

    ///  Apply cluster inflation to the given matrix with given power
    /// 
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// # #[macro_use] extern crate approx;
    /// use markov_clustering_rs::mcl::*;
    /// use ndarray::Array2;
    /// let input: Array2<f64> = array![[0.5, 0.5],
    ///                                 [1.,   1.]];
    /// let output: Array2<f64> = array![[0.2, 0.2],
    ///                                  [0.8, 0.8]];
    /// assert_abs_diff_eq!(input.inflate(2.).unwrap(), output)
    /// ```
    fn inflate(&self, power: A) -> Result<Array2<A>>;

    /// prune the matrix below threshold
    /// The maximum value in each col is not pruned
    ///
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// # #[macro_use] extern crate approx;
    /// use markov_clustering_rs::mcl::*;
    /// use ndarray::Array2;
    /// let threshold = 2.5;
    /// let input: Array2<f64> = array![[1., 2., 3.], [3., 1., 4.]];
    /// let output: Array2<f64> = array![[0., 2., 3.,], [3., 0., 4.]];
    /// assert_abs_diff_eq!(input.prune(threshold).unwrap(), output)
    /// ``` 
    fn prune(&self, threshold: A) -> Result<Array2<A>>;

    /// Add self loop to the matrix
    fn add_self_loop(&mut self, loop_value: A) -> Result<()>;

    /// mcl clustering from ndarray::Array2
    ///
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// # #[macro_use] extern crate approx;
    /// use markov_clustering_rs::mcl::*;
    /// use ndarray::Array2;
    ///
    /// let expansion = 2;
    /// let inflation = 2.;
    /// let loop_value = 1.;
    /// let iterations = 100;
    /// let pruning_threshold = 0.0001;
    /// let pruning_frequency = 1;
    /// let convergence_check_frequency = 1;
    /// let input: Array2<f64> = array![[1., 1., 1., 0., 0., 0., 0.],
    ///                                 [1., 1., 1., 0., 0., 0., 0.],
    ///                                 [1., 1., 1., 1., 0., 0., 0.],
    ///                                 [0., 0., 1., 1., 1., 0., 1.],
    ///                                 [0., 0., 0., 1., 1., 1., 1.],
    ///                                 [0., 0., 0., 0., 1., 1., 1.],
    ///                                 [0., 0., 0., 1., 1., 1., 1.]];
    /// let output: Array2<f64> = array![[0., 0., 0., 0., 0., 0., 0.],
    ///                                 [0., 0., 0., 0., 0., 0., 0.],
    ///                                 [1., 1., 1., 0., 0., 0., 0.],
    ///                                 [0., 0., 0., 0., 0., 0., 0.],
    ///                                 [0., 0., 0., 0.5, 0.5, 0.5, 0.5],
    ///                                 [0., 0., 0., 0., 0., 0., 0.],
    ///                                 [0., 0., 0., 0.5, 0.5, 0.5, 0.5]];
    /// assert_abs_diff_eq!(input.mcl(expansion, inflation, loop_value, iterations, pruning_threshold, pruning_frequency, convergence_check_frequency).unwrap(), output)
    /// ```
    ///
    fn mcl(&self,
        expansion: i32,
        inflation: A,
        loop_value: A,
        iterations: usize,
        pruning_threshold: A,
        pruning_frequency: usize,
        convergence_check_frequency: usize,
    ) -> Result<Array2<A>>;
}

fn _handle_zeros_in_scale<A: Float>(scale: A) -> A {
    if scale == zero() { one() } else { scale }
} 

impl<A> MclExt<A> for Array2<A>
where
    A: 'static + Float + Sum + AbsDiffEq,
{
    fn normalize(&self) -> Result<Array2<A>> {
        let mut vec: Vec<A> = Vec::new();

        for row in self.t().axis_iter(Axis(0)) {
            let norm_l1: A = _handle_zeros_in_scale(row.iter().map(|x| x.abs()).sum());

            for &x in row.iter() {
                vec.push( x / norm_l1)
            } 
        }

        let shape = (self.shape()[1], self.shape()[0]);
        let mat: Array2<A> = Array2::from_shape_vec(shape, vec)?.reversed_axes();
        Ok(mat)
    }

    fn expand(&self, power: i32) -> Result<Array2<A>> {
        let mut mat: Array2<A> = self.to_owned();

        for _ in 0..power-1 {
            mat = mat.dot(self)
        }

        Ok(mat)
    }

    fn inflate(&self, power: A) -> Result<Array2<A>> {
        self.mapv(|x| x.powf(power)).normalize()
    }

    fn add_self_loop(&mut self, loop_value: A) -> Result<()> {
        let shape = self.shape();
        assert_eq!(shape[0], shape[1]);

        for i in 0..shape[0] {
            self[(i, i)] = loop_value;
        }

        Ok(())
    }

    fn prune(&self, threshold: A) -> Result<Array2<A>> {
        let mat: Array2<A> = self.to_owned();
        let mut pruned = mat.mapv(|x| if x < threshold { zero() } else {x});

        for (i, row) in mat.axis_iter(Axis(1)).enumerate() {
            let c = row.argmax()?;
            pruned[(c, i)] = self[(c, i)];
        }
        
        Ok(pruned)
    }

    fn mcl(&self, expansion: i32, inflation: A, loop_value: A, iterations: usize, pruning_threshold: A, pruning_frequency: usize, convergence_check_frequency: usize) -> Result<Array2<A>> {
        let mut mat: Array2<A> = self.to_owned();

        if loop_value > zero() {
            mat.add_self_loop(loop_value)?
        }

        mat = mat.normalize()?;

        for i in 0..iterations {
            let last_mat = mat.clone();

            mat = mat.expand(expansion)?.inflate(inflation)?;
            
            if i % pruning_frequency == pruning_frequency - 1 {
                mat = mat.prune(pruning_threshold)?;
            }

            if i % convergence_check_frequency == convergence_check_frequency - 1 {
                
                #[allow(deprecated)]
                if mat.all_close(&last_mat, A::from(1e-8).unwrap()) {
                    break;
                }
            }
        }
        
        Ok(mat)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::{AbsDiffEq, assert_abs_diff_eq};

    #[test]
    fn test_normalize_1() {
        let matrix: Array2<f64> = array![[-0.02630925,  0.34560928], [ 0.49153899,  0.37912572]];
        let normed_matrix: Array2<f64> = array![[-0.05080494,  0.47687676], [ 0.94919506,  0.52312324]];
        // dbg!(&matrix.normalize().unwrap());
        assert!(matrix.normalize().unwrap().abs_diff_eq(&normed_matrix, 1e-8))
    }

    #[test]
    fn test_normalize_2() {
        let matrix: Array2<f64> = array![[-1.1990861 ,  0.21545948,  1.62300919, -0.04798489],
                                            [-0.197236  ,  0.27418912, -2.31481452, -0.67681584],
                                            [-1.00124304, -0.63797063,  0.01981998, -0.43391746]];
        let normed_matrix: Array2<f64> = array![[-0.5001266 ,  0.19107467,  0.41009482, -0.04141204],
                        [-0.08226513,  0.24315755, -0.58489715, -0.58410738],
                        [-0.41760827, -0.56576778,  0.00500802, -0.37448058]];
        // dbg!(&matrix.normalize());
        assert!(matrix.normalize().unwrap().abs_diff_eq(&normed_matrix, 1e-8))
    }

    #[test]
    fn test_normalize_3() {
        let input: Array2<f64> = array![[1., 1., 0.],
                                        [0., 1., 1.],
                                        [0., 0., 1.]];
        let output: Array2<f64> = array![[1., 0.5, 0.],
                                        [0., 0.5, 0.5],
                                        [0., 0., 0.5]];
        assert_abs_diff_eq!(input.normalize().unwrap(), output)
    }

    #[test]
    fn test_normalize_4() {
        let input = array![[0., 0.], [0., 0.]];
        let output = array![[0., 0.], [0., 0.]];

        assert_abs_diff_eq!(input.normalize().unwrap(), output)
    }

    #[test]
    fn test_pruned() {
        let threshold = 2.5;
        let matrix = array![[1., 2., 3.], [3., 1., 4.]];
        let pruned_mat = array![[0., 2., 3.,], [3., 0., 4.]];
        // dbg!(&matrix.prune(threshold).unwrap());
        assert!(matrix.prune(threshold).unwrap().abs_diff_eq(&pruned_mat, 1e-8))
    }

    #[test]
    fn test_expand() {

        let input = array![[1., 0.5, 0.],
                                        [0., 0.5, 0.5],
                                        [0., 0., 0.5]];
        let output = array![[1., 0.75, 0.25],
                                                                [0., 0.25, 0.5 ],
                                                                [0., 0., 0.25]];
        assert_abs_diff_eq!(input.expand(2).unwrap(), output)    
    }

    #[test]
    fn test_inflate() {
        let input: Array2<f64> = array![[0.5, 0.5],
                                        [1.,   1.]];
        let output: Array2<f64> = array![[0.2, 0.2],
                                        [0.8, 0.8]];
        assert_abs_diff_eq!(input.inflate(2.).unwrap(), output)
    }

    #[test]
    fn test_mcl() {
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
        assert_abs_diff_eq!(input.mcl(
            2, 2., 1., 100, 0.001, 1, 1,
        ).unwrap(), output)
    }
}