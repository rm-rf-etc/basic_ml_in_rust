use super::matrix2d;
use ndarray::Array2;
use std::collections::HashMap;
use std::f32::consts::E;

type Matrix = Array2<f32>;
type MatrixTriple = (Matrix, Matrix, Matrix);

/*
The goal of this file is to demonstrate how the basic functionality of a multi-layer
neural network is implemented.
*/

// sigmoid activation function
pub fn sigmoid(f: f32) -> f32 {
    1.0 / (1.0 + E.powf(-f))
}

// relu activation function
pub fn relu(f: f32) -> f32 {
    f32::max(0.0, f)
}

// This function generates an N layer NN from a set of integers specifying the
// respective layer sizes in order.
pub fn init_deep_nn_params(layers: Vec<usize>) -> Result<HashMap<String, Matrix>, String> {
    let mut map = HashMap::new();

    for i in 1..layers.len() {
        let this_l = layers[i - 1];
        let next_l = layers[i];
        let w = matrix2d::new_rand(next_l, this_l);
        let b = Array2::<f32>::zeros((next_l, 1));
        map.insert(format!("W{}", i), w);
        map.insert(format!("b{}", i), b);
    }

    Ok(map)
}

// Linear forward is the preceding step to calculating activation.
pub fn linear_forward(a: Matrix, w: Matrix, b: Matrix) -> (Matrix, MatrixTriple) {
    let z = w.dot(&a) + &b;
    // (a, w, b) is the cache tuple
    (z, (a, w, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!((sigmoid(1.0) * 10000.0).round() / 10000.0, 0.7311);
        assert_eq!((sigmoid(-1.0) * 10000.0).round() / 10000.0, 0.2689);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(-1.0), 0.0);
    }

    #[test]
    fn test_init_deep_nn_params() {
        let nn1 = init_deep_nn_params(vec![3, 2, 2, 1]).unwrap();
        assert_eq!(nn1["W1"].shape(), [2, 3]);
        assert_eq!(nn1["b1"].shape(), [2, 1]);
        assert_eq!(nn1["W2"].shape(), [2, 2]);
        assert_eq!(nn1["b2"].shape(), [2, 1]);
        assert_eq!(nn1["W3"].shape(), [1, 2]);
        assert_eq!(nn1["b3"].shape(), [1, 1]);
    }

    #[test]
    fn test_linear_forward() {
        let prec = 100000.0;
        let round = |f: f32| (f * prec).round() / prec;

        let a = ndarray::arr2(&[
            [1.62434536, -0.61175641],
            [-0.52817175, -1.07296862],
            [0.86540763, -2.3015387],
        ]);
        let b = ndarray::arr2(&[[1.74481176, -0.7612069, 0.3190391]]);
        let c = ndarray::arr2(&[[-0.24937038]]);

        let (z, _) = linear_forward(a, b, c);
        let expected = ndarray::arr2(&[[3.26295337, -1.23429987]]);

        assert_eq!(expected.shape(), z.shape());
        assert_eq!(round(expected[[0, 0]]), round(z[[0, 0]]));
        assert_eq!(round(expected[[0, 1]]), round(z[[0, 1]]));
    }
}
