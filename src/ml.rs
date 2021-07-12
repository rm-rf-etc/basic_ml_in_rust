use super::matrix2d;
use ndarray::Array2;
use std::collections::HashMap;
use std::f32::consts::E;

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
pub fn init_deep_nn_params(layers: Vec<usize>) -> Result<HashMap<String, Array2<f32>>, String> {
    let mut map = HashMap::new();

    for i in 1..layers.len() {
        let this_l = layers[i - 1];
        let next_l = layers[i];
        let w = matrix2d::new_rand(next_l, this_l);
        let b: Array2<f32> = Array2::zeros((next_l, 1));
        map.insert(format!("W{}", i), w);
        map.insert(format!("b{}", i), b);
    }

    Ok(map)
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
}
