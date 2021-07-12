use super::matrix2d;
use ndarray::Array2;
use std::collections::HashMap;
use std::f32::consts::E;

// sigmoid activation function
pub fn sigmoid(f: f32) -> f32 {
    1.0 / (1.0 + E.powf(-f))
}

// relu activation function
pub fn relu(f: f32) -> f32 {
    f32::max(0.0, f)
}

// This function generates a 2 layer NN. We will later improve this function,
// making it more generic, so we can init an arbitrary number of layers.
pub fn init_2l_nn(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) -> Result<HashMap<String, Array2<f32>>, String> {
    let w1 = matrix2d::new_rand(hidden_size, input_size);
    let b1: Array2<f32> = Array2::zeros((hidden_size, 1));
    let w2 = matrix2d::new_rand(output_size, hidden_size);
    let b2: Array2<f32> = Array2::zeros((output_size, 1));

    let mut map = HashMap::new();

    map.insert("W1".to_string(), w1);
    map.insert("b1".to_string(), b1);
    map.insert("W2".to_string(), w2);
    map.insert("b2".to_string(), b2);

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
    fn test_init_2l_nn() {
        let nn1 = init_2l_nn(3, 2, 1).unwrap();
        assert_eq!(nn1["W1"].shape(), [2, 3]);
        assert_eq!(nn1["b1"].shape(), [2, 1]);
        assert_eq!(nn1["W2"].shape(), [1, 2]);
        assert_eq!(nn1["b2"].shape(), [1, 1]);
    }
}
