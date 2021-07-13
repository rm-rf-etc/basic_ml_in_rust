extern crate rand;
mod matrix2d;
mod ml;
mod shared;

use ndarray::Array2;

fn main() {
    println!("\nsigmoid of -1: {:?}", ml::sigmoid(-1.0));

    println!("\nrelu of -1: {:?}", ml::relu(-1.0));

    let m1 = matrix2d::new_rand(3, 2);
    println!("\nmatrix of 3x2:\n{:?}", m1);

    let m2 = m1 * 0.01;
    println!("\nscalar multiplication:\n{:?}", m2);

    let m3 = Array2::<f32>::zeros((3, 1));
    println!("\nelement-wise multiplication:\n{:?}", m2 * m3);

    let nn1 = ml::init_deep_nn_params(vec![3, 2, 2, 1]).unwrap();
    println!("\n3L NN:");
    println!("\nW1:\n{:?}", nn1["W1"]);
    println!("\nb1:\n{:?}", nn1["b1"]);
    println!("\nW2:\n{:?}", nn1["W2"]);
    println!("\nb2:\n{:?}", nn1["b2"]);
    println!("\nW2:\n{:?}", nn1["W3"]);
    println!("\nb2:\n{:?}", nn1["b3"]);

    let m1 = ndarray::arr2(&[
        [1.62434536, -0.61175641],
        [-0.52817175, -1.07296862],
        [0.86540763, -2.3015387],
    ]);
    let m2 = ndarray::arr2(&[[1.74481176, -0.7612069, 0.3190391]]);
    let b = ndarray::arr2(&[[-0.24937038]]);
    let (z, _) = ml::linear_forward(m1, m2, b);
    let expected_out = ndarray::arr2(&[[3.26295337, -1.23429987]]);

    println!("{:?}", z[[0, 0]]);
    println!("{:?}", expected_out[[0, 0]]);
    println!("{:?}", z[[0, 1]]);
    println!("{:?}", expected_out[[0, 1]]);
}
