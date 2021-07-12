extern crate rand;
mod matrix2d;
mod ml;

use ndarray::{Array, Array2};

fn main() {
    println!("\nsigmoid of -1: {:?}", ml::sigmoid(-1.0));

    println!("\nrelu of -1: {:?}", ml::relu(-1.0));

    let m1 = matrix2d::new_rand(3, 2);
    println!("\nmatrix of 3x2:\n{:?}", m1);

    let m2 = m1 * 0.01;
    println!("\nscalar multiplication:\n{:?}", m2);

    let m3: Array2<f32> = Array::zeros((3, 1));
    println!("\nelement-wise multiplication:\n{:?}", m2 * m3);

    let nn1 = ml::init_2l_nn(3, 2, 1).unwrap();
    println!("\n2L NN:");
    println!("\nW1:\n{:?}", nn1["W1"]);
    println!("\nb1:\n{:?}", nn1["b1"]);
    println!("\nW2:\n{:?}", nn1["W2"]);
    println!("\nb2:\n{:?}", nn1["b2"]);
}
