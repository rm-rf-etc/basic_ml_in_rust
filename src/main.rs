extern crate rand;
mod matrix2d;
mod ml;

use ndarray::{Array, Array2};

fn main() {
    println!("sigmoid of -1: {:?}", ml::sigmoid(-1.0));
    println!("relu of -1: {:?}", ml::relu(-1.0));
    let m1 = matrix2d::new_rand(3, 2) * 0.01;
    println!("matrix of 3x2:\n{:?}", m1);
    let m2: Array2<f32> = Array::zeros((3, 1));
    println!("element-wise multiplication:\n{:?}", m1 * m2);
}
