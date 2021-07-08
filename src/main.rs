extern crate rand;
mod matrix2d;
mod ml;

fn main() {
    println!("sigmoid of -1: {:?}", ml::sigmoid(-1.0));
    println!("relu of -1: {:?}", ml::relu(-1.0));
    println!("matrix of 3x2: {:?}", matrix2d::new_fill_rand(3, 2, 0.01));
}
