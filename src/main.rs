extern crate image;
extern crate rand;
mod file_ops;
mod matrix2d;
mod ml;
mod shared;

use ml::*;
use ndarray::Array2;

fn main() {
    let num_iterations = 6;
    let vec = file_ops::dir_to_vec("images/test/")
        .iter()
        .map(|file| file_ops::image_file_to_vecf32(file))
        .collect::<Vec<Vec<f32>>>();
    let num_images = vec.len();
    let input_layer = matrix2d::from_2d_vec(&vec, 64 * 64 * 3)
        .map(|f| f / 255.0)
        .t()
        .to_owned();
    let labels = Array2::ones((1, num_images));

    println!("Labels {:?}", labels.shape());
    println!("Inputs {:?}\n", input_layer.shape());

    let layers = vec![
        (12288, ActivationFn::Relu),
        (7, ActivationFn::Relu),
        (1, ActivationFn::Sigmoid),
    ];
    let mut model = ml::ModelTrainer::new(input_layer, layers, labels);

    for _ in 0..num_iterations {
        // for (w, b, _) in &model.parameters {
        //     println!("W {:?}", w.shape());
        //     println!("b {:?}\n", b.shape());
        //     println!("b: {:?}", b);
        // }
        model.train(0.0075);
        println!("Cost after: {}", model.cost);
    }
    println!("Cost after: {}", model.cost);
}
