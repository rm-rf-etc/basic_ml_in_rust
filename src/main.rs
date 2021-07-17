extern crate image;
extern crate rand;
mod matrix2d;
mod ml;
mod shared;

use image::io::Reader;
use ml::*;
use ndarray::Array2;
use std::fs;

fn file_to_vec(path: &str) -> Vec<f32> {
    Reader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8()
        .to_vec()
        .iter()
        .map(|i| *i as f32)
        .collect::<Vec<f32>>()
}

fn files_vec(root: &str) -> Vec<String> {
    fs::read_dir(root)
        .unwrap()
        .into_iter()
        .map(|f| f.unwrap().path().display().to_string())
        .collect::<Vec<String>>()
}

fn main() {
    let num_iterations = 6;
    let vec = files_vec("images/test/")
        .iter()
        .map(|file| file_to_vec(file))
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
