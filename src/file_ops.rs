use csv::{Error, Reader as CsvReader, StringRecord};
use image::io::Reader as FileReader;
use std::fs;
extern crate rayon;

type Vec2d<T> = Vec<Vec<T>>;

#[allow(dead_code)]
pub fn image_file_to_vecf32(path: &str) -> Vec<f32> {
    FileReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8()
        .to_vec()
        .iter()
        .map(|i| *i as f32)
        .collect::<Vec<f32>>()
}

#[allow(dead_code)]
pub fn dir_to_vec(root: &str) -> Vec<String> {
    fs::read_dir(root)
        .unwrap()
        .into_iter()
        .map(|f| f.unwrap().path().display().to_string())
        .collect::<Vec<String>>()
}

pub fn load_from_csv(path: &str, divisor: f32) -> Result<Vec2d<f32>, String> {
    let mut v = csv_file_to_2dvecf32(path).unwrap();

    use rayon::prelude::*;
    v.par_iter_mut().for_each(|row| {
        row.par_iter_mut().for_each(|n| *n /= divisor);
    });

    Ok(v)
}

fn f32_from_string_record(line: Result<StringRecord, Error>) -> Vec<f32> {
    line.unwrap()
        .iter()
        .map(|s| s.parse::<f32>().unwrap())
        .collect::<Vec<f32>>()
}

fn csv_file_to_2dvecf32(path: &str) -> Result<Vec<Vec<f32>>, Error> {
    let vec = CsvReader::from_path(path)?
        .records()
        .map(f32_from_string_record)
        .collect::<Vec<Vec<f32>>>();

    Ok(vec)
}
