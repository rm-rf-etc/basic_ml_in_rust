use super::ml::Matrix;
use ndarray::{Array, Array2, Axis};
use rand::prelude::*;

pub fn new_rand(r: usize, c: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn((r, c), |_| rng.gen::<f32>())
}

pub fn sum_keepdims(axis: usize, m: &Matrix) -> Result<Matrix, String> {
    match m.shape() {
        [x, y] => {
            let shape = match axis {
                0 => Ok((*y, 1)),
                1 => Ok((1, *x)),
                _ => Err("matrix2d::sum_keepdims only supports 2D matrices"),
            }
            .unwrap();

            Ok(m.sum_axis(Axis(axis)).to_shape(shape).unwrap().to_owned())
        }
        _ => Err("Only accepts 2D matrices".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_rand() {
        let m = new_rand(2, 2) * 0.01;
        assert_eq!(m.len(), 4);
        assert_eq!(m.shape(), [2, 2]);
        assert!(m[[0, 0]] <= 0.01 && m[[0, 0]] >= 0.0);
        assert!(m[[0, 1]] <= 0.01 && m[[0, 1]] >= 0.0);
        assert!(m[[1, 0]] <= 0.01 && m[[1, 0]] >= 0.0);
        assert!(m[[1, 1]] <= 0.01 && m[[1, 1]] >= 0.0);
    }

    #[test]
    fn test_sum_keepdims() {
        let m1 = ndarray::arr2(&[[0.2, 0.3], [0.1, 0.4]]);
        let m2 = sum_keepdims(0, &m1).unwrap();
        let m3 = sum_keepdims(1, &m1).unwrap();
        assert_eq!(m2.shape(), [2, 1]);
        assert_eq!(m3.shape(), [1, 2]);
    }
}
