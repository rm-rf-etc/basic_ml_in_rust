use ndarray::{Array, Array2};
use rand::prelude::*;

pub fn new2d_rand(r: usize, c: usize, scale: f32) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn((r, c), |_| rng.gen::<f32>() * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new2d_rand() {
        let m = new2d_rand(2, 2, 0.01);
        assert_eq!(m.len(), 4);
        assert_eq!(m.shape(), [2, 2]);
        assert!(m[[0, 0]] <= 0.01 && m[[0, 0]] >= 0.0);
        assert!(m[[0, 1]] <= 0.01 && m[[0, 1]] >= 0.0);
        assert!(m[[1, 0]] <= 0.01 && m[[1, 0]] >= 0.0);
        assert!(m[[1, 1]] <= 0.01 && m[[1, 1]] >= 0.0);
    }
}
