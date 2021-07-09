use ndarray::{Array, Array2};
use rand::prelude::*;

pub fn new_rand(r: usize, c: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn((r, c), |_| rng.gen::<f32>())
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
}
