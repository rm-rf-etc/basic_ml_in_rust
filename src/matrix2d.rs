use super::ml::Matrix2D;
use ndarray::{Array, Array2, Axis, Zip};
use rand::prelude::*;

pub fn new_rand(r: usize, c: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array::from_shape_fn((r, c), |_| rng.gen::<f32>())
}

pub fn sum_keepdims(axis: usize, m: &Matrix2D) -> Result<Matrix2D, String> {
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

pub fn divide(a: &Matrix2D, b: &Matrix2D) -> Matrix2D {
    Zip::from(a).and(b).map_collect(|n1, n2| {
        let v = *n1 / *n2;
        if v.is_nan() {
            0.0
        } else {
            v
        }
    })
}

pub fn from_2d_vec(vec: &Vec<Vec<f32>>, row_l: usize) -> Matrix2D {
    let shape = (vec.len(), row_l);
    Array::from_shape_fn(shape, |(c, r)| vec[c][r])
}

pub fn log(m: &Matrix2D) -> Matrix2D {
    m.map(|f| f.log(std::f32::consts::E))
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

    #[test]
    fn test_from_2d_vec() {
        let mat = from_2d_vec(&vec![vec![1.0, 2.0], vec![0.0, -1.0]], 2);
        let exp = ndarray::arr2(&[[1.0, 2.0], [0.0, -1.0]]);
        assert_eq!(mat, exp);
    }
}
