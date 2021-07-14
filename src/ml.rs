use super::matrix2d;
use ndarray::Array2;
use std::f32::consts::E;

#[allow(dead_code)]
pub type Matrix = Array2<f32>;

#[allow(dead_code)]
pub type MatrixTriple = (Matrix, Matrix, Matrix);

#[allow(dead_code)]
pub enum ActivationFn {
    Relu,
    Sigmoid,
}

/*
The goal of this file is to demonstrate how the basic functionality of a multi-layer
neural network is implemented.
*/

// sigmoid activation function
#[allow(dead_code)]
pub fn sigmoid(f: f32) -> f32 {
    1.0 / (1.0 + E.powf(-f))
}

// sigmoid activation function
#[allow(dead_code)]
pub fn sigmoid_m(z: Matrix) -> (Matrix, Matrix) {
    let a = 1.0 / (1.0 + z.map(|f| E.powf(-f)));
    let activation_cache = z;

    (a, activation_cache)
}

// relu activation function
#[allow(dead_code)]
pub fn relu(f: f32) -> f32 {
    f32::max(0.0, f)
}

// relu activation function on matrices
#[allow(dead_code)]
pub fn relu_m(z: Matrix) -> (Matrix, Matrix) {
    let a = z.map(|f| relu(*f));
    let activation_cache = z;

    (a, activation_cache)
}

// This function generates an N layer NN from a set of integers specifying the
// respective layer sizes in order.
#[allow(dead_code)]
pub fn init_deep_nn_params(layers: Vec<usize>) -> Result<Vec<(Matrix, Matrix)>, String> {
    let mut vec = vec![];

    for i in 1..layers.len() {
        let this_l = layers[i - 1];
        let next_l = layers[i];
        let w = matrix2d::new_rand(next_l, this_l);
        let b = Array2::<f32>::zeros((next_l, 1));
        vec.push((w, b));
    }

    Ok(vec)
}

// Linear forward is the preceding step to calculating activation, (a, w, b) is the cache tuple.
pub fn linear_forward(a: Matrix, w: Matrix, b: Matrix) -> (Matrix, MatrixTriple) {
    let z = w.dot(&a) + &b;
    let cache = (a, w, b);

    (z, cache)
}

// Linear->Activation forward, combines linear_forward with activation.
#[allow(dead_code)]
pub fn linear_activation_forward(
    a_prev: Matrix,
    w: Matrix,
    b: Matrix,
    act_fn: ActivationFn,
) -> (Matrix, (MatrixTriple, Matrix)) {
    let (z, linear_cache) = linear_forward(a_prev, w, b);
    let (a, activation_cache) = match act_fn {
        ActivationFn::Relu => relu_m(z),
        ActivationFn::Sigmoid => sigmoid_m(z),
    };
    (a, (linear_cache, activation_cache))
}

// backward propagation
#[allow(dead_code)]
pub fn linear_backward(dz: Matrix, cache: MatrixTriple) -> Result<MatrixTriple, String> {
    let (a_prev, w, _) = cache;
    let m = a_prev.shape()[1] as f32;
    let dw = 1.0 / m * dz.dot(&a_prev.t());
    let db = 1.0 / m * matrix2d::sum_keepdims(1, &dz).unwrap();
    let da_prev = w.t().dot(&dz);

    Ok((da_prev, dw, db))
}

#[allow(dead_code)]
pub fn sigmoid_backward_m(da: Matrix, z_cache: Matrix) -> Result<Matrix, String> {
    if da.shape() == z_cache.shape() {
        let e = (-1.0 * z_cache).map(|x| x.exp());
        let s = 1.0 / (1.0 + e);
        let dz = da * &s * (1.0 - &s);

        Ok(dz)
    } else {
        Err("Z matrix shape does not match cache shape".to_string())
    }
}

// relu backward propagation function
#[allow(dead_code)]
pub fn relu_backward_m(z: Matrix, z_cache: Matrix) -> Result<Matrix, String> {
    if z.shape() == z_cache.shape() {
        let mask = z_cache.map(|f: &f32| if *f > 0.0 { 1.0 } else { 0.0 });
        Ok(z * mask)
    } else {
        Err("Z matrix shape does not match cache shape".to_string())
    }
}

#[allow(dead_code)]
pub fn linear_activation_backward(
    da: Matrix,
    cache: (MatrixTriple, Matrix),
    act_fn: ActivationFn,
) -> Result<MatrixTriple, String> {
    let (linear_cache, activation_cache) = cache;

    let dz = match act_fn {
        ActivationFn::Relu => relu_backward_m(da, activation_cache).unwrap(),
        ActivationFn::Sigmoid => sigmoid_backward_m(da, activation_cache).unwrap(),
    };
    let (da_prev, dw, db) = linear_backward(dz, linear_cache).unwrap();

    Ok((da_prev, dw, db))
}

#[allow(dead_code)]
pub fn l_model_forward(
    x: Matrix,
    params: &mut Vec<(Matrix, Matrix)>,
) -> Result<(Matrix, Vec<(MatrixTriple, Matrix)>), String> {
    let mut a = x.to_owned();
    let mut caches = vec![];
    let len = params.len() - 1;

    for _ in 0..len {
        let a_prev = a.to_owned();
        let (w, b) = params.remove(0);
        let (new_a, cache) = linear_activation_forward(a_prev, w, b, ActivationFn::Relu);
        a = new_a.to_owned();
        caches.push(cache);
    }

    let (w, b) = params.remove(0);
    let (al, cache) = linear_activation_forward(a, w, b, ActivationFn::Sigmoid);
    caches.push(cache);

    Ok((al, caches))
}

// ======================================================================

#[cfg(test)]
mod tests {
    use super::super::shared;
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!((sigmoid(1.0) * 10000.0).round() / 10000.0, 0.7311);
        assert_eq!((sigmoid(-1.0) * 10000.0).round() / 10000.0, 0.2689);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(-1.0), 0.0);
    }

    #[test]
    fn test_init_deep_nn_params() {
        let nn1 = init_deep_nn_params(vec![3, 2, 2, 1]).unwrap();
        assert_eq!(nn1[0].0.shape(), [2, 3]);
        assert_eq!(nn1[0].1.shape(), [2, 1]);
        assert_eq!(nn1[1].0.shape(), [2, 2]);
        assert_eq!(nn1[1].1.shape(), [2, 1]);
        assert_eq!(nn1[2].0.shape(), [1, 2]);
        assert_eq!(nn1[2].1.shape(), [1, 1]);
    }

    #[test]
    fn test_linear_forward() {
        let prec = 100000.0;
        let round = |f: f32| (f * prec).round() / prec;

        let a = ndarray::arr2(&[
            [1.62434536, -0.61175641],
            [-0.52817175, -1.07296862],
            [0.86540763, -2.3015387],
        ]);
        let b = ndarray::arr2(&[[1.74481176, -0.7612069, 0.3190391]]);
        let c = ndarray::arr2(&[[-0.24937038]]);

        let (z, _) = linear_forward(a, b, c);
        let expected = ndarray::arr2(&[[3.26295337, -1.23429987]]);

        assert_eq!(expected.shape(), z.shape());
        assert_eq!(round(expected[[0, 0]]), round(z[[0, 0]]));
        assert_eq!(round(expected[[0, 1]]), round(z[[0, 1]]));
    }

    #[test]
    fn test_linear_activation_forward() {
        // inputs
        let get_awb = || {
            let a = ndarray::arr2(&[
                [-0.41675785, -0.05626683],
                [-2.1361961, 1.64027081],
                [-1.79343559, -0.84174737],
            ]);
            let w = ndarray::arr2(&[[0.50288142, -1.24528809, -1.05795222]]);
            let b = ndarray::arr2(&[[-0.90900761]]);

            (a, w, b)
        };

        // expected
        let a_sig = ndarray::arr2(&[[0.96890023, 0.11013289]]);
        let a_rel = ndarray::arr2(&[[3.43896131, 0.0]]);

        let (a, w, b) = get_awb();
        let (m_r, _) = linear_activation_forward(a, w, b, ActivationFn::Relu);
        shared::assert_matrices_eq(&m_r, &a_rel);

        let (a, w, b) = get_awb();
        let (m_s, _) = linear_activation_forward(a, w, b, ActivationFn::Sigmoid);
        shared::assert_matrices_eq(&m_s, &a_sig);
    }

    #[test]
    fn test_linear_backward() {
        // inputs
        let z = ndarray::arr2(&[
            [1.62434536, -0.61175641, -0.52817175, -1.07296862],
            [0.86540763, -2.3015387, 1.74481176, -0.7612069],
            [0.3190391, -0.24937038, 1.46210794, -2.06014071],
        ]);

        let a = ndarray::arr2(&[
            [-0.3224172, -0.38405435, 1.13376944, -1.09989127],
            [-0.17242821, -0.87785842, 0.04221375, 0.58281521],
            [-1.10061918, 1.14472371, 0.90159072, 0.50249434],
            [0.90085595, -0.68372786, -0.12289023, -0.93576943],
            [-0.26788808, 0.53035547, -0.69166075, -0.39675353],
        ]);

        let w = ndarray::arr2(&[
            [-0.6871727, -0.8452056, -0.6712461, -0.0126646, -1.11731035],
            [0.2344157, 1.65980218, 0.74204416, -0.19183555, -0.88762896],
            [-0.74715829, 1.6924546, 0.05080775, -0.63699565, 0.19091548],
        ]);

        let b = ndarray::arr2(&[[2.10025514], [0.12015895], [0.61720311]]);

        // expected output
        let exp_da = ndarray::arr2(&[
            [-1.15171336, 0.06718465, -0.3204696, 2.09812712],
            [0.60345879, -3.72508701, 5.81700741, -3.84326836],
            [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
            [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
            [-2.52214926, 2.67882552, -0.67947465, 1.48119548],
        ]);

        let exp_dw = ndarray::arr2(&[
            [0.07313866, -0.0976715, -0.87585828, 0.73763362, 0.00785716],
            [0.85508818, 0.37530413, -0.59912655, 0.71278189, -0.58931808],
            [0.97913304, -0.24376494, -0.08839671, 0.55151192, -0.1029090],
        ]);

        let exp_db = ndarray::arr2(&[[-0.14713786], [-0.11313155], [-0.13209101]]);

        let linear_cache = (a, w, b);

        // test
        let (da, dw, db) = linear_backward(z, linear_cache).unwrap();

        shared::assert_matrices_eq(&da, &exp_da);
        shared::assert_matrices_eq(&dw, &exp_dw);
        shared::assert_matrices_eq(&db, &exp_db);
    }

    #[test]
    fn test_linear_activation_backward() {
        // inputs
        let al1 = ndarray::arr2(&[[-0.41675785, -0.05626683]]);
        let al2 = ndarray::arr2(&[[-0.41675785, -0.05626683]]);
        let get_linear_cache = || {
            let a = ndarray::arr2(&[
                [-2.1361961, 1.64027081],
                [-1.79343559, -0.84174737],
                [0.50288142, -1.24528809],
            ]);
            let w = ndarray::arr2(&[[-1.05795222, -0.90900761, 0.55145404]]);
            let b = ndarray::arr2(&[[2.29220801]]);

            (a, w, b)
        };
        let activation_cache1 = ndarray::arr2(&[[0.04153939, -1.11792545]]);
        let activation_cache2 = ndarray::arr2(&[[0.04153939, -1.11792545]]);

        let linear_cache1 = get_linear_cache();
        let linear_cache2 = get_linear_cache();

        // expected outputs
        let exp_da_p_s = ndarray::arr2(&[
            [0.11017994, 0.01105339],
            [0.09466817, 0.00949723],
            [-0.05743092, -0.00576154],
        ]);
        let exp_dw_s = ndarray::arr2(&[[0.10266786, 0.09778551, -0.01968084]]);
        let exp_db_s = ndarray::arr2(&[[-0.05729622]]);
        let exp_da_p_r = ndarray::arr2(&[[0.44090989, 0.0], [0.37883606, 0.0], [-0.2298228, 0.0]]);
        let exp_dw_r = ndarray::arr2(&[[0.44513824, 0.37371418, -0.10478989]]);
        let exp_db_r = ndarray::arr2(&[[-0.20837892]]);

        // test
        let cache1 = (linear_cache1, activation_cache1);
        let (da_p_r, dw_r, db_r) =
            linear_activation_backward(al1, cache1, ActivationFn::Relu).unwrap();

        let cache2 = (linear_cache2, activation_cache2);
        let (da_p_s, dw_s, db_s) =
            linear_activation_backward(al2, cache2, ActivationFn::Sigmoid).unwrap();

        shared::assert_matrices_eq(&da_p_r, &exp_da_p_r);
        shared::assert_matrices_eq(&dw_r, &exp_dw_r);
        shared::assert_matrices_eq(&db_r, &exp_db_r);

        shared::assert_matrices_eq(&da_p_s, &exp_da_p_s);
        shared::assert_matrices_eq(&dw_s, &exp_dw_s);
        shared::assert_matrices_eq(&db_s, &exp_db_s);
    }

    #[test]
    fn test_l_model_forward() {
        // inputs
        let a = ndarray::arr2(&[
            [-0.31178367, 0.72900392, 0.21782079, -0.8990918],
            [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
            [1.63929108, -0.4298936, 2.63128056, 0.60182225],
            [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
            [0.07612761, -0.15512816, 0.63422534, 0.810655],
        ]);

        let w1 = ndarray::arr2(&[
            [0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384],
            [-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953],
            [-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143],
            [-0.02195668, -2.1271446, -0.83440747, -0.4655083, 0.23371059],
        ]);
        let b1 = ndarray::arr2(&[[1.38503523], [-0.51962709], [-0.78015214], [0.95560959]]);

        let w2 = ndarray::arr2(&[
            [-0.12673638, -1.36861282, 1.21848065, -0.85750144],
            [-0.56147088, -1.0335199, 0.35877096, 1.07368134],
            [-0.37550472, 0.39636757, -0.47144628, 2.33660781],
        ]);
        let b2 = ndarray::arr2(&[[1.50278553], [-0.59545972], [0.52834106]]);

        let w3 = ndarray::arr2(&[[0.9398248, 0.42628539, -0.75815703]]);
        let b3 = ndarray::arr2(&[[-0.16236698]]);

        let mut vec = vec![];
        vec.push((w1, b1));
        vec.push((w2, b2));
        vec.push((w3, b3));

        // expected output
        let exp_al = ndarray::arr2(&[[0.03921668, 0.70498921, 0.19734387, 0.04728177]]);

        // test
        match l_model_forward(a, &mut vec) {
            Ok((al, _)) => shared::assert_matrices_eq(&al, &exp_al),
            Err(msg) => println!("{}", msg),
        }
    }
}
