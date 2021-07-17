use super::matrix2d::{divide, log, new_rand, sum_keepdims};
use ndarray::{arr2, Array, Array2};
use std::f32::consts::E;

#[allow(dead_code)]
pub type Matrix2D = Array2<f32>;

#[allow(dead_code)]
pub type MatrixDouble = (Matrix2D, Matrix2D);

#[allow(dead_code)]
pub type MatrixTriple = (Matrix2D, Matrix2D, Matrix2D);

#[derive(Clone, Copy)]
pub enum ActivationFn {
    Relu,
    Sigmoid,
}

fn safe_nan(n: f32) -> f32 {
    if n.is_nan() {
        0.0
    } else {
        n
    }
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
pub fn sigmoid_m(z: Matrix2D) -> (Matrix2D, Matrix2D) {
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
pub fn relu_m(z: Matrix2D) -> (Matrix2D, Matrix2D) {
    let a = z.map(|f| relu(*f));
    let activation_cache = z;

    (a, activation_cache)
}

// This function generates an N layer NN from a set of integers specifying the
// respective layer sizes in order.
#[allow(dead_code)]
pub fn init_deep_nn_params(layers: Vec<usize>) -> Result<Vec<(Matrix2D, Matrix2D)>, String> {
    let mut vec = vec![];

    for i in 1..layers.len() {
        let this_l = layers[i - 1];
        let next_l = layers[i];
        let w = new_rand(next_l, this_l);
        let b = Array2::<f32>::zeros((next_l, 1));
        vec.push((w, b));
    }

    Ok(vec)
}

// Linear forward is the preceding step to calculating activation, (a, w, b) is the cache tuple.
pub fn linear_forward(a: &Matrix2D, w: Matrix2D, b: Matrix2D) -> (Matrix2D, MatrixTriple) {
    let z = w.dot(a) + &b;
    let cache = (a.to_owned(), w, b);

    (z, cache)
}

// Linear->Activation forward, combines linear_forward with activation.
#[allow(dead_code)]
pub fn linear_activation_forward(
    a_prev: &Matrix2D,
    w: Matrix2D,
    b: Matrix2D,
    act_fn: ActivationFn,
) -> (Matrix2D, (MatrixTriple, Matrix2D)) {
    let (z, linear_cache) = linear_forward(a_prev, w, b);
    let (a, activation_cache) = match act_fn {
        ActivationFn::Relu => relu_m(z),
        ActivationFn::Sigmoid => sigmoid_m(z),
    };

    (a, (linear_cache, activation_cache))
}

// backward propagation
#[allow(dead_code)]
pub fn linear_backward(dz: Matrix2D, cache: &MatrixTriple) -> Result<MatrixTriple, String> {
    let (a_prev, w, _) = cache;
    let m = a_prev.shape()[1] as f32;
    let one = arr2(&[[1.0]]);
    let dw = &one / m * dz.dot(&a_prev.t());
    let db = &one / m * sum_keepdims(1, &dz).unwrap();
    let da_prev = w.t().dot(&dz);

    Ok((da_prev, dw, db))
}

#[allow(dead_code)]
pub fn sigmoid_backward_m(da: &Matrix2D, z_cache: &Matrix2D) -> Result<Matrix2D, String> {
    let s = 1.0 / (1.0 + z_cache.map(|x| (-x).exp()));
    let dz = da * &s * (1.0 - &s);
    if dz.shape() == z_cache.shape() {
        Ok(dz)
    } else {
        Err("dZ shape does not match cache shape".to_string())
    }
}

// relu backward propagation function
#[allow(dead_code)]
pub fn relu_backward_m(z: &Matrix2D, z_cache: &Matrix2D) -> Result<Matrix2D, String> {
    let mask = z_cache.map(|f| if *f > 0.0 { 1.0 } else { 0.0 });
    let dz = z * mask;
    if dz.shape() == z_cache.shape() {
        Ok(dz)
    } else {
        Err("Z matrix shape does not match cache shape".to_string())
    }
}

#[allow(dead_code)]
pub fn linear_activation_backward(
    da: &Matrix2D,
    cache: &(MatrixTriple, Matrix2D),
    act_fn: ActivationFn,
) -> Result<MatrixTriple, String> {
    let (linear_cache, activation_cache) = cache;

    let dz = match act_fn {
        ActivationFn::Relu => relu_backward_m(da, &activation_cache),
        ActivationFn::Sigmoid => sigmoid_backward_m(da, &activation_cache),
    };
    let (da_prev, dw, db) = linear_backward(dz.unwrap(), &linear_cache).unwrap();

    Ok((da_prev, dw, db))
}

#[allow(dead_code)]
pub fn l_model_forward(
    x: Matrix2D,
    params: &mut Vec<(Matrix2D, Matrix2D)>,
) -> Result<(Matrix2D, Vec<(MatrixTriple, Matrix2D)>), String> {
    let mut a = x.to_owned();
    let mut caches = vec![];
    let len = params.len() - 1;

    for _ in 0..len {
        let (w, b) = params.remove(0);
        let (new_a, cache) = linear_activation_forward(&a, w, b, ActivationFn::Relu);
        a = new_a;
        caches.push(cache);
    }

    let (w, b) = params.remove(0);
    let (al, cache) = linear_activation_forward(&a, w, b, ActivationFn::Sigmoid);
    caches.push(cache);

    Ok((al, caches))
}

#[allow(dead_code)]
pub fn compute_cost(al: &Matrix2D, y: &Matrix2D) -> f32 {
    let m = y.shape()[1] as f32;
    let yy = y * log(al) + (1.0 - y) * log(&(1.0 - al));

    -1.0 / m * yy.sum()
}

#[allow(dead_code)]
pub fn l_model_backward(
    al: &Matrix2D,
    y: &Matrix2D,
    caches: &mut Vec<(MatrixTriple, Matrix2D)>,
) -> Vec<MatrixTriple> {
    let mut grads = vec![];
    let l = caches.len() - 1;
    let dal = -(divide(&y, &al) - divide(&(1.0 - y), &(1.0 - al)));
    let (da, dw, db) = linear_activation_backward(&dal, &caches[l], ActivationFn::Sigmoid).unwrap();
    let mut prev_da = da;
    grads.insert(0, (prev_da.to_owned(), dw, db));

    for i in (1..=l).rev() {
        let tpl = linear_activation_backward(&prev_da, &caches[i - 1], ActivationFn::Relu).unwrap();
        prev_da = tpl.0.to_owned();
        grads.insert(0, tpl);
    }

    grads
}

#[allow(dead_code)]
pub fn update_parameters(params: &mut Vec<MatrixDouble>, grads: Vec<MatrixTriple>, rate: f32) {
    let len = params.len();
    for i in 0..len {
        let (w, b) = &params[i];
        let (_, dw, db) = &grads[i];
        params[i] = (w - rate * dw, b - rate * db);
    }
}

// ======================================================================

pub struct ModelTrainer {
    pub labels: Matrix2D,
    pub input_layer: Matrix2D,
    pub parameters: Vec<(Matrix2D, Matrix2D, ActivationFn)>,
    pub cache: Vec<(MatrixTriple, Matrix2D, ActivationFn)>,
    pub cost: f32,
}

impl ModelTrainer {
    pub fn new(
        input_layer: Matrix2D,
        layers: Vec<(usize, ActivationFn)>,
        labels: Matrix2D,
    ) -> ModelTrainer {
        let mut parameters = vec![];

        for i in 1..layers.len() {
            let (this_l, _) = layers[i - 1];
            let (next_l, af) = &layers[i];
            let w = new_rand(*next_l, this_l);
            let b = Array2::<f32>::zeros((*next_l, 1));
            parameters.push((w, b, *af));
        }

        ModelTrainer {
            labels,
            parameters,
            input_layer,
            cache: vec![],
            cost: 0.0,
        }
    }

    // Linear->Activation forward, combines linear_forward with activation.
    pub fn train(self: &mut Self, learn_rate: f32) {
        let mut al = self.input_layer.to_owned();

        for (w, b, act_fn) in &self.parameters {
            let z = w.dot(&al) + b;

            let (new_a, activation_cache) = match act_fn {
                ActivationFn::Relu => relu_m(z),
                ActivationFn::Sigmoid => sigmoid_m(z),
            };

            al = new_a;

            let linear_cache = (al.to_owned(), w.to_owned(), b.to_owned());

            self.cache.push((linear_cache, activation_cache, *act_fn));
        }

        self.cost = compute_cost(&al, &self.labels);

        // compute gradients
        let y_vec = self.labels.iter().map(|f| *f).collect::<Vec<f32>>();
        let al_vec = al.iter().map(|f| *f).collect::<Vec<f32>>();

        let dal = y_vec
            .iter()
            .zip(al_vec)
            .map(|(y, a)| safe_nan(-(y / a) - (1.0 - y) / (1.0 - a)))
            .collect::<Vec<f32>>();

        let shape = al.shape();
        let (row, col) = (shape[0], shape[1]);
        let dal = Array::from_shape_vec((row, col), dal).unwrap();

        println!("\ndAL {:?}", dal);

        let mut grads = vec![];

        self.cache
            .iter()
            .rev()
            .fold(dal, |da, (linear_cache, activation_cache, act_fn)| {
                println!("\ndA: {:?}", &da);
                println!("A cache: {:?}\n", &activation_cache);

                let dz = match act_fn {
                    ActivationFn::Sigmoid => sigmoid_backward_m(&da, &activation_cache),
                    ActivationFn::Relu => relu_backward_m(&da, &activation_cache),
                };
                let dz = dz.unwrap();
                println!("dZ shape: {:?}", dz.shape());
                println!("dZ: {:?}", dz);

                let (new_da, dw, db) = linear_backward(dz, &linear_cache).unwrap();
                grads.insert(0, (dw, db));

                new_da
            });

        self.cache = vec![];

        // update parameters
        let len = self.parameters.len();
        for i in 0..len {
            let (w, b, act_fn) = &self.parameters[i];
            let (dw, db) = &grads[i];

            println!("b {:?}", b);
            println!("db {:?}", db);

            self.parameters[i] = (w - learn_rate * dw, b - learn_rate * db, *act_fn);
        }
    }
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
        let a = ndarray::arr2(&[
            [1.62434536, -0.61175641],
            [-0.52817175, -1.07296862],
            [0.86540763, -2.3015387],
        ]);
        let b = ndarray::arr2(&[[1.74481176, -0.7612069, 0.3190391]]);
        let c = ndarray::arr2(&[[-0.24937038]]);

        let (z, _) = linear_forward(&a, b, c);
        let expected = ndarray::arr2(&[[3.26295337, -1.23429987]]);

        assert_eq!(expected.shape(), z.shape());
        shared::assert_matrices_eq(&expected, &z);
        shared::assert_matrices_eq(&expected, &z);
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
        let (m_r, _) = linear_activation_forward(&a, w, b, ActivationFn::Relu);
        shared::assert_matrices_eq(&m_r, &a_rel);

        let (a, w, b) = get_awb();
        let (m_s, _) = linear_activation_forward(&a, w, b, ActivationFn::Sigmoid);
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
        let (da, dw, db) = linear_backward(z, &linear_cache).unwrap();

        shared::assert_matrices_eq(&da, &exp_da);
        shared::assert_matrices_eq(&dw, &exp_dw);
        shared::assert_matrices_eq(&db, &exp_db);
    }

    #[test]
    fn test_linear_activation_backward() {
        // inputs
        let al = ndarray::arr2(&[[-0.41675785, -0.05626683]]);
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
            linear_activation_backward(&al, &cache1, ActivationFn::Relu).unwrap();

        let cache2 = (linear_cache2, activation_cache2);
        let (da_p_s, dw_s, db_s) =
            linear_activation_backward(&al, &cache2, ActivationFn::Sigmoid).unwrap();

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

    #[test]
    fn test_compute_cost() {
        // inputs
        let y = ndarray::arr2(&[[1.0, 1.0, 0.0]]);
        let al = ndarray::arr2(&[[0.8, 0.9, 0.4]]);
        // expected
        let exp_cost = 0.2797765635793422;
        // test
        let cost = compute_cost(&al, &y);
        assert_eq!(cost, exp_cost);
    }

    #[test]
    fn test_l_model_backward() {
        // inputs
        let al = arr2(&[[1.78862847, 0.43650985]]);
        let y = arr2(&[[1.0, 0.0]]);
        let linear_cache1 = (
            arr2(&[
                [0.09649747, -1.8634927],
                [-0.2773882, -0.35475898],
                [-0.08274148, -0.62700068],
                [-0.04381817, -0.47721803],
            ]),
            arr2(&[
                [-1.31386475, 0.88462238, 0.88131804, 1.70957306],
                [0.05003364, -0.40467741, -0.54535995, -1.54647732],
                [0.98236743, -1.10106763, -1.18504653, -0.2056499],
            ]),
            arr2(&[[1.48614836], [0.23671627], [-1.02378514]]),
        );
        let activation_cache1 = arr2(&[
            [-0.7129932, 0.62524497],
            [-0.16051336, -0.76883635],
            [-0.23003072, 0.74505627],
        ]);
        let linear_cache2 = (
            arr2(&[
                [1.97611078, -1.24412333],
                [-0.62641691, -0.80376609],
                [-2.41908317, -0.92379202],
            ]),
            arr2(&[[-1.02387576, 1.12397796, -0.13191423]]),
            arr2(&[[-1.62328545]]),
        );
        let activation_cache2 = arr2(&[[0.64667545, -0.35627076]]);
        let mut caches = vec![
            (linear_cache1, activation_cache1),
            (linear_cache2, activation_cache2),
        ];

        // expected
        let exp_da0 = arr2(&[
            [0.0, 0.522579010],
            [0.0, -0.32692060],
            [0.0, -0.32070404],
            [0.0, -0.74079187],
        ]);
        let exp_dw1 = arr2(&[
            [0.41010002, 0.07807203, 0.13798444, 0.10502167],
            [0.0, 0.0, 0.0, 0.0],
            [0.05283652, 0.01005865, 0.01777766, 0.0135308],
        ]);
        let exp_db1 = arr2(&[[-0.22007063], [0.0], [-0.02835349]]);
        let exp_da1 = arr2(&[
            [0.12913162, -0.44014127],
            [-0.14175655, 0.483172960],
            [0.01663708, -0.05670698],
        ]);
        let exp_dw2 = arr2(&[[-0.39202432, -0.13325855, -0.04601089]]);
        let exp_db2 = arr2(&[[0.15187861]]);

        // test
        let grads = l_model_backward(&al, &y, &mut caches);
        let (da0, dw1, db1) = &grads[0];
        let (da1, dw2, db2) = &grads[1];

        shared::assert_matrices_eq(&da0, &exp_da0);
        shared::assert_matrices_eq(&dw1, &exp_dw1);
        shared::assert_matrices_eq(&db1, &exp_db1);

        shared::assert_matrices_eq(&da1, &exp_da1);
        shared::assert_matrices_eq(&dw2, &exp_dw2);
        shared::assert_matrices_eq(&db2, &exp_db2);
    }

    #[test]
    fn test_update_parameters() {
        // inputs
        let w1 = arr2(&[
            [-0.41675785, -0.05626683, -2.1361961, 1.64027081],
            [-1.79343559, -0.84174737, 0.50288142, -1.24528809],
            [-1.05795222, -0.90900761, 0.55145404, 2.29220801],
        ]);
        let b1 = arr2(&[[0.04153939], [-1.11792545], [0.53905832]]);

        let w2 = arr2(&[[-0.5961597, -0.0191305, 1.17500122]]);
        let b2 = arr2(&[[-0.74787095]]);
        let mut parameters = vec![(w1, b1), (w2, b2)];

        let da0 = arr2(&[[0.0]]);
        let dw1 = arr2(&[
            [1.78862847, 0.43650985, 0.09649747, -1.8634927],
            [-0.2773882, -0.35475898, -0.08274148, -0.62700068],
            [-0.04381817, -0.47721803, -1.31386475, 0.88462238],
        ]);
        let db1 = arr2(&[[0.88131804], [1.70957306], [0.05003364]]);

        let da1 = arr2(&[[0.0]]);
        let dw2 = arr2(&[[-0.40467741, -0.54535995, -1.54647732]]);
        let db2 = arr2(&[[0.98236743]]);
        let grads = vec![(da0, dw1, db1), (da1, dw2, db2)];

        // expected
        let exp_w1 = arr2(&[
            [-0.59562069, -0.09991781, -2.14584584, 1.82662008],
            [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
            [-1.0535704, -0.86128581, 0.68284052, 2.20374577],
        ]);
        let exp_b1 = arr2(&[[-0.04659241], [-1.28888275], [0.53405496]]);
        let exp_w2 = arr2(&[[-0.55569196, 0.0354055, 1.32964895]]);
        let exp_b2 = arr2(&[[-0.84610769]]);

        // test
        update_parameters(&mut parameters, grads, 0.1);
        let (w1, b1) = &parameters[0];
        let (w2, b2) = &parameters[1];
        shared::assert_matrices_eq(&exp_w1, w1);
        shared::assert_matrices_eq(&exp_b1, b1);
        shared::assert_matrices_eq(&exp_w2, w2);
        shared::assert_matrices_eq(&exp_b2, b2);
    }
}
