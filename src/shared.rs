use super::ml::Matrix2D;

#[allow(dead_code)]
pub fn round(f: f32) -> f32 {
    let prec = 100000.0;
    (f * prec).round() / prec
}

#[allow(dead_code)]
pub fn assert_matrices_eq(mat: &Matrix2D, exp_mat: &Matrix2D) {
    let y = mat.shape()[0];
    let x = mat.shape()[1];

    for (i, j) in (0..y).zip(0..x) {
        assert_eq!(round(mat[[i, j]]), round(exp_mat[[i, j]]));
    }
}
