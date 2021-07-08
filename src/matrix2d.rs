use rand::prelude::*;

fn new_vec_1d_rnd(l: u16, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..l).map(|_| rng.gen::<f32>() * scale).collect()
}

pub fn new_fill_rand(r: u16, c: u16, scale: f32) -> Vec<Vec<f32>> {
    (0..r).map(|_| new_vec_1d_rnd(c, scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_fill_rand() {
        let m = new_fill_rand(2, 2, 0.01);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 2);
        assert!(m[0][0] <= 0.01 && m[0][0] >= 0.0);
        assert!(m[0][1] <= 0.01 && m[0][1] >= 0.0);
        assert!(m[1][0] <= 0.01 && m[1][0] >= 0.0);
        assert!(m[1][1] <= 0.01 && m[1][1] >= 0.0);
    }
}
