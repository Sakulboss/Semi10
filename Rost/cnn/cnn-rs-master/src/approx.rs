pub const EPSILON: f64 = 1e-6;

pub fn 
approx_equal (a: f64, b: f64, ep: f64) -> bool 
{
  (a - b).abs() <= ep
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn 
  test_equal () 
  {
    assert!(approx_equal(1.0, 0.99, 0.1));
    assert!(approx_equal(1.0, 0.999, 0.01));
    assert!(approx_equal(1.0, 0.9999, 0.001));

    assert!(approx_equal(1.0, 1.01, 0.1));
    assert!(approx_equal(1.0, 1.001, 0.01));
    assert!(approx_equal(1.0, 1.0001, 0.001));
  }

  #[test]
  fn 
  test_not_equal () 
  {
    assert!(!approx_equal(1.0, 0.99, 0.001));
    assert!(!approx_equal(1.0, 0.999, 0.0001));
    assert!(!approx_equal(1.0, 0.9999, 0.00001));

    assert!(!approx_equal(1.0, 1.001, 0.0001));
    assert!(!approx_equal(1.0, 1.0001, 0.00001));
    assert!(!approx_equal(1.0, 1.00001, 0.000001));
  }
}