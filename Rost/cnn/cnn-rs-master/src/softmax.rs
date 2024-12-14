use std::ops::{Add, Sub};

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::{Uniform, Normal};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub use crate::approx::*;

/// The Softmax layer computes a final probability distribution over
/// the output dimension, based on inputs from the max pooling and 
/// convlutional layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Softmax
{
  pub weights: Array2<f64>,
  pub bias: Array1<f64>
}

/// Caches data used during forward propagation which is necessary
/// for backward propagation during training
pub struct SoftmaxContext<'a> {
  alpha: f64,
  input: Option<&'a Array3<f64>>,
  flattened: Option<Array1<f64>>,
  dot_result: Option<Array1<f64>>,
  output: Option<Array1<f64>>
}

impl<'a> SoftmaxContext<'a> {
  pub fn 
  init (alpha: f64) -> Self 
  {
    SoftmaxContext { 
      alpha: alpha, 
      input: None, 
      flattened: None, 
      dot_result: None, 
      output: None
    }
  }
}

/// The softmax function converts a vector of K real numbers into a probability distribution of 
/// K possible outcomes
/// 
/// https://en.wikipedia.org/wiki/Softmax_function
fn 
softmax (x: &Array1<f64>) -> Array1<f64> 
{
  let x_exp = x.mapv(f64::exp);
  let sum_exp = x_exp.sum_axis(Axis(0)).insert_axis(Axis(0));
  &x_exp / &sum_exp
}

impl Softmax {
  pub fn 
  init (input_size: usize, output_size: usize) -> Self 
  {
    let mut rng = StdRng::from_entropy();
    let mut weights = Array2::zeros((input_size, output_size)) / input_size as f64;
    let bias = Array1::zeros(output_size);

    // He weights initialization
    let std = (2.0 / weights.len() as f64).sqrt();
    let distribution = Normal::new(0.0, std).expect("Failed to create distribution");

    for val in weights.iter_mut() {
      *val = rng.sample(distribution);
    }

    Softmax {
      weights: weights,
      bias: bias
    }
  }

  /// Similar initialization as above, but assigns stable values to weights to facilitate 
  /// unit testing.
  pub fn 
  init_for_test (input_size: usize, output_size: usize) -> Self 
  {
    let weights = Array2::ones((input_size, output_size)) / input_size as f64;
    let bias = Array1::zeros(output_size);

    Softmax {
      weights: weights,
      bias: bias
    }
  }

  /// Runs forward propagation in this layer of the network. Flattens input, computes probabilities of 
  /// outputs based on the dot product with this layer's weights and softmax outputs.
  pub fn 
  forward_propagation<'a> (&mut self, input: &'a Array3<f64>, ctx: &mut SoftmaxContext<'a>) -> Array1<f64> 
  {
    let flattened: Array1<f64> = input.to_owned().into_shape((input.len(),)).unwrap();
    let dot_result = flattened.dot(&self.weights).add(&self.bias);
    let probabilities = softmax(&dot_result);

    ctx.input = Some(input);
    ctx.flattened = Some(flattened);
    ctx.dot_result = Some(dot_result);
    ctx.output = Some(probabilities.clone());

    probabilities
  }

  /// Runs backward propagation in this layer of the network. Computes and returns the gradient of the loss
  /// based on weights, bias, and input from earlier layers.
  pub fn 
  back_propagation (&mut self, input: &Array1<f64>, ctx: &SoftmaxContext) -> Array3<f64>
  {
    let mut indx = 0;

    for gradient in input {
      if (*gradient) != 0.0 {
        let exp_vector = ctx.dot_result.as_ref().unwrap().mapv(f64::exp);
        let exp_sum = exp_vector.sum();

        let mut x: Array1<f64> = -exp_vector[indx] * exp_vector.clone() / exp_sum.pow(2);
        x[indx] = exp_vector[indx] * (exp_sum - exp_vector[indx]) / exp_sum.pow(2);

        let y = x * (*gradient);
        let z = ctx.flattened.as_ref().unwrap()
          .clone().insert_axis(Axis(1))
          .dot(&y.clone().insert_axis(Axis(0)));

        let w = self.weights.clone().dot(&y);

        // Update internal weight state
        self.weights = self.weights.clone().sub(ctx.alpha * z);
        self.bias = self.bias.clone().sub(ctx.alpha * y);

        // Force the matrix into a 3D shape; kind of awkward ndarray API
        return w
          .into_shape(ctx.input.unwrap().shape()).unwrap()
          .into_dimensionality().unwrap();
      }

      indx += 1;
    }

    // When this happens, there were no previously-computed nonzero gradient values
    // from earlier layers
    Array3::zeros(ctx.input.unwrap().raw_dim())
  }
}

#[cfg(test)]
mod tests 
{
  use super::*;

  const EPSILON: f64 = 1e-10;

  fn is_approx_1(sum: f64) -> bool {
      (1.0 - EPSILON..=1.0 + EPSILON).contains(&sum)
  }

  #[test]
  fn 
  test_softmax_basic () 
  {
    // 3 element array filled with 1s
    let matrix = Array1::from_elem(3, 1.0); 

    let sm_matrix = softmax(&matrix);
    for sum in sm_matrix.sum_axis(Axis(0)) {
      assert!(is_approx_1(sum));
    }
  }
  
  #[test]
  fn 
  test_softmax_stability () 
  {
    // 3 element array with random values between 0 and 10
    let matrix = Array1::random(3, Uniform::new(0.0, 10.0)); 

    let sm_matrix = softmax(&matrix);
    for sum in sm_matrix.sum_axis(Axis(0)) {
      assert!(is_approx_1(sum));
    }
  }

  #[test]
  fn 
  test_forward_propagation_shape () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3)); // adjust shape as necessary

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);
    let result = softmax.forward_propagation(&image, &mut ctx);

    assert_eq!(result.dim(), output_size, "Output shape is incorrect");
  }

  #[test]
  fn 
  test_forward_propagation_distribution () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);

    let result = softmax.forward_propagation(&image, &mut ctx);

    // Ensure that all probabilities are between 0 and 1 and they sum to 1
    for &prob in result.iter() {
        assert!(prob >= 0.0 && prob <= 1.0);
    }
    assert!(approx_equal(result.sum(), 1.0, 1e-6));
  }

  #[test]
  fn 
  test_forward_propagation_updates_object_state () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3)); 

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);

    softmax.forward_propagation(&image, &mut ctx);

    // Ensure that `flattened` and `output` are Some after forward_propagation
    assert!(ctx.flattened.is_some(), "`flattened` not updated");
    assert!(ctx.output.is_some(), "`output` not updated");
  }

  #[test]
  fn 
  test_back_propagation_shape () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);

    softmax.forward_propagation(&image, &mut ctx);

    let input = Array1::ones(output_size);
    let result = softmax.back_propagation(&input, &ctx);

    assert_eq!(result.shape(), image.shape(), "Output shape of dE_dX is incorrect");
  }

  #[test]
  fn 
  test_back_propagation_shape_2 () 
  {
    let input_size = 13 * 13 * 16;
    let output_size = 10;
    let image = Array3::<f64>::ones((13, 13, 16));

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);
    
    softmax.forward_propagation(&image, &mut ctx);

    let input = Array1::ones(output_size);
    let result = softmax.back_propagation(&input, &ctx);

    assert_eq!(result.shape(), image.shape(), "Output shape of dE_dX is incorrect");
  }

  #[test]
  fn 
  test_back_propagation_weights_and_bias () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);

    let original_weights = softmax.weights.clone();
    let original_bias = softmax.bias.clone();
    
    softmax.forward_propagation(&image, &mut ctx);
    let input = Array1::ones(output_size);
    softmax.back_propagation(&input, &ctx);

    assert_ne!(original_weights, softmax.weights, "Weights not modified");
    assert_ne!(original_bias, softmax.bias, "Bias not modified");
  }

  #[test]
  fn
  test_back_propagation_zero_gradient ()
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);

    let original_weights = softmax.weights.clone();
    let original_bias = softmax.bias.clone();
    
    softmax.forward_propagation(&image, &mut ctx);
    let input = Array1::zeros(output_size);
    softmax.back_propagation(&input, &ctx);

    assert_eq!(original_weights, softmax.weights, "Weights modified unexpectedly");
    assert_eq!(original_bias, softmax.bias, "Bias modified unexpectedly");
  }

  #[test]
  fn
  test_back_propagation_values ()
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::init_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::init(0.01);

    softmax.forward_propagation(&image, &mut ctx);

    let mut input = Array1::zeros(output_size);
    input[0] = 0.123;

    softmax.back_propagation(&input, &ctx);

    assert!(approx_equal(softmax.weights[[0, 0]], 0.08306, 1e-6));
    assert!(approx_equal(softmax.weights[[1, 1]], 0.083469, 1e-6));
    assert!(approx_equal(softmax.weights[[2, 2]], 0.083469, 1e-6));

    assert!(approx_equal(softmax.bias[0], -0.000273, 1e-6));
    assert!(approx_equal(softmax.bias[1], 0.0001366, 1e-6));
    assert!(approx_equal(softmax.bias[2], 0.0001366, 1e-6));
  }
}