use std::ops::Sub;

use ndarray::{s, Array1, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use ndarray::{Array2, Array3};

pub use crate::approx::*;
pub use crate::patch::Patch;

/// Convolutional layer of a convolutional neural network. Extracts features
/// from input images into a series of 'kernels', which are dimensionally-reduced
/// floating-point matrices capturing spacial features from images during training and 
/// can be used to map original image data to modified or processed data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Convolution {
  num_kernels: usize,
  // Row dimensions of the kernels
  rows: usize,
  // Column dimensions of the kernels
  cols: usize,
  // Kernel data
  kernels: Array3<f64>
}

/// This context is used to capture processing information during forward propgation
/// which is then later used in backpropagation for training.
pub struct ConvolutionContext<'a> {
  alpha: f64,
  input: Option<&'a Array2<f64>>,
  patches: Option<Vec<Patch>>
}

impl<'a> ConvolutionContext<'a> {
  pub fn
  init (alpha: f64) -> Self
  {
    ConvolutionContext { 
      alpha: alpha, 
      input: None,
      patches: None
    }
  }
}

impl Convolution {
  pub fn 
  init (num_kernels: usize, rows: usize, cols: usize) -> Self
  {
    let size = rows * cols;
    let mut kernels: Array3<f64> = Array3::zeros((num_kernels, rows, cols));

    for i in 0..num_kernels {
      // Initialize kernel data with random values from a uniform distribution over 0
      let row = Array2::random((rows, cols), Uniform::new(0., (rows * cols) as f64)) / size.pow(2) as f64;
      kernels.slice_mut(s![i, .., ..]).assign(&row);
    };

    Convolution { 
      num_kernels:num_kernels, 
      rows: rows, 
      cols: cols, 
      kernels: kernels
    }
  }

  /// Initializes this Convolution with predictable kernel values to facilitate unit tests
  pub fn 
  init_for_test (num_kernels: usize, rows: usize, cols: usize) -> Self
  {
    let mut kernels: Array3<f64> = Array3::zeros((num_kernels, rows, cols));

    for i in 0..num_kernels {
      let row = Array2::ones((rows, cols));
      kernels.slice_mut(s![i, .., ..]).assign(&row);
    };

    Convolution { 
      num_kernels:num_kernels, 
      rows: rows, 
      cols: cols, 
      kernels: kernels
    }
  }

  /// A patch is a concept from image processing which captures subregions of the image pixels, 
  /// used to capture groups of spatial features from the underlying image.
  pub fn
  patches (&mut self, image: &Array2<f64>) -> Vec<Patch>
  {
    let mut data: Vec<Patch> = Vec::new();

    for x in 0..(image.shape()[0] - self.rows + 1) {
      for y in 0..(image.shape()[1] - self.cols + 1) {
        let p = Patch {
          x: x,
          y: y,
          data: image.slice(
            // Collect a 2-D slice of the image from 
            // [(x -> x + kernel rows), (y -> y + kernel_cols)]
            s![
              x..(x + self.rows), 
              y..(y + self.cols)
            ])
            .to_owned()
            .insert_axis(Axis(2))
        };

        data.push(p)
      }
    }

    data
  }

  /// Runs forward propagation in this layer of the network. Generates patches over the input, 
  /// multiplies each patch against the set of kernels and returns the results.
  pub fn 
  forward_propagation<'a> (
    &mut self, 
    input: &'a Array2<f64>,
    ctx: &mut ConvolutionContext<'a>
  ) -> Array3<f64> 
  {
    let mut a: Array3<f64> = Array3::zeros((
      input.shape()[0] - self.rows + 1, 
      input.shape()[1] - self.cols + 1, 
      self.num_kernels
    ));

    let patches = self.patches(input);

    for p in patches.iter() {
      let m: Array1<f64> = self.kernels.axis_iter(Axis(0))
          .map(|kernel| (p.data.index_axis(Axis(2), 0).to_owned() * kernel).sum())
          .collect();

      a.slice_mut(s![p.x, p.y, ..]).assign(&m);
    }

    ctx.input = Some(input);
    ctx.patches = Some(patches);

    a
  }

  /// Runs backward propagation in this layer of the network. Given previously-computed context 
  /// from an earlier forward-propagation run, updates this layer's kernels with the gradient of the 
  /// network loss function.
  pub fn 
  back_propagation (&mut self, input: &Array3<f64>, ctx: &ConvolutionContext) -> Array3<f64>
  { 
    let mut output: Array3<f64> = Array3::zeros([self.kernels.shape()[0], self.rows, self.cols]);

    for p in ctx.patches.as_ref().unwrap() {
      for k in 0..self.num_kernels {
        output
        .slice_mut(s![k, .., ..])
        .scaled_add(
          input[[p.x, p.y, k]], 
          &p.data.index_axis(Axis(2), 0)
        );
      }
    }

    self.kernels = self.kernels.clone().sub(ctx.alpha * &output);

    output
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::array;

  #[test]
  fn 
  test_init_dimensions () 
  {
    let num_kernels = 3;
    let rows = 4;
    let cols = 5;
    let conv = Convolution::init(num_kernels, rows, cols);

    assert_eq!(conv.num_kernels, num_kernels);
    assert_eq!(conv.rows, rows);
    assert_eq!(conv.cols, cols);
    assert_eq!(conv.kernels.dim(), (num_kernels, rows, cols));
  }

  #[test]
  fn 
  test_init_kernels () 
  {
    let num_kernels = 3;
    let rows = 4;
    let cols = 5;
    let conv = Convolution::init(num_kernels, rows, cols);

    // Ensuring all kernels are not zero-matrices
    for i in 0..num_kernels {
        assert_ne!(conv.kernels.slice(s![i, .., ..]).sum(), 0.0);
    }
  }

  #[test]
  fn 
  test_init_unique_kernels () 
  {
    let num_kernels = 3;
    let rows = 4;
    let cols = 5;
    let conv = Convolution::init(num_kernels, rows, cols);

    // Ensure that all kernels are unique (This might fail by chance,
    // because random doesn't guarantee unique, but it's highly unlikely)
    for i in 0..num_kernels {
        for j in i+1..num_kernels {
            assert_ne!(
                conv.kernels.slice(s![i, .., ..]),
                conv.kernels.slice(s![j, .., ..]),
                "Kernels {} and {} are equal, which is unexpected!", i, j
            );
        }
    }
  }

  #[test]
  fn 
  test_patches_length ()
  {
    let mut conv = Convolution::init(3, 2, 2); // assuming you have this init method
    let image = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let patches = conv.patches(&image);

    // Expected 4 patches for a 3x3 image with 2x2 patches: (0,0), (0,1), (1,0), (1,1)
    assert_eq!(patches.len(), 4);
  }

  #[test]
  fn 
  test_patches_shape () 
  {
    let mut conv = Convolution::init(3, 2, 2);
    let image = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let patches = conv.patches(&image);

    for patch in patches {
        assert_eq!(patch.data.shape(), [2, 2, 1]);
    }
  }

  #[test]
  fn 
  test_patches_contain_expected_values () 
  {
    let mut conv = Convolution::init(3, 2, 2);
    let image = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let patches = conv.patches(&image);

    // Manually verify the content of each patch
    let expected_patches: Vec<Array2<f64>> = vec![
        array![[1., 2.], [4., 5.]],
        array![[2., 3.], [5., 6.]],
        array![[4., 5.], [7., 8.]],
        array![[5., 6.], [8., 9.]],
    ];

    for (p, e) in patches.iter().zip(expected_patches.iter()) {
      for (e1, e2) in p.data.iter().zip(e.iter()) {
        assert!(approx_equal(*e1, *e2, 1e-6));
      }
    }
  }

  #[test]
  fn 
  test_forward_propagation_shape () 
  {
    let mut conv = Convolution::init(3, 2, 2);
    let image = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let mut ctx = ConvolutionContext::init(0.01);
    let result = conv.forward_propagation(&image, &mut ctx);

    assert_eq!(result.shape(), &[2, 2, 3]);
    assert!(ctx.input.is_some());
    assert_eq!(ctx.input.unwrap(), image);
  }

  #[test]
  fn 
  test_forward_propagation_values () 
  {
    let image = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let mut conv = Convolution::init_for_test(3, 2, 2);
    let mut ctx = ConvolutionContext::init(0.01);
    let result = conv.forward_propagation(&image, &mut ctx);

    let expected = array![
      [[12.0, 12.0, 12.0], [16.0, 16.0, 16.0]], 
      [[24.0, 24.0, 24.0], [28.0, 28.0, 28.0]]
    ];

    assert!(expected.shape() == result.shape());

    println!("expected: {:?}", expected);
    println!("result: {:?}", result);
    for (r, e) in result.iter().zip(expected.iter()) {
      assert!(approx_equal(*r, *e, 1e-6));
    }
  }

  #[test]
  fn 
  test_back_propagation_updates_kernels () 
  {
    let image = array![[0.5, 1.0, 1.5],
      [0.7, 1.2, 1.1],
      [0.9, 1.3, 1.4]];
    let mut conv = Convolution::init_for_test(3, 2, 2);
    let mut ctx = ConvolutionContext::init(0.01);

    let dE_dY: Array3<f64>  = Array3::from_elem((3, 3, 3), 1.0);
    
    // Store a copy of the original kernels to compare with after the update
    let original_kernels = conv.kernels.clone();

    conv.forward_propagation(&image, &mut ctx);
    conv.back_propagation(&dE_dY, &ctx);

    // The kernels should have changed after calling back_propagation
    assert_ne!(original_kernels, conv.kernels);
  }

  #[test]
  fn 
  test_back_propagation_with_zero_alpha () 
  {
    let image = array![[0.5, 1.0, 1.5],
      [0.7, 1.2, 1.1],
      [0.9, 1.3, 1.4]];
    let mut conv = Convolution::init_for_test(3, 2, 2);
    let mut ctx = ConvolutionContext::init(0.00);

    let dE_dY: Array3<f64>  = Array3::from_elem((3, 3, 3), 1.0);
    
    let original_kernels = conv.kernels.clone();

    // When alpha is 0, the kernels should not change
    conv.forward_propagation(&image, &mut ctx);
    conv.back_propagation(&dE_dY, &ctx);

    assert_eq!(original_kernels, conv.kernels);
  }

  #[test]
  fn 
  test_back_propagation_values() 
  {
    let image = array![[0.5, 1.0, 1.5],
      [0.7, 1.2, 1.1],
      [0.9, 1.3, 1.4]];
    let mut conv = Convolution::init_for_test(3, 2, 2);
    let mut ctx = ConvolutionContext::init(0.01);

    let dE_dY: Array3<f64> = Array3::from_elem((3, 3, 3), 1.0);
    
    conv.forward_propagation(&image, &mut ctx);
    let result = conv.back_propagation(&dE_dY.clone(), &ctx);
    let expected = array!
      [
        [[3.4, 4.8], [4.1, 5.0]], 
        [[3.4, 4.8], [4.1, 5.0]], 
        [[3.4, 4.8], [4.1, 5.0]], 
      ];

    assert!(expected.shape() == result.shape());

    for (r, e) in result.iter().zip(expected.iter()) {
      assert!(approx_equal(*r, *e, 1e-6));
    }
  }
}
