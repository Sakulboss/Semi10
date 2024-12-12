use ndarray::{s, Array1, Array3, Axis};
use ndarray_rand::RandomExt;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub use crate::approx::*;
pub use crate::patch;
pub use crate::patch::*;

/// The pooling layer extracts dominant features in the image space via the 
/// max-pooling method. This method is also effective at suppressing noise in
/// the parts of the image covered by the earlier in the earlier layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pooling {
  kernel_rows: usize,
  kernel_cols: usize
}

/// Caches data used during forward propagation which is necessary
/// for backward propagation during training
pub struct PoolingContext<'a> {
  input: Option<&'a Array3<f64>>,
  patches: Option<Vec<Patch>>
}

impl<'a> PoolingContext<'a> {
  pub fn 
  init () -> Self
  {
    PoolingContext { 
      input: None,
      patches: None
    }
  }
}

impl Pooling {
  pub fn 
  init (
    kernel_rows: usize,
    kernel_cols: usize) -> Self
  {
    Pooling { 
      kernel_rows: kernel_rows,
      kernel_cols: kernel_cols
    }
  }

  /// A patch is a concept from image processing which captures subregions of the image pixels, 
  /// used to capture groups of spatial features from the underlying image.
  pub fn 
  patches (&self, image: &Array3<f64>) -> Vec<Patch>
  {
    let mut data: Vec<Patch> = Vec::new();

    for x in 0..(image.shape()[0] / self.kernel_rows) {
      for y in 0..(image.shape()[1] / self.kernel_cols) {
        let p = Patch {
          x: x,
          y: y,
          data: image.slice(
            s![
              (x * self.kernel_rows)..(x * self.kernel_rows + self.kernel_rows), 
              (y * self.kernel_cols)..(y * self.kernel_cols + self.kernel_cols), 
              ..]
            ).to_owned()
        };

        data.push(p);
      }
    }

    data
  }

  /// Runs forward propagation in this layer of the network. Generates patches over the input, 
  /// multiplies each patch against the set of kernels and returns the results.
  pub fn 
  forward_propagation<'a> (
    &mut self, 
    input: &'a Array3<f64>, 
    ctx: &mut PoolingContext<'a>
  ) -> Array3<f64>
  {
    let mut a: Array3<f64> = Array3::zeros((
      input.shape()[0] / self.kernel_rows, 
      input.shape()[1] / self.kernel_cols, 
      input.shape()[2] // number of kernels
    ));

    let patches = self.patches(input);

    for p in patches.iter() {
      let depth = p.data.dim().2;
      let v: Vec<f64> = (0..depth).map(|i| {
        p.data.
          slice(s![.., .., i])
          .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v))
      }).collect();

      a.slice_mut(s![p.x, p.y, ..]).assign(&Array1::from(v));
    }

    ctx.input = Some(input);
    ctx.patches = Some(patches);

    a
  }

  /// Runs backward propagation in this layer of the network. Returns gradients of the loss
  /// based on kernel weights.
  pub fn 
  back_propagation (&self, input: &Array3<f64>, ctx: &PoolingContext) -> Array3<f64>
  {
    let mut output: Array3<f64> = Array3::zeros(ctx.input.unwrap().raw_dim());

    for p in ctx.patches.as_ref().unwrap() {
      let width = p.data.shape()[0];
      let height = p.data.shape()[1];
      let num_kernels = p.data.shape()[2];
      
      // Collect maximum set of values from across the patch
      let max_values = p.data
        .fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b))
        .fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));

      for i in 0..height {
        for j in 0..width {
          for k in 0..num_kernels {
            // Find the indices of the maximal value in the patch, sets the 
            // element in the output matrix to the corresponding element in the 
            // input matrix using patch coordinates
            if approx_equal(p.data[[i, j, k]], max_values[k], EPSILON) {
              output[[
                p.x * self.kernel_rows + i, 
                p.y * self.kernel_cols + j, 
                k
              ]] = input[[p.x, p.y, k]]
            }
          }
        }
      }
    }

    output
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn
  test_patches_num ()
  {
    let image_width = 6;
    let image_height = 6;
    let kernel_dim = 2;
    let image = Array3::<f64>::zeros((image_height, image_width, 3));
    let pooling = Pooling::init(kernel_dim, kernel_dim);
    let patches = pooling.patches(&image);

    // Should create 9 patches as the image size is 5x5
    assert_eq!(
      patches.len(), 
      image_height * image_width / (kernel_dim * kernel_dim), 
      "Incorrect number of patches created"
    );
  }

  #[test]
  fn 
  test_patches_size () 
  {
    let image = Array3::<f64>::zeros((6, 6, 3));
    let pooling = Pooling::init(2, 2);
    let patches = pooling.patches(&image);

    // Should create patches of size 2x2x3
    for patch in patches {
      assert_eq!(patch.data.shape(), &[2, 2, 3], "Patch has incorrect shape");
    }
  }

  #[test]
  fn 
  test_patches_indices () 
  {
    let image = Array3::<f64>::zeros((6, 6, 3));
    let pooling = Pooling::init(2, 2);
    let patches = pooling.patches(&image);

    // Check whether patches have correct x and y
    for (i, patch) in patches.iter().enumerate() {
      assert_eq!(
        patch.x, i / 3,
        "Incorrect x index for patch {}",
        i
      );
      assert_eq!(
        patch.y, i % 3,
        "Incorrect y index for patch {}",
        i
      );
    }
  }

  #[test]
  fn 
  test_patches_data () 
  {
    let image = Array3::<f64>::from_shape_vec((6, 6, 3), (0..108).map(|x| x as f64).collect()).unwrap();
    let pooling = Pooling::init(2, 2);
    let patches = pooling.patches(&image);

    assert_eq!(
      patches[0].data, 
      Array3::from_shape_vec((2, 2, 3), vec![
        0., 1., 2., 
        3., 4., 5., 
        18., 19., 20., 
        21., 22., 23.]
      ).unwrap(),
      "Data in patch 0 is incorrect"
    );
  }

  #[test]
  fn 
  test_forward_propagation_output_shape () 
  {
    let image = Array3::<f64>::zeros((26, 26, 16));
    let mut pooling = Pooling::init(2, 2);
    let mut ctx = PoolingContext::init();

    let result = pooling.forward_propagation(&image, &mut ctx);
    
    assert_eq!(result.dim(), (13, 13, 16), "Output has incorrect shape");
  }

  #[test]
  fn 
  test_forward_propagation_values () 
  {
    // Creating an image with sequential values for easy debugging
    let image = Array3::from_shape_vec(
        (6, 6, 3), 
        (0..108).map(|x| x as f64).collect()
    ).unwrap();

    let mut pooling = Pooling::init(2, 2);
    let mut ctx = PoolingContext::init();

    let result = pooling.forward_propagation(&image, &mut ctx);

    assert!(approx_equal(result[[0, 0, 0]], 21.0, 1e-6));
    assert!(approx_equal(result[[1, 1, 1]], 64.0, 1e-6));
    assert!(approx_equal(result[[2, 2, 2]], 107.0, 1e-6));
  }

  #[test]
  fn 
  test_back_propagation_output_shape () 
  {
    let image = Array3::<f64>::zeros((26, 26, 16));
    let mut pooling = Pooling::init(2, 2);
    let mut ctx = PoolingContext::init();

    let dE_dY = Array3::<f64>::zeros((26, 26, 16));

    pooling.forward_propagation(&image, &mut ctx);
    let result = pooling.back_propagation(&dE_dY, &ctx);

    assert_eq!(result.dim(), (26, 26, 16), "Output has incorrect shape");
  }
}
