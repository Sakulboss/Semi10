use serde::{Deserialize, Serialize};

const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Configuration variables which manage overall model training properties
#[derive(Clone, Serialize, Deserialize)]
pub struct HyperParams 
{
  pub epochs: usize,
  pub learning_rate: f64
}

pub struct Args 
{
  pub training_image_path: Option<String>,
  pub training_labels_path: Option<String>,
  pub test_image_path: Option<String>,
  pub test_labels_path: Option<String>,
  pub save: bool,
  pub load: Option<String>,
  pub input_dir: Option<String>,
  pub cheat: bool,
  pub hyper_params: HyperParams
}

impl HyperParams 
{
  pub fn 
  new (epochs_opt: Option<String>, learning_rate_opt: Option<String>) -> Self
  {
    let epochs = match epochs_opt {
      Some(num_epochs) => {
        num_epochs.parse::<usize>().expect("number of epochs must be an integer")
      }
      _ => DEFAULT_EPOCHS
    };

    let learning_rate = match learning_rate_opt {
      Some(learning_rate) => {
        learning_rate.parse::<f64>().expect("learning rate must be a float")
      }
      _ => DEFAULT_LEARNING_RATE
    };

    HyperParams 
    { 
      epochs: epochs, 
      learning_rate: learning_rate
    }
  }
}

impl Args 
{
  pub fn 
  new (
    training_image_path: Option<String>,
    training_lables_path: Option<String>,
    test_image_path: Option<String>,
    test_lables_path: Option<String>,
    save_opt: Option<bool>,
    load_opt: Option<String>,
    input_dir_opt: Option<String>,
    cheat_opt: Option<bool>,
    epochs_opt: Option<String>,
    learning_rate_opt: Option<String>
  ) -> Self 
  {
     Args { 
      training_image_path: training_image_path, 
      training_labels_path: training_lables_path,
      test_image_path: test_image_path, 
      test_labels_path: test_lables_path,
      save: save_opt.unwrap_or(true),
      load: load_opt,
      cheat: cheat_opt.unwrap_or(false),
      input_dir: input_dir_opt,
      hyper_params: HyperParams::new(epochs_opt, learning_rate_opt)
    } 
  }
}