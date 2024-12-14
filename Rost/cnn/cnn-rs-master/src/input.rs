pub struct InputData {
  pub training_images: Option<Vec<u8>>,
  pub training_labels: Option<Vec<u8>>,
  pub test_images: Option<Vec<u8>>,
  pub test_labels: Option<Vec<u8>>
}

impl InputData {
  pub fn
  default () -> Self {
    InputData {
      training_images: None, 
      training_labels: None,
      test_images: None, 
      test_labels: None 
    }
  }
  pub fn 
  new (
    training_images: Option<Vec<u8>>,
    training_labels: Option<Vec<u8>>,
    test_images: Option<Vec<u8>>,
    test_labels: Option<Vec<u8>>
  ) -> Self {
    InputData { 
      training_images: training_images, 
      training_labels: training_labels,
      test_images: test_images, 
      test_labels: test_labels 
    } 
  }
}