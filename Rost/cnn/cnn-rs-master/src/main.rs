use clap::{arg, Command};
use image::{open, ImageBuffer, Luma};
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use walkdir::WalkDir;

use cnn_rs::*;
pub use crate::args::Args;
pub use crate::cnn::CNN;
pub use crate::convolution::*;
pub use crate::input::InputData;
pub use crate::mnist::*;
pub use crate::pooling::*;
pub use crate::softmax::*;
pub use crate::timer::*;

const BATCH_SIZE: usize = 1000;

/// Loads any of the training image, test image, training label or test label 
/// files into the corresponding fields of the given `InputData` struct
fn 
load_data (args: &Args, input_data: &mut InputData) 
{
  input_data.training_images = match args.training_image_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };

  input_data.training_labels = match args.training_labels_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };

  input_data.test_images = match args.test_image_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };

  input_data.test_labels = match args.test_labels_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };
}

/// Loads model data from disk, if a valid path to a model data file exists. Trains the model for the
/// specified number of iterations, if training images and labels are present.
fn 
load_and_train (cnn: &mut CNN, args: &Args, training_data: ImageData, labels: LabelData)
{
  let epochs = args.hyper_params.epochs;
  let mut timer = Timer::new();
  let mut indices: Vec<usize> = (0..training_data.images.len()).collect();

  // Train for `epochs` iterations over training images and labels, if they're present
  if !training_data.images.is_empty() && !labels.labels.is_empty() {
    for i in 0..epochs {
      println!(">> epoch {} / {}", i+1, epochs);
      indices.shuffle(&mut thread_rng());

      let training_data_shuffled: Vec<Array2<u8>> = indices.iter().map(|&i| training_data.images[i].clone()).collect();
      let labels_shuffled: Vec<u8> = indices.iter().map(|&i| labels.labels[i].clone()).collect();

      let mut indx = 0;
      let mut loss:f64 = 0.0;
      let mut accuracy: usize = 0;
      let mut time:u128 = 0;

      timer.start();

      for (image, label) in training_data_shuffled.iter().zip(labels_shuffled.iter()) {
        indx += 1;
        
        let (acc, ce) = cnn.train(
          &image,
          *label
        );

        loss += ce;
        accuracy += acc;

        if indx % BATCH_SIZE == 0 {
          time = timer.stop();

          println!("{} examples, avg loss {:.4} accuracy {:.1}% avg batch time {}ms", 
            indx, loss / BATCH_SIZE as f64, 100.0 * (accuracy as f64 / BATCH_SIZE as f64), time);

          // This output makes plotting data easier
          //println!("{:.4},{:.4}", loss / BATCH_SIZE as f64, (accuracy as f64 / BATCH_SIZE as f64));

          loss = 0.0;
          accuracy = 0;

          timer.start();
        }
      }
    }

    // Save results
    if args.save {
      // Any errors saving are printed within this function
      cnn.save_to_file("cnn.model").unwrap();
    }
  }
}

/// Runs prediction with the given `CNN` against the provided
/// `ImageData` and compares against given `LabelData`. Outputs prediction accracy.
fn 
load_and_test (cnn: &mut CNN, image_data: &ImageData, label_data: &LabelData) 
{
  println!("predicting digits for {} images", image_data.images.len());
  let mut count = 0;

  for (image, label) in image_data.images.iter().zip(label_data.labels.iter()) {
    let p = cnn.predict(&image);

    if p == *label as usize {
      count += 1;
    }
  }

  println!("correctly predicted {} / {} digits", count, image_data.images.len());
}

/// Given a path and an matrix of bytes, converts the byte matrix into a jpeg
/// and writes the contents to the specified path.
fn 
write_image (path: &str, image: &Array2<u8>) 
{
  let pixels: Vec<u8> = image.iter().cloned().collect();

  // Convert the Vec<u8> to an ImageBuffer
  let img = ImageBuffer::<Luma<u8>, _>::from_raw(28, 28, pixels).unwrap();

  // Save the ImageBuffer as a JPEG file
  img.save(path).unwrap(); 
  println!("saved image {}", path);
}

/// Given a directory path, loads all jpeg files as greyscale, converts them to
/// ndarray::Array2<u8>, and returns a vector of name, vector pairs
fn 
load_images (dir_path: &str) -> Result<Vec<(String, Array2<u8>)>, Box<dyn std::error::Error>> 
{
  let mut image_pairs: Vec<(String, Array2<u8>)> = Vec::new();

  // Iterate over each entry in the directory
  for entry in WalkDir::new(dir_path) {
    let entry = entry?;
    let path = entry.path();

    // Check if the entry is a file and has a .jpg extension
    if path.is_file() && path.extension().unwrap().eq(std::ffi::OsStr::new("jpeg")) {
      let img = open(path)?;

      // Convert the image to grayscale
      let gray_img = img.to_luma8();

      // Convert the grayscale image to Array2<u8>
      let a = Array2::from_shape_vec(
        (gray_img.height() as usize, 
         gray_img.width() as usize),
         gray_img.into_raw(),
        )?;

      // Add the ndarray to the images Vec
      image_pairs.push((path.file_name().unwrap().to_str().unwrap().to_string(), a));
    }
  }

  Ok(image_pairs)
}

/// Train the model on a specific file name and image; it is assumed that the file
/// is named by its digit label; ie 
///   0.jpeg
///   1.jpeg
/// etc
/// 
/// This brittle behavior is inteded for use with the --cheat flag :)
fn 
train_on_custom_images (cnn: &mut CNN, file_name: &String, image: &Array2<u8>) 
{
  let parts = file_name.split(".").collect::<Vec<&str>>();
  if parts.len() == 2 && parts[0].len() == 1 {
    let label = parts[0].parse::<u8>().unwrap();
    for _ in 0..3 {
      cnn.train(image, label);
    }
  }
}

fn 
main () 
{
  let matches = Command::new("cnn-rs")
  .version("0.1")
  .about("A convolutional neural network implementation in rust")
  .arg(arg!(--training_images <r"C:\Users\sfz-a\Videos\pictures_for_cnn">)
    .required(false)
    .value_name("path")
    .help("Path to MNIST training image file"))
  .arg(arg!(--training_labels <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST image labels file"))
  .arg(arg!(--test_images <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST test image file"))
  .arg(arg!(--test_labels <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST test labels file"))  
  .arg(arg!(--epochs <VALUE>)
    .required(false)
    .value_name("num")
    .help("Number of epochs to run over the training dataset; defaults to 3"))
  .arg(arg!(--learning_rate <VALUE>)
    .required(false)
    .value_name("float")
    .help("Learning rate; defaults to 0.01"))
  .arg(arg!(--save)
    .required(true)
    .value_name("BOOL")
    .help("if true, will save computed model weights to ./cnn.data, overwriting any existing local file"))
  .arg(arg!(--load <VALUE>)
    .required(false)
    .value_name("FILE_PATH")
    .help("path to a previously-written file containing model data"))
  .arg(arg!(--input_dir <VALUE>)
    .required(false)
    .value_name("FILE_PATH")
    .help("path to a directory containing 28x28 px jpeg files"))
  .arg(arg!(--cheat)
    .required(false)
    .value_name("BOOL")
    .help("if true, will use the contents of files under the input_dir argument to fine-tune the model before predicting outputs"))
  .get_matches();

  let training_image_path_opt = matches.get_one::<String>("training_images").cloned();
  let training_labels_path_opt = matches.get_one::<String>("training_labels").cloned();
  let test_image_path_opt = matches.get_one::<String>("test_images").cloned();
  let test_labels_path_opt = matches.get_one::<String>("test_labels").cloned();
  let epochs_opt = matches.get_one::<String>("epochs").cloned();
  let learning_rate_opt = matches.get_one::<String>("learning_rate").cloned();
  let load_opt = matches.get_one::<String>("load").cloned();
  let save_opt = matches.get_one::<bool>("save").cloned();
  let cheat_opt = matches.get_one::<bool>("cheat").cloned();
  let input_dir_opt = matches.get_one::<String>("input_dir").cloned();

  let args = Args::new( 
      training_image_path_opt,
      training_labels_path_opt,
      test_image_path_opt,
      test_labels_path_opt,
      save_opt,
      load_opt,
      input_dir_opt,
      cheat_opt,
      epochs_opt,
      learning_rate_opt
  );

  // Initializes the new CNN; if loading argument is present, reads
  // previously-serialized neural network data from path in arguments
  let mut cnn = cnn::CNN::new(args.hyper_params.clone());
  match args.load.as_ref() {
    Some(path) => cnn.load_from_file(&path).unwrap(),
    _ => ()
  }

  // Loads raw training and test images and labels, if any are specified
  let mut input_data = InputData::default();
  load_data(&args, &mut input_data);

  // If training images are present, train the model
  match (input_data.training_images, input_data.training_labels) {
    (Some(images_bytes), Some(labels_bytes)) => {
      let training_image_data = ImageData::init(&images_bytes);
      let training_label_data = LabelData::init(&labels_bytes);

      load_and_train(&mut cnn, &args, training_image_data, training_label_data);
    }
    (_, _) => ()
  }

  // If test images are present, test the model
  match (input_data.test_images, input_data.test_labels) {
    (Some(images_bytes), Some(labels_bytes)) => {
      let test_image_data = ImageData::init(&images_bytes);
      let test_label_data = LabelData::init(&labels_bytes);

      load_and_test(&mut cnn, &test_image_data, &test_label_data)
    }
    (_, _) => ()
  }

  // If a custom input directory is provided in arguments, read the image contents,
  // infer their digits, and report the predicted numbers.
  match args.input_dir {
    Some(dir) => {
      match load_images(&dir) {
        Ok(image_pairs) => {
          for (file_name, image) in image_pairs.iter() {
            if args.cheat {
              train_on_custom_images(&mut cnn, file_name, image);
            }

            let digit = cnn.predict(image);
            println!("predict that the digit in {} is {}", file_name, digit);
          }
        }
        Err(e) => {
          println!("error loading custom images: {}", e);
        }
      }
    }
    _ => ()
  }
}
