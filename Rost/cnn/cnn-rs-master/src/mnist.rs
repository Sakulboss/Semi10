use byteorder::{BigEndian, ByteOrder};
use ndarray::Array2;

/// Data and utilities for working with the MNIST dataset of handwriten digits;
/// see http://yann.lecun.com/exdb/mnist/ for details of the dataset and 
/// its formatting.
#[derive(Debug, Clone)]
pub struct ImageData {
  pub num_images: usize,
  pub num_rows: usize,
  pub num_cols: usize,
  pub images: Vec<Array2<u8>>
}

#[derive(Debug, Clone)]
pub struct LabelData {
  pub labels: Vec<u8>
}

fn 
extract_images_header (data: &Vec<u8>) -> (usize, usize, usize)
{
  assert!(data.len() >= 16);

  let magic_number = BigEndian::read_u32(&data[0..4]);
  let number_of_images = BigEndian::read_u32(&data[4..8]);
  let number_of_rows = BigEndian::read_u32(&data[8..12]);
  let number_of_columns = BigEndian::read_u32(&data[12..16]);

  assert!(magic_number == 2051);

  (number_of_images as usize, number_of_rows as usize, number_of_columns as usize)
}

fn 
extract_images (data: &Vec<u8>, number_of_images: usize, rows: usize, cols: usize) -> Vec<Array2<u8>> 
{
  let mut images = Vec::new();
  let mut start = 16; // Skip the header bytes

  for _ in 0..number_of_images {
      let end = start + rows * cols;
      let image_data = &data[start..end];
      let image = Array2::from_shape_vec((rows, cols), image_data.to_vec()).unwrap();

      images.push(image);
      start = end;
  }

  images
}

fn 
extract_labels_header (data: &Vec<u8>) -> usize
{
  assert!(data.len() >= 8);

  let magic_number = BigEndian::read_u32(&data[0..4]);
  let number_of_labels = BigEndian::read_u32(&data[4..8]);

  assert!(magic_number == 2049);

  number_of_labels as usize
}

fn 
extract_labels (data: &Vec<u8>, number_of_labels: usize) -> Vec<u8> 
{
  let labels = data[8..].to_vec();

  assert!(labels.len() == number_of_labels);

  labels
}

impl ImageData {
  pub fn
  init (bytes: &Vec<u8>) -> Self 
  {
    let header = extract_images_header(bytes);
    let images = extract_images(bytes, header.0, header.1, header.2);

    ImageData {
      num_images: header.0,
      num_rows: header.1,
      num_cols: header.2,
      images: images
    }
  }
}

impl LabelData {
  pub fn
  init (bytes: &Vec<u8>) -> Self 
  {
    let num_labels = extract_labels_header(bytes);
    let labels = extract_labels(bytes, num_labels);

    LabelData {
      labels: labels
    }
  }
}