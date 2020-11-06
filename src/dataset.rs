use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;
use std::collections::HashMap;

use lzma::reader::LzmaReader;

#[derive(Debug)]
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub label_mapping: HashMap<usize, String>
}

impl Dataset {
    // Generates a new dataset
    pub fn new(features: Vec<Vec<f64>>, labels: Vec<usize>, label_mapping: HashMap<usize, String>) -> Dataset {
        Dataset {
            features,
            labels,
            label_mapping
        }
    }

    // Generate train/test indices for k-fold cross validation
    pub fn kfold(&self, k: usize, fold_num: usize) -> (Vec<usize>, Vec<usize>) {
        let mut train = Vec::new();
        let mut test = Vec::new();

        for i in 0..self.labels.len() {
            if i % k == fold_num {
                test.push(i);
            } else {
                train.push(i);
            }
        }

        (train, test)
    }
}

// Reads in the compressed data file
fn unpack_xz(filename: &str) -> Result<BufReader<LzmaReader<File>>, Box<dyn Error>> {
    // Open the file
    let file = File::open(filename)?;
    let decompress = LzmaReader::new_decompressor(file)?;

    Ok(BufReader::new(decompress))
}

pub fn read_dataset(features_file: &str, labels_file: &str) -> Result<Dataset, Box<dyn Error>> {
    // Read in the features
    let raw_features = unpack_xz(features_file)?;
    
    // Parse the data; split on whitespace, and for every
    // line, split by commas and map each item to a Point
    let features = raw_features.lines()
        .skip(1)
        .map(|line| {
            let l = line?;
            l.split(',')
                .skip(1)
                .map(|s| s.parse::<f64>().map_err(|e| e.into()))
                .collect::<Result<_, Box<dyn Error>>>()
        })
        .collect::<Result<Vec<Vec<f64>>, Box<dyn Error>>>()?;

    // Read in the labels
    let raw_labels = unpack_xz(labels_file)?;

    // Parse the data; split on whitespace, and for every
    // line, split by commas and map each label to an integer
    let mut label_mapping = HashMap::new();
    let labels = raw_labels.lines()
        .skip(1)
        .map(|line| {
            let l = line.ok()?;
            let label = l.split(',').nth(1)?;
            if label_mapping.get(label) == None {
                let num = label_mapping.len();
                label_mapping.insert(label.to_string(), num);
            }
            let res = label_mapping.get(label)?;
            Some(*res)
        })
        .collect::<Option<Vec<usize>>>()
        .unwrap();

    // Map each numerical class to its associated string
    let label_mapping = label_mapping.iter()
        .map(|(s, &n)| (n, s.clone()))
        .collect();
    Ok(Dataset::new(features, labels, label_mapping))
}
