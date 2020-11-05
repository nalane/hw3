use std::fs::File;
use std::io::Read;
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
    pub fn new(features: Vec<Vec<f64>>, labels: Vec<usize>, label_mapping: HashMap<usize, String>) -> Dataset {
        Dataset {
            features,
            labels,
            label_mapping
        }
    }
}

// Reads in the compressed data file
fn unpack_xz(filename: &str) -> Result<String, Box<dyn Error>> {
    // Open the file
    let file = File::open(filename)?;
    let mut decompress = LzmaReader::new_decompressor(file)?;

    // Decompress the file
    let mut buffer = String::new();
    decompress.read_to_string(&mut buffer)?;

    Ok(buffer)
}

pub fn read_dataset(features_file: &str, labels_file: &str) -> Result<Dataset, Box<dyn Error>> {
    // Read in the features
    let raw_features = unpack_xz(features_file)?;
    
    // Parse the data; split on whitespace, and for every
    // line, split by commas and map each item to a Point
    let features = raw_features.split_whitespace()
        .skip(1)
        .map(|x| {
            x.split(',')
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
    let labels = raw_labels.split_whitespace()
        .skip(1)
        .map(|x| {
            let label = x.split(',').skip(1).next()?;
            if label_mapping.get(label) == None {
                let num = label_mapping.len();
                label_mapping.insert(label.to_string(), num);
            }
            let res = label_mapping.get(label)?;
            Some(*res)
        })
        .collect::<Option<Vec<usize>>>()
        .unwrap();
    
    let label_mapping = label_mapping.iter()
        .map(|(s, &n)| (n, s.clone()))
        .collect();
    Ok(Dataset::new(features, labels, label_mapping))
}
