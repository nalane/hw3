mod dataset;

use crate::dataset::{Dataset, read_dataset};

fn main() {
    let dataset = read_dataset("data.xz", "labels.xz").unwrap();
}
