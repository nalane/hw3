mod decision_tree;
mod dataset;

use crate::dataset::read_dataset;
use crate::decision_tree::decision_tree;

fn main() {
    let dataset = read_dataset("data.xz", "labels.xz").unwrap();
    let tree = decision_tree(&dataset, 1, 0.9);
    println!("done");
}
