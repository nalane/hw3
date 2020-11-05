use clap::{Arg, App};

mod decision_tree;
mod dataset;

use crate::dataset::read_dataset;
use crate::decision_tree::decision_tree;

fn main() {
    let matches = App::new("hw3")
        .version("0.1.0")
        .author("Nathaniel Lane")
        .about("Homework 3 for Advanced Data Mining")
        .arg(Arg::with_name("eta")
             .short("e")
             .long("eta")
             .takes_value(true)
             .help("Minimum number of samples in leaf node (Default: 1)"))
        .arg(Arg::with_name("pi")
             .short("p")
             .long("pi")
             .takes_value(true)
             .help("Minimum purity for a leaf node (Default: 0.9)"))
        .get_matches();
    let eta = matches.value_of("eta").unwrap_or("1").parse().unwrap();
    let pi = matches.value_of("pi").unwrap_or("0.9").parse().unwrap();
    
    let dataset = read_dataset("data.xz", "labels.xz").unwrap();
    let _ = decision_tree(&dataset, eta, pi);
}
