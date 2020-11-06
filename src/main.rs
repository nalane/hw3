use std::collections::BTreeMap;
use clap::{Arg, App};

mod dataset;
mod decision_tree;
mod knn;

use crate::dataset::read_dataset;
use crate::decision_tree::decision_tree;
use crate::knn::k_nearest_neighbors;

fn f_score<'a>(actual: &Vec<&'a str>, predicted: &Vec<&'a str>) -> BTreeMap<&'a str, f64> {
    let mut true_pos: BTreeMap<&str, usize> = actual.iter().map(|&a| (a, 0)).collect();
    let mut false_pos: BTreeMap<&str, usize> = actual.iter().map(|&a| (a, 0)).collect();
    let mut false_neg: BTreeMap<&str, usize> = actual.iter().map(|&a| (a, 0)).collect();

    for i in 0..actual.len() {
        if actual[i] == predicted[i] {
            *true_pos.get_mut(actual[i]).unwrap() += 1;
        }

        else {
            *false_pos.get_mut(predicted[i]).unwrap() += 1;
            *false_neg.get_mut(actual[i]).unwrap() += 1;
        }
    }

    let mut f_scores = BTreeMap::new();
    for class in true_pos.keys() {
        let t_p = *true_pos.get(class).unwrap();
        let f_p = *false_pos.get(class).unwrap();
        let f_n = *false_neg.get(class).unwrap();

        let recall = t_p as f64 / (t_p + f_p) as f64;
        let prec = t_p as f64 / (t_p + f_n) as f64;
        let mut f_score = 2.0 * prec * recall / (recall + prec);
        if f_score.is_nan() {
            f_score = 0.0;
        }
        
        f_scores.insert(*class, f_score);
    }

    f_scores
}

fn main() {
    let matches = App::new("hw3")
        .version("0.1.0")
        .author("Nathaniel Lane")
        .about("Homework 3 for Advanced Data Mining")
        .arg(Arg::with_name("eta")
             .short("e")
             .long("eta")
             .takes_value(true)
             .help("Minimum number of samples in leaf node (Default: 15)"))
        .arg(Arg::with_name("pi")
             .short("p")
             .long("pi")
             .takes_value(true)
             .help("Minimum purity for a leaf node (Default: 1.0)"))
        .arg(Arg::with_name("knn")
             .short("k")
             .long("knn")
             .takes_value(true)
             .help("k for k nearest neighbors (Default: 1)"))
        .arg(Arg::with_name("kfold")
             .short("f")
             .long("kfold")
             .takes_value(true)
             .help("k for k fold (Default: 5)"))
        .arg(Arg::with_name("print_trees")
             .short("t")
             .long("print-trees")
             .help("Print all of the decision trees after they've been made (Default: false)"))
        .get_matches();
    let eta = matches.value_of("eta").unwrap_or("15").parse().unwrap();
    let pi = matches.value_of("pi").unwrap_or("1").parse().unwrap();
    let knn = matches.value_of("knn").unwrap_or("1").parse().unwrap();
    let kfold = matches.value_of("kfold").unwrap_or("5").parse().unwrap();
    let print_trees = matches.is_present("print_trees");
    
    let dataset = read_dataset("data.xz", "labels.xz").unwrap();
    let mut tree_macro_fs: Vec<f64> = Vec::new();
    let mut knn_macro_fs: Vec<f64> = Vec::new();
    for i in 0..kfold {
        let (train, test) = dataset.kfold(kfold, i);

        println!("Creating decision tree for fold {}...", i);
        let tree = decision_tree(&dataset, &train, eta, pi);
        if print_trees {
            tree.print();
        }

        let actual = test.iter()
            .map(|&idx| dataset.label_mapping.get(&dataset.labels[idx]).unwrap().as_str())
            .collect();
        let (tree_preds, knn_preds): (Vec<_>, Vec<_>) = test.iter()
            .map(|&t_idx| &dataset.features[t_idx])
            .map(|f| (tree.predict(f), k_nearest_neighbors(&dataset, f, knn).unwrap()))
            .unzip();
        
        let tree_fs = f_score(&actual, &tree_preds);
        println!("Decision tree f-scores per class for fold {}", i);
        for (class, f_score) in tree_fs.iter() {
            println!("\t{}: {}", class, f_score);
        }
        let tree_macro = tree_fs.values().sum::<f64>() / tree_fs.len() as f64;
        tree_macro_fs.push(tree_macro);
        println!("\tAverage: {}", tree_macro);
        
        let knn_fs = f_score(&actual, &knn_preds);
        println!("K nearest neighbors f-scores per class for fold {}", i);
        for (class, f_score) in knn_fs.iter() {
            println!("\t{}: {}", class, f_score);
        }
        let knn_macro = knn_fs.values().sum::<f64>() / knn_fs.len() as f64;
        knn_macro_fs.push(knn_macro);
        println!("\tAverage: {}", knn_macro);
        println!();
    }

    let tree_macro: f64 = tree_macro_fs.iter().sum::<f64>() / tree_macro_fs.len() as f64;
    let knn_macro: f64 = knn_macro_fs.iter().sum::<f64>() / knn_macro_fs.len() as f64;
    println!("Decision Tree macro F1 score: {}", tree_macro);
    println!("K nearest neighbors macro F1 score: {}", knn_macro);
}
