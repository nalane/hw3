use std::collections::{BTreeMap, BTreeSet};
use std::cmp::Ordering;
use rayon::prelude::*;

use crate::dataset::Dataset;

// Recursive data structure to hold a decision tree
// Contains a reference to the main dataset as well as a set of the
// rows active in that branch of the tree.
// This keeps the program from creating many copies of the dataset.
#[derive(Debug)]
pub struct DecisionTree<'a> {
    dataset: &'a Dataset,
    active_rows: BTreeSet<usize>,
    class: Option<usize>,
    split: Option<(usize, f64)>,
    y_tree: Option<Box<DecisionTree<'a>>>,
    n_tree: Option<Box<DecisionTree<'a>>>
}

impl<'a> DecisionTree<'a> {
    // Creates a new decision tree node
    pub fn new(dataset: &'a Dataset, active_rows: BTreeSet<usize>) -> DecisionTree {
        DecisionTree {
            dataset,
            active_rows,
            class: None,
            split: None,
            y_tree: None,
            n_tree: None
        }
    }
    
    // Determine the majority class of the node
    pub fn majority_class(&self) -> usize {
        // Identify how many of each class is present
        let mut class_count = BTreeMap::new();
        for &row in &self.active_rows {
            let class = self.dataset.labels[row];
            if class_count.get(&class) == None {
                class_count.insert(class, 0);
            }
            
            if let Some(x) = class_count.get_mut(&class) {
                *x += 1;
            }
        }

        // Return the entry that has the highest value
        let (max_class, _) = class_count.iter().max_by_key(|(_, &v)| v).unwrap();
        *max_class
    }

    // Calculates purity
    pub fn purity(&self) -> f64 {
        // Calculate the number of instances of the majority class
        let max_class = self.majority_class();
        let num_max_class = self.active_rows.iter()
            .map(|idx| self.dataset.labels[*idx])
            .filter(|c| *c == max_class)
            .count();
        
        num_max_class as f64 / self.active_rows.len() as f64
    }

    // Given a feature vector, what is its label?
    pub fn predict(&self, x: &[f64]) -> &str {
        // Case where tree is a leaf; emit the associated class
        if let Some(c) = self.class {
            self.dataset.label_mapping.get(&c).unwrap().as_str()
        }

        // Case where it's a split. Evaluate and recurse down the tree
        else {
            let (dim, val) = self.split.unwrap();
            if x[dim] < val {
                self.y_tree.as_ref().unwrap().predict(x)
            } else {
                self.n_tree.as_ref().unwrap().predict(x)
            }
        }
    }

    // Prints the tree
    pub fn print(&self) {
        self.print_worker(0);
    }

    // Recursive function where level is how deep into the tree we are
    fn print_worker(&self, levels: usize) {
        // Case where the tree is a leaf; just print the class
        if let Some(class) = self.class {
            for _ in 0..levels {
                print!("\t");
            }
            println!("Class: {}", self.dataset.label_mapping.get(&class).unwrap());
        }

        // Case where the tree is an internal node; recursively print the two halves
        else {
            let (split_dim, split_val) = self.split.unwrap();
            for _ in 0..levels {
                print!("\t");
            }
            println!("Split on dimension {}, vals < {}", split_dim, split_val);

            self.y_tree.as_ref().unwrap().print_worker(levels + 1);
            self.n_tree.as_ref().unwrap().print_worker(levels + 1);
        }
    }
}

// Given a dataset and some parameters, build a decision tree
pub fn decision_tree<'a>(dataset: &'a Dataset, active_rows: &[usize], eta: usize, pi: f64) -> DecisionTree<'a> {
    let active_rows = active_rows.iter().cloned().collect();
    let mut tree = DecisionTree::new(dataset, active_rows);
    decision_tree_worker(&mut tree, eta, pi);
    tree
}

// Given a decision tree node, populate that branch of the tree
fn decision_tree_worker(node: &mut DecisionTree, eta: usize, pi: f64) {
    // See if this node is a leaf
    if node.active_rows.len() <= eta || node.purity() >= pi {
        let class = node.majority_class();
        
        node.class = Some(class);
        return;
    }

    // Find the best split across all dimensions
    let (split_dim, split_val, _) = (0..node.dataset.features[0].len())
        .into_par_iter()
        .filter_map(|dim| {
            let (val, score) = evaluate_numeric(node, dim)?;
            Some((dim, val, score))
        })
        .max_by(|(_, _, x), (_, _, y)| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();

    // Figure out which items satisfy the split, and which don't
    let mut y_rows = BTreeSet::new();
    let mut n_rows = BTreeSet::new();
    for row in &node.active_rows {
        let val = node.dataset.features[*row][split_dim];
        if val < split_val {
            y_rows.insert(*row);
        }
        else {
            n_rows.insert(*row);
        }
    }

    // Create the trees for the two subsets and recurse
    let mut y_tree = DecisionTree::new(node.dataset, y_rows);
    let mut n_tree = DecisionTree::new(node.dataset, n_rows);
    decision_tree_worker(&mut y_tree, eta, pi);
    decision_tree_worker(&mut n_tree, eta, pi);

    // Update this node with the new children and the split
    node.split = Some((split_dim, split_val));
    node.y_tree = Some(Box::new(y_tree));
    node.n_tree = Some(Box::new(n_tree));
}

// Given a dimension, find the best split
fn evaluate_numeric(node: &DecisionTree, dim: usize) -> Option<(f64, f64)> {
    // Sort the entries by the given column
    let mut vals: Vec<_> = node.active_rows.iter()
        .map(|&r| (node.dataset.features[r][dim], node.dataset.labels[r]))
        .collect();
    vals.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let (features, labels): (Vec<_>, Vec<_>) = vals.iter().cloned().unzip();

    // Find all midpoints
    let k = *(labels.iter().max().unwrap()) + 1;
    let mut counts = vec![0; k];
    let mut midpoints = Vec::new();
    let mut nvis = Vec::new();
    for j in 0..labels.len() - 1 {
        counts[labels[j]] += 1;

        if (features[j + 1] - features[j]).abs() > f64::EPSILON {
            let midpoint = features[j] + (features[j + 1] - features[j]) / 2.0;
            midpoints.push(midpoint);
            nvis.push(counts.clone());
        }
    }
    counts[labels[labels.len() - 1]] += 1;

    // If no midpoint could be found, then the dimension
    // has all the same numbers and there is no split
    if midpoints.is_empty() {
        return None;
    }

    // Identify the best split
    let (best_split, best_score) = midpoints.iter()
        .enumerate()
        .map(|(mid_idx, &midpoint)| {
            // Find the incidence rate of each label in both splits
            let mut y_probs = vec![0.0; k];
            let mut n_probs = vec![0.0; k];
            for i in 0..k {
                let y_denom: i64 = nvis[mid_idx].iter().sum();
                y_probs[i] = nvis[mid_idx][i] as f64 / y_denom as f64;
            
                let n_denom: i64 = (0..k)
                    .map(|j| counts[j] - nvis[mid_idx][j])
                    .sum();
                n_probs[i] = (counts[i] - nvis[mid_idx][i]) as f64 / n_denom as f64;
            }

            // Calculate the entropy of the whole dataset
            let full_entropy: f64 = (0..k)
                .map(|class| {
                    let num_class = labels.iter()
                        .filter(|&c| *c == class)
                        .count();
                    let p = num_class as f64 / labels.len() as f64;
                    if p == 0.0 {
                        0.0
                    } else {
                        -p * p.log2()
                    }
                })
                .sum();

            // Calculate the entropy of the elements that satisfy the split
            let num_y = features.iter().take_while(|v| **v < midpoint).count();
            let y_entropy: f64 = y_probs.iter().map(|&p: &f64| {
                if p == 0.0 {
                    0.0
                } else {
                    -p * p.log2()
                }
            }).sum::<f64>() * num_y as f64 / labels.len() as f64;

            // Calculate the entropy for the elements that don't
            let num_n = labels.len() - num_y;
            let n_entropy: f64 = n_probs.iter().map(|&p: &f64| {
                if p == 0.0 {
                    0.0
                } else {
                    -p * p.log2()
                }
            }).sum::<f64>() * num_n as f64 / labels.len() as f64;

            // Calculate gain and see how it compares to the best score
            let gain = full_entropy - y_entropy - n_entropy;
            (midpoint, gain)
        })
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
        .unwrap();

    Some((best_split, best_score))
}
