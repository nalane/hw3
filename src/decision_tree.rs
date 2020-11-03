use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

use crate::dataset::Dataset;

struct DecisionTree<'a> {
    dataset: &<'a>Dataset,
    active_dims: HashSet<usize>,
    active_rows: HashSet<usize>,
    left: Option<DecisionTreeLeaf<'a>>,
    right: Option<DecisionTreeLeaf<'a>>,
    class: Option<usize>
}

impl DecisionTree<'a> {
    pub fn majority_class(&self) -> usize {
        let class_count = HashMap::new();
        for row in self.active_rows {
            let class = self.dataset.labels[row];
            if class_count.get(class) == None {
                class_count.insert(class, 0);
            }
            
            if let Some(x) = class_count.get_mut(class) {
                *x = *x + 1;
            }
        }

        let (max_class, _) = class_count.iter().max_by_key(|(_, v)| v).unwrap();
        max_class
    }
    
    pub fn purity(&self) -> f64 {
        let max_class = self.majority_class();
        let num_max_class = active_rows.iter()
            .map(|idx| self.dataset.labels[idx])
            .filter(|c| c == max_class)
            .count();
        num_max_class as f64 / self.dataset.labels.len() as f64
    }
}

pub fn decision_tree(dataset: &'a Dataset, eta: usize, pi: f64) -> DecisionTree<'a> {
    let active_dims = (0..dataset.features[0].len()).collect();
    let active_rows = (0..dataset.labels.len()).collect();
    let tree = DecisionTree {
        dataset,
        active_dims,
        active_rows,
        None,
        None,
        None
    };
    decision_tree_worker(tree, eta, pi);
    tree
}

fn decision_tree_worker(node: &<'a>mut DecisionTree, eta: usize, pi: f64) {
    if node.active_rows.len() <= eta || node.purity() >= pi {
        let class = node.majority_class();
        node.class = Some(class);
    }

    let mut max_score = 0.0;
    let mut split_dim = -1;
    let mut split_val = -1.0;
    for dim in node.active_rows {
        
    }
}

fn evaluate_numeric(dataset: &Dataset, dim: usize) {
    let mut vals = dataset.features.iter()
        .zip(dataset.labels.iter())
        .map(|(row, label)| (row[dim], label))
        .collect();
    vals.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let k = dataset.labels.iter().max().unwrap();
    let mut counts = vec![0; k];
    let mut midpoints = Vec::new();
    let mut nvis = HashMap::new();
    for j in 0..vals.len() - 1 {
        counts[dataset.labels[j]] += 1;

        if vals[j + 1] != vals[j] {
            let midpoint = vals[j] + (vals[j + 1] - vals[j]) / 2;
            midpoints.push_back(midpoint);
            nvis.insert(midpoints.len() - 1, counts.clone());
        }
    }

    counts[dataset.labels[dataset.labels.len() - 1]] += 1;
    let mut best_split = -1;
    let mut best_score = 0;
    for mid_idx in 0..midpoints.len() {
        for i in 0..k {
            
        }
    }
}
