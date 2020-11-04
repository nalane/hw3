use std::collections::{BTreeMap, BTreeSet};
use std::cmp::Ordering;

use crate::dataset::Dataset;

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
    pub fn majority_class(&self) -> usize {
        let mut class_count = BTreeMap::new();
        for row in &self.active_rows {
            let class = self.dataset.labels[*row];
            if class_count.get(&class) == None {
                class_count.insert(class, 0);
            }
            
            if let Some(x) = class_count.get_mut(&class) {
                *x = *x + 1;
            }
        }

        let (max_class, _) = class_count.iter().max_by_key(|(_, &v)| v).unwrap();
        *max_class
    }
    
    pub fn purity(&self) -> f64 {
        let max_class = self.majority_class();
        let num_max_class = self.active_rows.iter()
            .map(|idx| self.dataset.labels[*idx])
            .filter(|c| *c == max_class)
            .count();
        num_max_class as f64 / self.active_rows.len() as f64
    }
}

pub fn decision_tree<'a>(dataset: &'a Dataset, eta: usize, pi: f64) -> DecisionTree<'a> {
    let active_rows = (0..dataset.labels.len()).collect();
    let mut tree = DecisionTree {
        dataset,
        active_rows,
        class: None,
        y_tree: None,
        n_tree: None,
        split: None
    };
    decision_tree_worker(&mut tree, eta, pi);
    tree
}

fn decision_tree_worker<'a>(node: &'a mut DecisionTree, eta: usize, pi: f64) {
    if node.active_rows.len() <= eta || node.purity() >= pi {
        let class = node.majority_class();
        node.class = Some(class);
        return;
    }

    let mut max_score = 0.0;
    let mut split_dim = 0;
    let mut split_val = -1.0;
    for dim in 0..node.dataset.features[0].len() {
        let (split, score) = evaluate_numeric(node, dim);
        if score > max_score {
            split_dim = dim;
            split_val = split;
            max_score = score;
        }
    }

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

    let dataset = node.dataset;
    let mut y_tree = DecisionTree {
        dataset,
        active_rows: y_rows,
        class: None,
        y_tree: None,
        n_tree: None,
        split: None
    };
    decision_tree_worker(&mut y_tree, eta, pi);
    
    let mut n_tree = DecisionTree {
        dataset,
        active_rows: n_rows,
        class: None,
        y_tree: None,
        n_tree: None,
        split: None
    };
    decision_tree_worker(&mut n_tree, eta, pi);

    node.split = Some((split_dim, split_val));
    node.y_tree = Some(Box::new(y_tree));
    node.n_tree = Some(Box::new(n_tree));
}

fn evaluate_numeric<'a>(node: &'a DecisionTree, dim: usize) -> (f64, f64) {
    let mut vals: Vec<_> = node.active_rows.iter()
        .map(|&r| (node.dataset.features[r][dim], node.dataset.labels[r]))
        .collect();
    vals.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let (features, labels): (Vec<_>, Vec<_>) = vals.iter().cloned().unzip();

    let k = *(labels.iter().max().unwrap()) + 1;
    let mut counts = vec![0; k];
    let mut midpoints = Vec::new();
    let mut nvis = Vec::new();
    for j in 0..labels.len() - 1 {
        counts[labels[j]] += 1;

        if features[j + 1] != features[j] {
            let midpoint = features[j] + (features[j + 1] - features[j]) / 2.0;
            midpoints.push(midpoint);
            nvis.push(counts.clone());
        }
    }

    if midpoints.is_empty() {
        return (-1.0, -1.0);
    }

    counts[labels[labels.len() - 1]] += 1;
    let mut best_split = 0;
    let mut best_score = 0.0;
    for mid_idx in 0..midpoints.len() {
        let midpoint = midpoints[mid_idx];
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
        let num_y = features.iter().take_while(|v| **v < midpoint).count();
        let num_n = labels.len() - num_y;
        let y_entropy: f64 = y_probs.iter().map(|&p: &f64| {
            if p == 0.0 {
                0.0
            } else {
                -p * p.log2()
            }
        }).sum::<f64>() * num_y as f64 / labels.len() as f64;
        let n_entropy: f64 = n_probs.iter().map(|&p: &f64| {
            if p == 0.0 {
                0.0
            } else {
                -p * p.log2()
            }
        }).sum::<f64>() * num_n as f64 / labels.len() as f64;
        
        let gain = full_entropy - y_entropy - n_entropy;
        if gain > best_score {
            best_score = gain;
            best_split = mid_idx;
        }
    }

    (midpoints[best_split], best_score)
}
