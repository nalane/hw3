use std::cmp::Ordering;
use std::collections::HashMap;

use crate::dataset::Dataset;

pub fn k_nearest_neighbors<'a>(dataset: &'a Dataset, x: &Vec<f64>, k: usize) -> Option<&'a str> {
    let mut items: Vec<_> = dataset.features.iter()
        .map(|y| x.iter()
             .zip(y.iter())
             .map(|(a, b)| (a - b).powi(2))
             .sum::<f64>())
        .enumerate()
        .collect();

    items.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap_or(Ordering::Greater));
    let labels: Vec<_> = items.iter()
        .take(k)
        .map(|(idx, _)| dataset.label_mapping.get(&dataset.labels[*idx]).unwrap().as_str())
        .collect();

    let mut counts = HashMap::new();
    for label in labels {
        if let Some(c) = counts.get_mut(label) {
            *c += 1;
        }
        else {
            counts.insert(label, 1);
        }
    }
    
    let (label, _) = counts.iter().max_by_key(|(_, &c)| c)?;
    Some(label)
}
