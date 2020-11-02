// This module contains the Point struct,
// representing a point in d-dimensional space

#[derive(Debug, Clone)]
pub struct Point {
    pub coords: Vec<f64>
}

impl Point {
    pub fn new(coords: Vec<f64>) -> Point {
        Point {
            coords
        }
    }

    pub fn squared_dist(&self, rhs: &Point) -> f64 {
        self.coords.iter()
            .zip(rhs.coords.iter())
            .map(|(a, b)| (a - b).powf(2.0))
            .sum()
    }

    pub fn dist(&self, rhs: &Point) -> f64 {
        self.squared_dist(rhs).sqrt()
    }

    pub fn _length(&self) -> f64 {
        self.coords.iter().map(|a| a * a).sum::<f64>().sqrt()
    }
    
    pub fn _cos_sim(&self, rhs: &Point) -> f64 {
        let dot: f64 = self.coords.iter().zip(rhs.coords.iter()).map(|(a, b)| a * b).sum();
        dot / (self._length() * rhs._length())
    }
}
