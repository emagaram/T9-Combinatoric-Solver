
// use solver::solution_wrapper;

use std::time::Instant;

use freq_list::create_freq_list;
use globals::NUM_LETTERS;
use new_solver::new_solver1;
use node_iterator::NodeIterator;


pub mod bin_ary_tree;
pub mod node;
pub mod node_iterator;
pub mod evaluate;
pub mod globals;
pub mod set32;
pub mod tree_iterator;
pub mod tree_layer;
pub mod util;
pub mod solver;
pub mod new_solver;
pub mod new_evaluate;
pub mod heuristic_evaluate;
pub mod freq_list;


fn main() {
    
    let mut iter = NodeIterator::create_empty();
    let freq_list = create_freq_list("./word_freq.json");
    let start = Instant::now();
    new_solver1(&mut iter, 0, 0.28, &freq_list, 10, NUM_LETTERS, 2);
    new_solver1(&mut iter, 1, 0.28, &freq_list, 10, NUM_LETTERS, 2);
    new_solver1(&mut iter, 2, 0.28, &freq_list, 10, NUM_LETTERS, 2);
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);

}