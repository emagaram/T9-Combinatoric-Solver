

use solver::solver;


pub mod node;
pub mod node_iterator;
pub mod evaluate;
pub mod globals;
pub mod set32;
pub mod util;
pub mod solver;
pub mod new_evaluate;
pub mod heuristic_evaluate;
pub mod scorecast;
pub mod freq_list;


fn main() {
    // rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    solver(0.0248, 10, 3);
}