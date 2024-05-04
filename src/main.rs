

use new_solver::solver;



pub mod bin_ary_tree;
pub mod node;
pub mod node_iterator;
pub mod evaluate;
pub mod globals;
pub mod set32;
pub mod tree_iterator;
pub mod tree_layer;
pub mod util;
pub mod new_solver;
pub mod new_evaluate;
pub mod heuristic_evaluate;
pub mod scorecast;
pub mod scorecast_new;
pub mod freq_list;


fn main() {
    solver(0.0248, 10, 3);
}