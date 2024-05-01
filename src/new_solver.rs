use std::time::Instant;

use itertools::Itertools;

use crate::{
    evaluate::{evaluate, new_evaluate}, freq_list::FreqList, heuristic_evaluate::heuristic_evaluate, node::Node,
    node_iterator::NodeIterator, util::{set32_to_string, set32s_to_string},
};

pub fn new_solver1(
    iter: &mut NodeIterator,
    target_layer: usize,
    threshold: f32,
    freq_list: &FreqList,
    num_keys: usize,
    num_letters: usize,
    max_key_len: usize,
) {
    if iter.path.len() == target_layer {
        let children_to_evaluate =
            iter.new_get_children_to_evaluate(num_keys, num_letters, max_key_len);
        if children_to_evaluate.is_none() {
            return;
        }
        for child in children_to_evaluate.unwrap() {
            let mut new_path = iter.path.clone();
            new_path.push(child);
            println!("Evaluating {}", set32s_to_string(&new_path));
            let start: Instant = Instant::now();
            let heuristic_evaluate = heuristic_evaluate(iter, &new_path);
            println!("\tHeuristic duration: {:?}", start.elapsed());
            if heuristic_evaluate.is_none() {
                println!("\tHeuristic failed.");
                continue;
            }
            let heuristic_score = heuristic_evaluate.unwrap();
            let under_threshold = heuristic_score <= threshold;
            println!(
                "\tHeuristic score {}, under threshold: {}",
                heuristic_score, under_threshold
            );
            let start: Instant = Instant::now();
            let (real_score,_) = new_evaluate(&freq_list, &new_path, threshold);
            println!("\tReal evaluate duration: {:?}", start.elapsed());
            let under_threshold = real_score <= threshold;
            println!(
                "\tReal score {}, under threshold: {}",
                real_score, under_threshold
            );
            println!("\tHeuristic percentage error: %{}", 100.0*( heuristic_score - real_score).abs() /((heuristic_score+real_score)/2.0));
            if under_threshold {
                let node = Node::new(real_score, child, vec![]);
                iter.insert_node_from_here(node);
            }
        }
    } else {
        for child in &iter.current_node.borrow().children {
            let mut new_path = iter.path.clone();
            new_path.push(child.as_ref().borrow().letters);
            new_solver1(
                &mut iter.node_to_iter(child.clone(), &new_path),
                target_layer,
                threshold,
                &freq_list,
                num_keys,
                num_letters,
                max_key_len,
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::{
        freq_list::{self, create_freq_list},
        globals::NUM_LETTERS,
        node_iterator::NodeIterator,
    };

    use super::new_solver1;

    #[test]
    fn new_solver_new_solver() {
        let mut iter = NodeIterator::create_empty();
        let freq_list = create_freq_list("./word_freq.json");
        new_solver1(&mut iter, 0, 0.1, &freq_list, 10, NUM_LETTERS, 2);
        // new_solver(&mut iter, 1, 0.1, &freq_list, 10, NUM_LETTERS, 2);
    }
}
