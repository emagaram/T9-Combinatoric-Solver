use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use parking_lot::RwLock;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/*
New plan: 
    - Find minimum scoring key config that uses less letters and just use that as a scorecast. It should only take a few seconds.
        - Can also look at 18,1 and 17,2/16,3 and 16,2/15,3 to see if you can increase the number of letters safely.
    - Create a list of all word pairs and iterate through them, adding their frequency to the node. Also keep a separate hashmap 
      and count of seen words so that you don't repeat them more than 4x. E.g. for at + so, it + so, to+so, as+so astoi => "so" could be moved back very far
        - Use the same tree for storing these scores as your actual scoring tree. Store an "intersect" score and a "full" score.
     


*/


use crate::{
    evaluate::new_evaluate,
    freq_list::{create_freq_list, FreqList},
    globals::NUM_LETTERS,
    heuristic_evaluate::heuristic_evaluate,
    node::Node,
    node_iterator::NodeIterator,
    scorecast::Scorecast,
    set32::Set32,
    util::set32s_to_string,
};

pub fn solver(threshold: f32, desired_num_keys: usize, max_key_len: usize) {
    let iter = NodeIterator::create_empty();
    let freq_list = create_freq_list("./word_freq.json");
    let scorecast = Arc::new(RwLock::new(Scorecast::default()));
    for i in 0..desired_num_keys {
        println!("Starting layer {}", i);
        let start = Instant::now();
        solver_layer(
            iter.clone(),
            i,
            scorecast.clone(),
            threshold,
            &freq_list,
            desired_num_keys,
            NUM_LETTERS,
            max_key_len,
        );
        println!(
            "Finished layer {} in {:?}, creating scorecast score",
            i,
            start.elapsed()
        );
        let start = Instant::now();
        scorecast.write().setup_scorecast_tree(&iter);
        println!("Scorecast tree created in {:?}", start.elapsed());
        // if i == 0 {
        //     panic!("Done");
        // }
    }
}

fn solver_layer_evaluation_logic(
    child: &Set32,
    iter: NodeIterator,
    target_layer: usize,
    scorecast: Arc<RwLock<Scorecast>>,
    threshold: f32,
    freq_list: &FreqList,
    desired_num_keys: usize,
    desired_num_letters: usize,
    max_key_len: usize,
) -> Option<Node> {
    let mut new_path = iter.path.clone();
    new_path.push(*child);
    // println!("Evaluating {}", set32s_to_string(&new_path));
    let add = Some(0.01 - ((1.0 + target_layer as f32) *0.001));
    let mut heuristic_score = 0.0;
    let mut rng = rand::thread_rng();
    let random_number: u32 = rng.gen();
    // println!("Random number: {}", random_number);
    let should_print = random_number < 4294960; // about .01%
    if should_print {
        let mut fp = iter.path.clone();
        fp.push(*child);
        println!("Evaluating {}", set32s_to_string(&fp));
    }
    if true {
        let start: Instant = Instant::now();
        let heuristic_evaluate = heuristic_evaluate(&iter, &new_path);
        if should_print {
            println!("\tHeuristic duration: {:?}", start.elapsed());
        }
        
        if heuristic_evaluate.is_none() {
            println!("\tHeuristic failed.");
            return None;
        }
        heuristic_score = heuristic_evaluate.unwrap();
        let mut under_threshold: bool = heuristic_score <= threshold;
        let num_letters_used: usize = new_path.iter().map(|key| key.ones_indices().len()).sum();
        let start: Instant = Instant::now();
        // add = scorecast.read().get_add_amount(
        //     desired_num_keys - target_layer,
        //     desired_num_letters - num_letters_used,
        // );
        if should_print {
            println!("\tGetting add_amount took {:?}", start.elapsed());
        }
        
        if add.is_none() {
            if should_print {
                println!(
                    "\tScorecasting predicted failure for {}. This probably shouldn't happen",
                    set32s_to_string(&new_path)
                );
            }

        } else {
            if should_print {
                println!(
                    "\tHeuristic w/o scorecasting: {}, w/ scorecasting {}",
                    heuristic_score,
                    heuristic_score + add.unwrap()
                );
            }
            under_threshold = heuristic_score + add.unwrap() <= threshold;
        }
        if should_print {
            println!("\tHeuristic score under threshold: {}", under_threshold);
        }
        
        if !under_threshold {
            if should_print {
                println!("\tPruning!");
            }
            
            // should_continue = false;
            return None;
        }
    }

    let start: Instant = Instant::now();
    let (real_score, num_keys) = new_evaluate(&freq_list, &new_path, threshold- add.unwrap());
    if should_print {
        println!("\tReal evaluate duration: {:?}", start.elapsed());
    }
    if num_keys == max_key_len && real_score <= threshold {
        println!("\tSolution found: {}", set32s_to_string(&iter.path));
        panic!(
            "\tReal score: {}",
            new_evaluate(&freq_list, &new_path, threshold).0
        );
    }

    let real_score_with_scorecast = real_score + add.unwrap_or(0.0);
    let under_threshold = real_score_with_scorecast <= threshold;
    if should_print {
        println!(
            "\tReal score w/o scorecast: {}, with scorecast {}, under threshold: {}",
            real_score, real_score_with_scorecast, under_threshold
        );
        println!(
            "\tHeuristic percentage error: %{}",
            100.0 * (heuristic_score - real_score).abs() / ((heuristic_score + real_score) / 2.0)
        );
    }


    if under_threshold {
        let node = Node::new(real_score, *child, vec![]);
        return Some(node);
        // iter.insert_node_from_here(node);
        // nodes_to_insert.push(node);
    } else {
        if should_print {
            println!("\tPruning!");
        }
        
        return None;
    }
}

pub fn solver_layer(
    iter: NodeIterator,
    target_layer: usize,
    scorecast: Arc<RwLock<Scorecast>>,
    threshold: f32,
    freq_list: &FreqList,
    desired_num_keys: usize,
    desired_num_letters: usize,
    max_key_len: usize,
) {
    if iter.path.len() == target_layer {
        let children_to_evaluate =
            iter.new_get_children_to_evaluate(desired_num_keys, desired_num_letters, max_key_len);
        if children_to_evaluate.is_none() {
            return;
        }
        let nodes_to_insert = Mutex::new(Vec::new());
        if target_layer == 0 {
            children_to_evaluate
                .unwrap()
                .par_iter()
                .for_each(|child| {
                    let node = solver_layer_evaluation_logic(
                        child,
                        iter.clone(),
                        target_layer,
                        scorecast.clone(),
                        threshold,
                        freq_list,
                        desired_num_keys,
                        desired_num_letters,
                        max_key_len,
                    );
                    if node.is_some() {
                        nodes_to_insert.lock().unwrap().push(node.unwrap());
                    }
                });
        } else {
            children_to_evaluate
                .unwrap()
                .iter()
                .for_each(|child| {
                    let node = solver_layer_evaluation_logic(
                        child,
                        iter.clone(),
                        target_layer,
                        scorecast.clone(),
                        threshold,
                        freq_list,
                        desired_num_keys,
                        desired_num_letters,
                        max_key_len,
                    );
                    if node.is_some() {
                        nodes_to_insert.lock().unwrap().push(node.unwrap());
                    }
                });
        }
        for node in nodes_to_insert.lock().unwrap().iter() {
            iter.insert_node_from_here(node.clone());
        }
    } else {
        if iter.path.len() + 1 == target_layer {
            iter.current_node
                .read()
                .children
                .par_iter()
                .for_each(|child| {
                    let mut new_path = iter.path.clone();
                    new_path.push(child.read().letters);
                    solver_layer(
                        iter.node_to_iter(child.clone(), &new_path),
                        target_layer,
                        scorecast.clone(),
                        threshold,
                        &freq_list,
                        desired_num_keys,
                        desired_num_letters,
                        max_key_len,
                    );
                });
        } else {
            iter.current_node
                .read()
                .children
                .iter()
                .for_each(|child| {
                    let mut new_path = iter.path.clone();
                    new_path.push(child.read().letters);
                    solver_layer(
                        iter.node_to_iter(child.clone(), &new_path),
                        target_layer,
                        scorecast.clone(),
                        threshold,
                        &freq_list,
                        desired_num_keys,
                        desired_num_letters,
                        max_key_len,
                    );
                });
        }
    }
}
#[cfg(test)]
mod tests {

    #[test]
    fn new_solver_new_solver() {
        // let mut iter = NodeIterator::create_empty();
        // let freq_list = create_freq_list("./word_freq.json");
        // new_solver1(&mut iter, 0, 0.1, &freq_list, 10, NUM_LETTERS, 2);
        // new_solver(&mut iter, 1, 0.1, &freq_list, 10, NUM_LETTERS, 2);
    }
}
