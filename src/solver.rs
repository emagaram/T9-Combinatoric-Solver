use crate::{
    bin_ary_tree::BinAryTree, evaluate::evaluate_using_previous_solutions, freq_list::create_freq_list, globals::{NUM_LETTERS, TEST_MODE}, set32::Set32, tree_iterator::TreeIterator, util::set32s_to_string
};

use std::{cell::RefCell, collections::HashSet, io::stdin, rc::Rc};

fn vector_intersect(
    potential_children: &[Set32],
    aunt: &[Set32],
    nth_cousin_once_rm: &[Set32],
) -> Vec<Set32> {
    let aunt_set: HashSet<Set32> = aunt.iter().cloned().collect();
    let cousin_set: HashSet<Set32> = nth_cousin_once_rm.iter().cloned().collect();
    potential_children
        .iter()
        .filter(|child| aunt_set.contains(child) && cousin_set.contains(child))
        .cloned()
        .collect()
}

fn get_children_to_evaluate(
    prefix: Option<Set32>,
    potential_children: &Vec<Set32>,
    aunt: &Vec<Set32>,
    cousin: &Vec<Set32>,
) -> Vec<Set32> {
    if prefix.is_none() {
        return vector_intersect(potential_children, aunt, cousin);
    } else {
        let new_pot_children = vec![prefix.unwrap()];
        // new_pot_children.extend(potential_children);
        // TODO, binary search on potential children for location that prefix would be at and remove all previous children. Then insert prefix.
        // Or will prefix always be at the start of potential children? Yes. If prefix is EG, potential children could start at YZ BUT we don't evalutate YZ, we just evaluate.
        return new_pot_children;
    }
}

fn get_prefix(s: Set32) -> Option<Set32> {
    let ones = s.ones_indices();
    if ones.len() <= 2 {
        return None;
    }
    Some(s.remove(*ones.last().unwrap()))
}

fn get_sets_minus_one_letter(s:&Set32, ones_indices: &Vec<u32>)->Vec<Set32>{
    let mut res = Vec::new();
    for index in ones_indices {
        let add = s.clone().remove((*index).try_into().unwrap());
        res.push(add);
    }
    res
}

fn get_predecessors_to_check(path:&Vec<Set32>) -> Vec<Vec<Set32>>{
    let mut res:Vec<Vec<Set32>> = Vec::new();
    for (index, key) in path.iter().enumerate() {
        let ones_indices = key.ones_indices();
        if ones_indices.len() > 2 {
            let removed_letter_sets = get_sets_minus_one_letter(key, &ones_indices);
            for removed_letter_set in removed_letter_sets {
                let mut predecessor = path.clone();
                predecessor[index] = removed_letter_set;
                predecessor.sort_by(|a,b| TreeIterator::children_cmp(b,a));
                res.push(predecessor);
            }
        }
        else if index + 1 < path.len(){
            let mut predecessor = path.clone();
            predecessor.remove(index);
            res.push(predecessor);
        }
    }
    res

}



pub fn solution_wrapper(target_num_keys: usize) {
    let mut freq_list: Vec<(Vec<u8>, f32)> = create_freq_list("./word_freq.json");
    if TEST_MODE {
        freq_list.truncate(10000);
    }
    let mut best_scores: Vec<f32> = vec![];
    for current_num_keys in (target_num_keys..=NUM_LETTERS - 1).rev() {
        let mut iter = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
    };
        iter.set_base(2);
        let it_clone = iter.clone();
        let res = solution(
            iter,
            it_clone.clone(),
            it_clone,
            (f32::MAX, Vec::new()),
            current_num_keys,
            true,
            &freq_list,
            &best_scores,
        );
        best_scores.push(res.0);
        println!(
            "Best score for {} keys: {}, with score: {}",
            current_num_keys,
            set32s_to_string(&res.1),
            res.0
        );
        let mut temp = String::new();
        let _ = stdin().read_line(&mut temp);
    }
}

pub fn solution(
    mut iter: TreeIterator,
    aunt_iter: TreeIterator,
    nth_cousin_once_rm_iter: TreeIterator,
    mut best_solution: (f32, Vec<Set32>),
    target_num_keys: usize,
    target_above: bool,
    freq_list: &Vec<(Vec<u8>, f32)>,
    best_scores: &Vec<f32>,
) -> (f32, Vec<Set32>) {
    let prefix: Option<Set32> = match iter.path.is_empty() {
        true => None,
        false => get_prefix(iter.path[0]),
    };
    let (aunt_pot_children, cousin_once_rm_pot_children) = match iter.layer_index >= 1 {
        true => (
            aunt_iter.get_all_valid_children(),
            nth_cousin_once_rm_iter.get_all_valid_children(),
        ),
        false => (
            aunt_iter.potential_children.clone(),
            nth_cousin_once_rm_iter.potential_children.clone(),
        ),
    };

    let children_to_evaluate: Vec<Set32> = get_children_to_evaluate(
        prefix,
        &iter.potential_children,
        &aunt_pot_children,
        &cousin_once_rm_pot_children,
    );

    let mut set_first_child = false;

    // Todo: modify so that threshold no longer needed. Make Vec<Set32> an Option but rm outer option
    for child in children_to_evaluate {
        let mut child_path = iter.path.clone();
        child_path.push(child);
        //Check if the children of the node contains prefix
        let (score, score_num_keys) = evaluate_using_previous_solutions(
            &freq_list,
            &child_path,
            best_solution.0,
            target_num_keys, 
            best_scores,
        );
        if score >= best_solution.0{
            continue;
        }
        if score_num_keys <= target_num_keys {
            println!(
                "Valid solution of {}. {}",
                score,
                set32s_to_string(&child_path)
            );
            best_solution = (score, child_path);
        }
        iter.set_child_valid(child);
        if iter.layer_index == target_num_keys || score_num_keys <= target_num_keys {
            continue;
        }
        if let Some(next) = iter.create_children_block(child) {
            if !set_first_child {
                iter.set_ptr_to_next_layer_ptr(next.ptr);
                set_first_child = true;
            }
            let (sister, nth_cousin) = match iter.parent.is_some() {
                true => (
                    iter.find_sibling(child).unwrap(),
                    nth_cousin_once_rm_iter.find_child(child).unwrap(),
                ),
                false => (iter.clone(), iter.clone()),
            };
            best_solution = solution(
                next,
                sister,
                nth_cousin,
                best_solution,
                target_num_keys,
                target_above,
                freq_list,
                best_scores,
            );
        }
    }
    best_solution
}

#[cfg(test)]
pub mod tests {

    use std::cmp::Ordering;

    use crate::{create_set32_str, globals::CHAR_TO_INDEX, set32::Set32, solver::get_prefix, tree_iterator::TreeIterator, util::set32s_to_string};
    use super::get_predecessors_to_check;
    #[test]
    fn solver_get_predecessors_to_check() {
        let path = vec![create_set32_str!("ab"), create_set32_str!("cd")];
        let predecessors = get_predecessors_to_check(&path);
        assert_eq!(predecessors.len(), 1);
        assert_eq!(predecessors[0][0], create_set32_str!("cd"));
        
        // cde gyz (ab/az/bz) 
        // abz gyz (cd/ce/de)
        // abz cde (gy/gz/yz)
        let path = vec![create_set32_str!("abz"), create_set32_str!("cde"), create_set32_str!("gxy")];
        let predecessors = get_predecessors_to_check(&path);
        assert_eq!(predecessors.len(), 9);
        for predecessor in predecessors {
            println!("Predecessor: {}", set32s_to_string(&predecessor));
            for p_index in 0..predecessor.len() - 1 {
                assert_eq!(TreeIterator::children_cmp(&predecessor[p_index], &predecessor[p_index+1]), Ordering::Greater);
            }
        }
        
    }
    #[test]
    fn solver_get_prefix_none_for_one_letter() {
        assert!(get_prefix(Set32::singleton(21)).is_none());
    }

    #[test]
    fn solver_get_prefix_none_for_no_letters() {
        assert!(get_prefix(Set32::EMPTY).is_none());
    }
}
