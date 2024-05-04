use itertools::Itertools;

use crate::{node_iterator::NodeIterator, set32::Set32};

fn get_paths_for_heuristic(path: &Vec<Set32>) -> Vec<Vec<Vec<Set32>>> {
    let mut result = vec![];
    for i in (1..path.len()).rev() {
        let combos: Vec<Vec<Set32>> = path
            .clone()
            .into_iter()
            .combinations(path.len() - i)
            .collect();
        result.push(combos);
    }
    result
}

pub fn heuristic_evaluate(iter: &NodeIterator, path: &Vec<Set32>) -> Option<f32> {
    let paths = get_paths_for_heuristic(path);
    let mut is_add = true;
    let mut score = 0.0;
    for path_group in paths {
        for path in path_group {
            let node = iter.find_node_from_root(&path);
            if node.is_none() {
                return None;
            }
            let add = node.unwrap().read().score;
            score += if is_add { add } else { -add };
        }
        is_add = !is_add;
    }

    Some(score)
}

#[cfg(test)]
mod tests {
    use crate::{create_set32s_vec, globals::CHAR_TO_INDEX, set32::Set32};

    use super::get_paths_for_heuristic;

    #[test]
    fn heuristic_evaluate_get_paths_for_heuristic(){
        let paths = get_paths_for_heuristic(&create_set32s_vec!("ab,cd,ef"));
        assert_eq!(paths.len(),2);
        assert_eq!(paths[0].len(),3);
        assert_eq!(paths[1].len(),3);
    }
}
