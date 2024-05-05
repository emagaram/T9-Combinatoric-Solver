use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use itertools::Itertools;

use crate::{node_iterator::NodeIterator, set32::Set32};

/*
Plan:
Iterate through tree layer by layer. At each node, calculate its score + intersections with children above it.
Check if score < score for that node. If so, set it


*/

pub struct NewScorecast {
    root: Arc<RwLock<NewScorecastNode>>,
    best_sums: Arc<RwLock<HashMap<(usize, usize), Option<f32>>>>,
}

impl NewScorecast {
    fn sum_to_n_k_terms_helper(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        if k == 0 && n == 0 {
            results.push(current.clone());
            return;
        }
        if k == 0 || n <= 0 {
            return;
        }
        for i in start..=n {
            current.push(i);
            Self::sum_to_n_k_terms_helper(n - i, k - 1, i, current, results);
            current.pop();
        }
    }
    fn get_possible_key_lengths(n: usize, k: usize, max: usize) -> Vec<usize> {
        let mut sums: Vec<Vec<usize>> = vec![];
        let mut current: Vec<usize> = vec![];
        Self::sum_to_n_k_terms_helper(n, k, 1, &mut current, &mut sums);
        sums.iter()
            .map(|sum| {
                let mut clone = sum.clone();
                clone.reverse();
                clone
            })
            .map(|sum| sum[0])
            .sorted()
            .filter(|value| *value > 1)
            .filter(|value| *value <= max)
            .collect()
    }
    pub fn get_scorecast_add(&self, path: &[Set32]) {
        let mut num_letters_path = Vec::new();
        for key in path {
            num_letters_path.push(key.ones_indices().len());
        }
    }
    fn calculate_scorecast_score(parent: &NodeIterator, child: &NodeIterator) -> OrderedFloat<f32> {
        /*
        INT(ABC) + INT(BC) + INT(AC) = (ABC - AB - AC - BC + A + B + C) + (BC - B - C) + (AC-A-C) = ABC - AB - C
        Scorecast(ABCD) = D + (ABCD - ABC - D) = ABCD - ABC
        Scorecast(ABC) = C + (ABC - AB - C) => ABC - AB
        Scorecast(AB) = B + (AB - A - B) => AB - A
        Scorecast(A) = A
        Scorecast of ABCD: S(D) +  INT(ABCD) + INT(ABD) + INT(ACD) + INT(BCD) + INT(AD) + INT(BD) + INT(CD)
        =>
        D + (ABCD - ABC - ABD - ACD - BCD + AB + AC + AD + BC + BD + CD - A - B - C - D)
        + (ABD-AB-AD-BD+A+B+D)
        + (ACD-AC-AD-CD+A+C+D)
        + (BCD-BC-BD-CD+B+C+D)
        + (AD-A-D)
        + (BD-B-D)
        + (CD-C-D)
        =
        ABCD - ABC
         */
        let node_score_child = child.current_node.read().score;
        let node_score_parent = parent.current_node.read().score;
        return OrderedFloat(node_score_child - node_score_parent);
    }

    pub fn get_add_amount(&self, target_num_keys: usize, target_num_letters: usize) -> Option<f32> {
        None
    }

    pub fn setup_scorecast_tree(&mut self, current_iter: &NodeIterator) {
        self.root = Arc::new(RwLock::new(NewScorecastNode {
            num_letters: 0,
            scorecast_scores: BTreeSet::new(),
            children: vec![],
        }));
        self.best_sums.write().clear();
        self.setup_scorecast_tree_helper(&[], current_iter)
    }
    fn setup_scorecast_tree_helper(&mut self, key_len_path: &[usize], current_iter: &NodeIterator) {
        for child in &current_iter.current_node.read().children {
            let full_letter_path = &mut current_iter.path.clone();
            full_letter_path.push(child.read().letters);
            // println!("Full letter path: {:?}", set32s_to_string(full_letter_path));
            let mut full_key_len_path = key_len_path.to_vec();
            let child_key_len = child.read().letters.ones_indices().len();
            full_key_len_path.push(child_key_len);
            let child_iter = current_iter
                .find_node_iter_from_root(&full_letter_path)
                .expect("Child iter to exist");
            let scorecast_score = Self::calculate_scorecast_score(&current_iter, &child_iter);

            let our_child = self.find_node(&full_key_len_path);
            if let Some(our_child) = our_child {
                our_child
                    .write()
                    .scorecast_scores
                    .insert(scorecast_score);
            } else {
                let mut scorecast_scores = BTreeSet::new();
                scorecast_scores.insert(scorecast_score);
                self.insert_node_from_root(
                    &key_len_path,
                    NewScorecastNode {
                        num_letters: child_key_len,
                        children: vec![],
                        scorecast_scores,
                    },
                )
            }
            Self::setup_scorecast_tree_helper(self, &full_key_len_path, &child_iter);
        }
    }

    fn find_node(&self, path: &[usize]) -> Option<Arc<RwLock<NewScorecastNode>>> {
        if path.is_empty() {
            return Some(self.root.clone());
        }

        let mut current: Arc<RwLock<NewScorecastNode>> = self.root.clone();
        for num_letters in path {
            let search = &current
                .read()
                .children
                .binary_search_by(|node| node.read().num_letters.cmp(num_letters));
            match search {
                Ok(index) => {
                    let clone = current.read().children[*index].clone();
                    current = clone;
                }
                Err(_) => return None, // Return None if any key in the path does not match.
            }
        }
        Some(current.clone())
    }
    fn insert_node_from_root(&mut self, path: &[usize], node: NewScorecastNode) {
        let found_node = self.find_node(path);
        if found_node.is_none() {
            panic!("Node DNE on insertion");
        }
        let found_node = found_node.unwrap();
        // Now insert the final node at the current location
        found_node
            .write()
            .children
            .push(Arc::new(RwLock::new(node)));
    }
}

impl Default for NewScorecast {
    fn default() -> Self {
        Self {
            root: Arc::new(RwLock::new(NewScorecastNode {
                num_letters: 0,
                scorecast_scores: BTreeSet::new(),
                children: vec![],
            })),
            best_sums: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

pub struct NewScorecastNode {
    num_letters: usize,
    scorecast_scores: BTreeSet<OrderedFloat<f32>>,
    children: Vec<Arc<RwLock<NewScorecastNode>>>,
}

impl NewScorecastNode {
    pub fn new(
        num_letters: usize,
        scorecast_scores: BTreeSet<OrderedFloat<f32>>,
        children: Vec<Arc<RwLock<NewScorecastNode>>>,
    ) -> NewScorecastNode {
        NewScorecastNode {
            num_letters,
            scorecast_scores,
            children,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        sync::Arc,
    };

    use ordered_float::OrderedFloat;
    use parking_lot::RwLock;

    use crate::{
        create_btreeset, create_set32_str,
        globals::{CHAR_TO_INDEX, NUM_LETTERS},
        node::Node,
        node_iterator::NodeIterator,
        set32::Set32,
    };

    use super::{NewScorecast, NewScorecastNode};
    fn create_empty_node_iterator() -> NodeIterator {
        let root = Arc::new(RwLock::new(Node::new(0.0, Set32::EMPTY, vec![])));
        NodeIterator {
            current_node: root.clone(),
            path: vec![],
            root,
        }
    }
    fn get_simple_scorecast() -> NewScorecast {
        let mut scorecast = NewScorecast::default();

        scorecast.insert_node_from_root(&[], NewScorecastNode::new(2, create_btreeset![OrderedFloat(12.12)], vec![]));
        scorecast.insert_node_from_root(&[], NewScorecastNode::new(4, create_btreeset![OrderedFloat(2.9)], vec![]));
        scorecast.insert_node_from_root(&[], NewScorecastNode::new(5, create_btreeset![OrderedFloat(2.9)],  vec![]));
        scorecast
    }

    fn get_complex_scorecast() -> NewScorecast {
        let mut scorecast = NewScorecast::default();
        scorecast.insert_node_from_root(
            &[],
            NewScorecastNode::new(2, create_btreeset![OrderedFloat(0.112)], vec![]),
        );
        let c4_1 = Arc::new(RwLock::new(NewScorecastNode::new(3, create_btreeset![OrderedFloat(1.2)], vec![])));
        scorecast.insert_node_from_root(&[], NewScorecastNode::new(4, create_btreeset![OrderedFloat(2.9)], vec![c4_1]));

        scorecast.insert_node_from_root(&[], NewScorecastNode::new(5, create_btreeset![OrderedFloat(1.5)], vec![]));
        scorecast
    }

    #[test]
    fn empty_score_cast() {
        let scorecast = NewScorecast::default();
        let add = scorecast.get_add_amount(10, 11);
        assert!(add.is_none());
    }
    #[test]
    fn scorecast_set_child_sums_complex() {
        let mut iter = create_empty_node_iterator();
        let ab = Node::new(0.1, create_set32_str!("ab"), vec![]);
        let bc = Node::new(0.2, create_set32_str!("bc"), vec![]);
        let df = Node::new(0.23, create_set32_str!("df"), vec![]);
        let xyz = Node::new(0.25, create_set32_str!("xyz"), vec![]);
        let xyz_ab = Node::new(0.4, create_set32_str!("ab"), vec![]);
        let xyz_bc = Node::new(0.55, create_set32_str!("bc"), vec![]);
        let xyz_bc_df = Node::new(0.78, create_set32_str!("df"), vec![]);
        iter.insert_node_from_root(&[], df.clone());
        iter.insert_node_from_root(&[], bc.clone());
        iter.insert_node_from_root(&[], ab.clone());
        iter.insert_node_from_root(&[], xyz.clone());
        iter.insert_node_from_root(&[create_set32_str!("xyz")], xyz_bc.clone());
        iter.insert_node_from_root(&[create_set32_str!("xyz")], xyz_ab.clone());

        let mut scorecast = NewScorecast::default();
        scorecast.setup_scorecast_tree(&iter);
        /*
           7k, 7l: 0, all 1 keys
           3k, 7l: Scorecast(xyz) + scorecast(xyz ab) + scorecast(ab) = 0.25 + (0.4-0.25) + 0.1 = 0.5
        */
        let add = scorecast.get_add_amount(3, 7);
        assert!(add.is_some());
        assert_eq!(add.unwrap(), 0.5);
        iter.insert_node_from_root(
            &[create_set32_str!("xyz"), create_set32_str!("bc")],
            xyz_bc_df.clone(),
        );
        scorecast.setup_scorecast_tree(&iter);
        // 3 keys, 7 letters: xyz + xyz_ab + xyz_bc_df = Scorecast(xyz) + Scorecast(xyz ab) + Scorecast(xyz bc df) = 0.25 + (0.4-0.25) + 0.78-0.55 = 0.63
        let add = scorecast.get_add_amount(3, 7);
        assert!(add.is_some());
        assert_eq!(add.unwrap(), 0.63);
    }

    #[test]
    fn scorecast_setup_tree() {
        let mut iter = create_empty_node_iterator();
        let ab = Node::new(0.1, create_set32_str!("ab"), vec![]);
        let xyz_ab = Node::new(0.4, create_set32_str!("ab"), vec![]);
        let xyz = Node::new(0.2, create_set32_str!("xyz"), vec![]);
        iter.insert_node_from_root(&[], ab);
        iter.insert_node_from_root(&[], xyz);
        iter.insert_node_from_root(&[create_set32_str!("xyz")], xyz_ab);
        let mut scorecast = NewScorecast::default();
        scorecast.setup_scorecast_tree(&iter);
        let ab_score = scorecast.find_node(&[2]);
        assert!(ab_score.is_some());
        // assert!((ab_score.unwrap().read().scorecast_scores - 0.1).abs() < f32::EPSILON);
        // let xyz_ab_score = scorecast.find_node(&[3, 2]);
        // assert!(xyz_ab_score.is_some());
        // // S(ab) + INT(xyz ab) = 0.1 + (0.4-0.2-0.1) = 0.2
        // println!(
        //     "{} score",
        //     xyz_ab_score
        //         .clone()
        //         .unwrap()
        //         .read()
        //         .unwrap()
        //         .scorecast_scores
        // );
        // assert!(
        //     (xyz_ab_score.unwrap().read().scorecast_scores - 0.2).abs() < f32::EPSILON
        // );
    }

    #[test]
    fn scorecast_sum_to_n_k_terms() {
        let n = 27;
        let k = 10;
        let mut results = Vec::new();
        let mut current = Vec::new();

        NewScorecast::sum_to_n_k_terms_helper(n, k, 1, &mut current, &mut results);
        let len = results.len();
        for result in results {
            println!("{:?}", result.into_iter().rev().collect::<Vec<usize>>());
        }
        println!("{}", len);
    }

    #[test]
    fn scorecast_get_possible_key_lengths() {
        // 3 2 1 and 2 2 2
        let key_lengths = NewScorecast::get_possible_key_lengths(6, 3, 3);
        println!("{:?}", key_lengths);
        assert_eq!(key_lengths.len(), 2);
    }

    #[test]
    fn scorecast_insert_find_node() {
        let mut scorecast = get_simple_scorecast();
        let find = scorecast.find_node(&[2]);
        assert!(find.is_some());
        assert!(find.unwrap().read().scorecast_scores.contains(&OrderedFloat(12.12)));
        let find = scorecast.find_node(&[4]);
        assert!(find.is_some());
        assert!(find.unwrap().read().scorecast_scores.contains(&OrderedFloat(2.9)));
        scorecast.insert_node_from_root(
            &[2],
            NewScorecastNode {
                children: vec![],
                scorecast_scores:  create_btreeset!(OrderedFloat(1000.0)),
                num_letters: 2,
            },
        );
        let find = scorecast.find_node(&[2, 2]);
        assert!(find.is_some());
        assert!(find.unwrap().read().scorecast_scores.contains(&OrderedFloat(1000.0)));
    }
}
