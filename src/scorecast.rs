use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;
use parking_lot::RwLock;

use crate::{
    node_iterator::NodeIterator,
    set32::Set32,
};

/*
 Plan:
 Iterate through tree layer by layer. At ech node, calculate its score + intersections with children above it.
 Check if score < score for that node. If so, set it

 Loop through node children starting from leaf, set best score = min_subtrees + current score




 intersection: S(ABC) - S(AB) - S(AC) - S(BC) + S(A) + S(B) + S(C)

 Heuristic: S(ABC) approx=  S(AB) + S(AC) + S(BC) - S(A) - S(B)- S(C)

Take S(ABC)-Heuristic to get intersection(ABC

Int ABC = ABC - AB - AC - BC + A + B + C
Int ABC + AB + AC + BC = ABC - AB - AC - BC + A + B + C + (AB - A - B) + (AC - A - C) + (BC - B - C)
=  ABC - A -B - C


INT ABCD = ABCD - ABC - ABD - ACD - BCD + AB + AC + AD + BC + BD + CD - A - B - C - D

INT ABCD + ABC + ABD + ACD + BCD + AB + AC + AD + BC + BD + CD =

ABCD - ABC - ABD - ACD - BCD + AB + AC + AD + BC + BD + CD - A - B - C - D
ABC - AB - AC - BC + A + B + C
ABD - AB - AD - BD + A + B + D
ACD - AC - AD - CD + A + C + D
BCD - BC - BD - CD + B + C + D
AB - A - B
AC - A - C
AD - A - D
BC - B - C
BD - B - D
CD - C - D


FINAL: ABCD - A - B - C  D

 */

pub struct Scorecast {
    root: Arc<RwLock<ScorecastNode>>,
    best_child_sums: Arc<RwLock<HashMap<(usize,usize), Option<f32>>>>
}

impl Scorecast {
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
    pub fn create_scorecast(
        root: NodeIterator,
        tree_height: usize,
        target_height: usize,
        target_num_keys: usize,
        num_letters: usize,
        max_key_len: usize,
    ) -> Scorecast {
        let scorecast: Scorecast = Scorecast::default();
        // scorecast = Self::setup_scorecast_tree(scorecast, &[], root, target_num_keys, num_letters);
        scorecast
    }
    fn calculate_scorecast_score(parent: &NodeIterator, child: &NodeIterator) -> f32 {
        //Wrong: INT(ABC) + INT(AB) + INT(AC) + INT(AD) = S(ABCD) - S(A) - S(B) - S(C) - S(D)
        //Right: At C, INT(ABC) + INT(BC) + INT(AC), everything that intersects with C
        /*
        INT(ABC) + INT(BC) + INT(AC) = (ABC - AB - AC - BC + A + B + C) + (BC - B - C) + (AC-A-C)

        = ABC - AB - C
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
        =
        ABCD - ABC
         */
        // ^ add on node score Scorecast(ab cd ef) => S(ef) + S(ab cd ef) - S(ab) - S(cd) - S(ef)
        let node_score_child = child.current_node.read().score;
        let node_score_parent = parent.current_node.read().score;
        return node_score_child - node_score_parent;
    }

    fn set_child_sums_helper2(
        scorecast: &Scorecast,
        root: Arc<RwLock<ScorecastNode>>,
        current: Arc<RwLock<ScorecastNode>>,
        remaining_num_letters: usize,
        remaining_num_keys: usize,
        best_scores_for: &mut HashMap<(usize, usize), f32>, // num keys, num letters
    ) {
        if remaining_num_keys > remaining_num_letters {
            current.write().child_sum = f32::MAX;
            return;
        }
        if remaining_num_keys == remaining_num_letters {
            current.write().child_sum = 0.0;
            return;
        }
        let current_children = current.read().children.clone();
        let mut child_sum: f32 = f32::MAX;
        for child in current_children {
            let child_letters = child.read().num_letters;
            if child.read().num_letters <= remaining_num_letters {
                let mut find = best_scores_for
                    .get(&(remaining_num_keys, remaining_num_letters))
                    .cloned();
                if find.is_none() {
                    find = Self::dfs_min_score(
                        root.clone(),
                        remaining_num_keys,
                        remaining_num_letters,
                    );
                    best_scores_for
                        .insert((remaining_num_keys, remaining_num_letters), find.unwrap());
                }
                child_sum = child_sum.min(find.unwrap());
                Self::set_child_sums_helper(
                    scorecast,
                    root.clone(),
                    child.clone(),
                    remaining_num_letters - child_letters,
                    remaining_num_keys - 1,
                    best_scores_for,
                );
            }
        }
        current.write().child_sum = child_sum;
    }

    // TODO, use f64's everywhere to keep precision, save as f32 at end
    fn set_child_sums_helper(
        scorecast: &Scorecast,
        root: Arc<RwLock<ScorecastNode>>,
        current: Arc<RwLock<ScorecastNode>>,
        remaining_num_letters: usize,
        remaining_num_keys: usize,
        best_scores_for: &mut HashMap<(usize, usize), f32>, // num keys, num letters
    ) -> f32 {
        if remaining_num_keys > remaining_num_letters {
            current.write().child_sum = f32::MAX;
            return f32::MAX;
        }
        if remaining_num_keys == remaining_num_letters {
            current.write().child_sum = 0.0;
            return 0.0;
        }
        let current_children = current.read().children.clone();
        if current_children.len() == 0 {
            let mut child_sum = best_scores_for
                .get(&(remaining_num_keys, remaining_num_letters))
                .cloned();
            if child_sum.is_none() {
                let min_score =
                    Self::dfs_min_score(root, remaining_num_keys, remaining_num_letters);
                let min_score = min_score.unwrap_or(f32::MAX);
                best_scores_for.insert((remaining_num_keys, remaining_num_letters), min_score);
                child_sum = Some(min_score);
            }
            current.write().child_sum = child_sum.unwrap();
            return child_sum.unwrap();
        } else {
            let mut child_sum = f32::MAX;
            for child in current_children {
                let child_letters = child.read().num_letters;
                if child.read().num_letters <= remaining_num_letters {
                    let child_find_sum = Self::set_child_sums_helper(
                        scorecast,
                        root.clone(),
                        child.clone(),
                        remaining_num_letters - child_letters,
                        remaining_num_keys - 1,
                        best_scores_for,
                    );
                    child_sum = child_sum.min(child_find_sum + child.read().score);
                }
            }
            current.write().child_sum = child_sum;
            return child_sum;
        }
    }
    pub fn set_child_sums(&self, target_num_letters: usize, target_num_keys: usize) {
        let mut best_scores_for: HashMap<(usize, usize), f32> = HashMap::new();
        Self::set_child_sums_helper(
            &self,
            self.root.clone(),
            self.root.clone(),
            target_num_letters,
            target_num_keys,
            &mut best_scores_for,
        );
        // Self::set_child_sums_helper2(
        //     &self,
        //     self.root.clone(),
        //     self.root.clone(),
        //     target_num_letters,
        //     target_num_keys,
        //     &mut best_scores_for,
        // )
    }

    pub fn get_add_amount(&self,target_num_keys: usize, target_num_letters: usize)->Option<f32>{
        if self.root.read().children.len() == 0 {
            return None;
        }
        let tup = (target_num_keys, target_num_letters);
        if self.best_child_sums.read().contains_key(&tup){
            return *self.best_child_sums.read().get(&tup).unwrap();
        }
        else {
            println!("Calculating add for {}K,{}L", target_num_keys, target_num_letters);
            let min_score = Self::dfs_min_score(self.root.clone(), target_num_keys, target_num_letters);
            self.best_child_sums.write().insert(tup, min_score);
            return min_score;
        }
    }
    // Function to find the path with the smallest score using DFS
    pub fn dfs_min_score(
        root: Arc<RwLock<ScorecastNode>>,
        desired_key_len: usize,
        desired_num_letters: usize,
    ) -> Option<f32> {
        fn dfs(
            root: Arc<RwLock<ScorecastNode>>,
            node: Arc<RwLock<ScorecastNode>>,
            num_keys_remaining: usize,
            num_letters_remaining: usize,
            mut cumulative_score: f32,
            mut best_found: Option<f32>,
            max_allowed_key_size: usize,
        ) -> Option<f32> {
            cumulative_score += node.read().score;
            let num_keys_remaining = if node.read().num_letters > 0 {
                num_keys_remaining - 1
            } else {
                num_keys_remaining
            };
            let num_letters_remaining = num_letters_remaining - node.read().num_letters;
            if num_keys_remaining == num_letters_remaining {
                if let Some(bf) = best_found {
                    best_found = Some(bf.min(cumulative_score));
                } else {
                    best_found = Some(cumulative_score);
                }
                return best_found;
            }
            if num_keys_remaining == 0
                || num_keys_remaining > num_letters_remaining
                || node.read().num_letters > num_letters_remaining
            {
                return None;
            }

            if node.read().children.is_empty() {
                let res = dfs(
                    root.clone(),
                    root,
                    num_keys_remaining,
                    num_letters_remaining,
                    cumulative_score,
                    best_found,
                    max_allowed_key_size
                );
                match (res.is_some(), best_found.is_some()) {
                    (true, true) => {
                        let res = res.unwrap();
                        let best_found = best_found.unwrap();
                        if res < best_found {
                            return Some(res);
                        }
                        return Some(best_found);
                    }
                    (true, false) => {
                        return res;
                    }
                    (_, _) => {
                        return best_found;
                    }
                }
            } else {
                for child in &node.read().children {
                    if child.read().num_letters <= num_letters_remaining && child.read().num_letters <= max_allowed_key_size {
                        let res = dfs(
                            root.clone(),
                            child.clone(),
                            num_keys_remaining,
                            num_letters_remaining,
                            cumulative_score,
                            best_found,
                            max_allowed_key_size.min(child.read().num_letters)
                        );
                        match (res.is_some(), best_found.is_some()) {
                            (true, true) => {
                                let res_uw = res.unwrap();
                                let best_found_uw = best_found.unwrap();
                                if res_uw < best_found_uw {
                                    best_found = res;
                                }
                            }
                            (true, false) => {
                                best_found = res;
                            }
                            (_, _) => {}
                        }
                    }
                }
            }
            return best_found;
        }

        dfs(
            root.clone(),
            root,
            desired_key_len,
            desired_num_letters,
            0.0,
            None,
            desired_num_letters,
        )
    }

    // fn add_missing_scorecast_nodes(
    //     &mut self,
    //     current: Rc<RefCell<ScorecastNode>>,
    //     path: Vec<usize>,
    //     tree_height: usize,
    //     target_height: usize,
    //     num_letters_remaining: usize,
    //     target_num_keys: usize,
    //     max_key_len: usize,
    // ) {
    //     if num_letters_remaining <= 1 {
    //         return;
    //     }
    //     for child in &current.as_ref().borrow().children {
    //         // At leaf, operate
    //         if child.as_ref().borrow().children.is_empty() {
    //             // Get num letters we need to iterate through
    //             let node_start = path.len() + 1 - tree_height;
    //             let last_n_keys = &path[node_start..path.len()].to_vec();
    //             let possible_key_lengths = Self::get_possible_key_lengths(
    //                 num_letters_remaining,
    //                 target_num_keys,
    //                 max_key_len,
    //             );
    //             for key in possible_key_lengths {
    //                 let mut search = last_n_keys.clone();
    //                 Self::ordered_insert(&mut search, key);
    //                 if let Some(node) = self.find_node(&search) {
    //                     child.borrow_mut().children.push(node.clone());
    //                     self.add_missing_scorecast_nodes(
    //                         node,
    //                         path.clone(),
    //                         tree_height,
    //                         target_height,
    //                         num_letters_remaining - key,
    //                         target_num_keys,
    //                         max_key_len,
    //                     );
    //                 }
    //             }
    //         }
    //     }
    // }

    pub fn setup_scorecast_tree(&mut self, current_iter: &NodeIterator) {
        self.root = Arc::new(RwLock::new(ScorecastNode {
            num_letters: 0,
            score: 0.0,
            child_sum: f32::MAX,
            children: vec![],
        }));
        self.best_child_sums.write().clear();
        self.setup_scorecast_tree_helper(&[], current_iter)
    }
    fn setup_scorecast_tree_helper(&mut self, key_len_path: &[usize], current_iter: &NodeIterator) {
        for child in &current_iter.current_node.read().children {
            // println!(
            //     "Child {} children: {}",
            //     set32_to_string(child.as_ref().borrow().letters),
            //     set32s_to_string(
            //         &child
            //             .as_ref()
            //             .borrow()
            //             .children
            //             .iter()
            //             .map(|child| child.as_ref().borrow().letters)
            //             .collect::<Vec<Set32>>()
            //     )
            // );
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
                if our_child.read().score > scorecast_score {
                    our_child.write().score = scorecast_score;
                }
            } else {
                self.insert_node_from_root(
                    &key_len_path,
                    ScorecastNode {
                        child_sum: f32::MAX,
                        num_letters: child_key_len,
                        children: vec![],
                        score: scorecast_score,
                    },
                )
            }
            Self::setup_scorecast_tree_helper(self, &full_key_len_path, &child_iter);
        }
    }

    fn find_node(&self, path: &[usize]) -> Option<Arc<RwLock<ScorecastNode>>> {
        if path.is_empty() {
            return Some(self.root.clone());
        }

        let mut current: Arc<RwLock<ScorecastNode>> = self.root.clone();
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
    fn insert_node_from_root(&mut self, path: &[usize], node: ScorecastNode) {
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

impl Default for Scorecast {
    fn default() -> Self {
        Self {
            root: Arc::new(RwLock::new(ScorecastNode {
                num_letters: 0,
                score: 0.0,
                child_sum: f32::MAX,
                children: vec![],
            })),
            best_child_sums: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

pub struct ScorecastNode {
    num_letters: usize,
    score: f32,
    child_sum: f32,
    children: Vec<Arc<RwLock<ScorecastNode>>>,
}

impl ScorecastNode {
    pub fn new(
        num_letters: usize,
        score: f32,
        child_sum: f32,
        children: Vec<Arc<RwLock<ScorecastNode>>>,
    ) -> ScorecastNode {
        ScorecastNode {
            num_letters,
            score,
            child_sum,
            children,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use crate::{
        create_set32_str,
        globals::{CHAR_TO_INDEX, NUM_LETTERS},
        node::Node,
        node_iterator::NodeIterator,
        set32::Set32,
    };

    use super::{Scorecast, ScorecastNode};
    fn create_empty_node_iterator() -> NodeIterator {
        let root = Arc::new(RwLock::new(Node::new(0.0, Set32::EMPTY, vec![])));
        NodeIterator {
            current_node: root.clone(),
            path: vec![],
            root,
        }
    }
    fn get_simple_scorecast() -> Scorecast {
        let mut scorecast = Scorecast::default();
        scorecast.insert_node_from_root(&[], ScorecastNode::new(2, 12.12, 0.0, vec![]));
        scorecast.insert_node_from_root(&[], ScorecastNode::new(4, 2.9, 0.0, vec![]));
        scorecast.insert_node_from_root(&[], ScorecastNode::new(5, 2.9, 0.0, vec![]));
        scorecast
    }

    fn get_complex_scorecast() -> Scorecast {
        let mut scorecast = Scorecast::default();
        scorecast.insert_node_from_root(&[], ScorecastNode::new(2, 0.112, 0.0, vec![]));
        let c4_1 = Arc::new(RwLock::new(ScorecastNode::new(3, 1.2, 0.0, vec![])));
        scorecast.insert_node_from_root(&[], ScorecastNode::new(4, 2.9, 0.0, vec![c4_1]));

        scorecast.insert_node_from_root(&[], ScorecastNode::new(5, 1.5, 0.0, vec![]));
        scorecast
    }

    #[test]
    fn empty_score_cast(){
        let mut scorecast = Scorecast::default();
        let add = scorecast.get_add_amount(10, 11);
        assert!(add.is_none());
    }
    #[test]
    fn scorecast_set_child_sums_complex(){
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
        
        let mut scorecast = Scorecast::default();
        scorecast.setup_scorecast_tree(&iter);  
        /*
            7k, 7l: 0, all 1 keys
            3k, 7l: Scorecast(xyz) + scorecast(xyz ab) + scorecast(ab) = 0.25 + (0.4-0.25) + 0.1 = 0.5
         */
        let add = scorecast.get_add_amount(3, 7);
        assert!(add.is_some());
        assert_eq!(add.unwrap(), 0.5);
        iter.insert_node_from_root(&[create_set32_str!("xyz"),create_set32_str!("bc")], xyz_bc_df.clone());
        scorecast.setup_scorecast_tree(&iter);
        // 3 keys, 7 letters: xyz + xyz_ab + xyz_bc_df = Scorecast(xyz) + Scorecast(xyz ab) + Scorecast(xyz bc df) = 0.25 + (0.4-0.25) + 0.78-0.55 = 0.63
        let add = scorecast.get_add_amount(3, 7);
        assert!(add.is_some());
        assert_eq!(add.unwrap(), 0.63);
    }

    #[test]
    fn scorecast_set_child_sums() {
        let mut iter = create_empty_node_iterator();
        let ab = Node::new(0.1, create_set32_str!("ab"), vec![]);
        let xyz_ab = Node::new(0.4, create_set32_str!("ab"), vec![]);
        let xyz = Node::new(0.25, create_set32_str!("xyz"), vec![]);
        iter.insert_node_from_root(&[], ab.clone());
        iter.insert_node_from_root(&[], xyz.clone());
        iter.insert_node_from_root(&[create_set32_str!("xyz")], xyz_ab.clone());
        let mut scorecast = Scorecast::default();
        scorecast.setup_scorecast_tree(&iter);

        // ab, ab, _ and xyz _ _ is the only way to make 5 letters with 3 keys
        // Since ab ab is less than xyz, we should use ab ab = 0.2 at root
        // child sum of ab should be Scorecast(ab) = S(A) = 0.1
        // child sum of xyz should be Scorecast(nothing) = 0
        scorecast.set_child_sums(5, 3);
        let ab_sc = scorecast.find_node(&[2]);
        let xyz_sc = scorecast.find_node(&[3]);
        let root_sc = scorecast.find_node(&[]);
        assert!(xyz_sc.is_some());
        assert!(ab_sc.is_some());
        assert!(root_sc.is_some());
        println!(
            "ab child_sum: {}",
            ab_sc.clone().unwrap().read().child_sum
        );
        assert!(
            (ab_sc.clone().unwrap().read().child_sum - ab.score).abs() < f32::EPSILON
        );
        println!(
            "xyz child_sum: {}",
            xyz_sc.clone().unwrap().read().child_sum
        );
        println!(
            "root child_sum: {}",
            root_sc.clone().unwrap().read().child_sum
        );
        assert!((xyz_sc.clone().unwrap().read().child_sum).abs() < f32::EPSILON);
        assert!(
            (root_sc.clone().unwrap().read().child_sum
                - 2.0 * ab_sc.clone().unwrap().read().child_sum)
                .abs()
                < f32::EPSILON
        );
        println!("Modified tree");
        iter.find_node_from_root(&[create_set32_str!("ab")])
            .unwrap()
            .as_ref()
            .write()
            .score = 0.28;
        scorecast.setup_scorecast_tree(&iter);
        // Now we should get a different result, xyz is least
        scorecast.set_child_sums(5, 3);
        let root_sc = scorecast.find_node(&[]);
        println!(
            "root child_sum: {}",
            root_sc.clone().unwrap().read().child_sum
        );
        assert!(
            (root_sc.clone().unwrap().read().child_sum - xyz.score).abs() < f32::EPSILON
        );
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
        let mut scorecast = Scorecast::default();
        scorecast.setup_scorecast_tree(&iter);
        let ab_score = scorecast.find_node(&[2]);
        assert!(ab_score.is_some());
        assert!((ab_score.unwrap().read().score - 0.1).abs() < f32::EPSILON);
        let xyz_ab_score = scorecast.find_node(&[3, 2]);
        assert!(xyz_ab_score.is_some());
        // S(ab) + INT(xyz ab) = 0.1 + (0.4-0.2-0.1) = 0.2
        println!(
            "{} score",
            xyz_ab_score.clone().unwrap().read().score
        );
        assert!((xyz_ab_score.unwrap().read().score - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn scorecast_dfs_min_score() {
        let scorecast = get_complex_scorecast();
        let letters2 = Scorecast::dfs_min_score(scorecast.root.clone(), 1, 2);
        assert!(letters2.is_some());
        assert_eq!(letters2.unwrap(), 0.112);
        let letters22 = Scorecast::dfs_min_score(scorecast.root.clone(), 2, 4);
        assert!(letters22.is_some());
        assert_eq!(letters22.unwrap(), 0.112 * 2.0);
        let letters4 = Scorecast::dfs_min_score(scorecast.root.clone(), 1, 4);
        assert!(letters4.is_some());
        assert_eq!(letters4.unwrap(), 2.9);
        let letters5 = Scorecast::dfs_min_score(scorecast.root.clone(), 2, 6);
        assert!(letters5.is_some());
        assert_eq!(letters5.unwrap(), 1.5);
    }

    #[test]
    fn scorecast_sum_to_n_k_terms() {
        let n = 27;
        let k = 10;
        let mut results = Vec::new();
        let mut current = Vec::new();

        Scorecast::sum_to_n_k_terms_helper(n, k, 1, &mut current, &mut results);
        let len = results.len();
        for result in results {
            println!("{:?}", result.into_iter().rev().collect::<Vec<usize>>());
        }
        println!("{}", len);
    }

    #[test]
    fn scorecast_get_possible_key_lengths() {
        // 3 2 1 and 2 2 2
        let key_lengths = Scorecast::get_possible_key_lengths(6, 3, 3);
        println!("{:?}", key_lengths);
        assert_eq!(key_lengths.len(), 2);
    }

    #[test]
    fn scorecast_insert_find_node() {
        let mut scorecast = get_simple_scorecast();
        let find = scorecast.find_node(&[2]);
        assert!(find.is_some());
        assert_eq!(find.unwrap().read().score, 12.12);
        let find = scorecast.find_node(&[4]);
        assert!(find.is_some());
        assert_eq!(find.unwrap().read().score, 2.9);
        scorecast.insert_node_from_root(
            &[2],
            ScorecastNode {
                child_sum: 0.0,
                children: vec![],
                score: 1000.0,
                num_letters: 2,
            },
        );
        let find = scorecast.find_node(&[2, 2]);
        assert!(find.is_some());
        assert_eq!(find.unwrap().read().score, 1000.0);
    }
    #[test]
    fn scorecast_create_scorecast_helper() {
        let mut iter = create_empty_node_iterator();
        let ab = Node {
            children: vec![],
            letters: create_set32_str!("ab"),
            score: 0.15,
        };
        iter.insert_node_from_root(&[], ab);
        let scorecast = Scorecast::create_scorecast(iter, 1, 0, 10, NUM_LETTERS, 5);
        let scorecast_children = &scorecast.root.read().children;
        assert_eq!(scorecast_children.len(), 0);
    }
}
