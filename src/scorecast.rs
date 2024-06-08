use std::{collections::HashMap, sync::Arc};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;

use itertools::Itertools;

use crate::set32::Set32;

/*
Plan:
Iterate through tree layer by layer. At each node, calculate its score + intersections with children above it.
Check if score < score for that node. If so, set it


*/

pub struct Scorecast {
    root: Arc<RwLock<ScorecastNode>>,
    best_sums: Arc<RwLock<HashMap<(usize, usize), Option<f32>>>>, // letters, keys
    scorecast_scores_by_layer: Arc<RwLock<Vec<Arc<RwLock<HashMap<Set32, f32>>>>>>,
}

impl Scorecast {
    
    fn generate_balanced_no_ones(num_letters: usize, num_keys: usize) -> Vec<usize> {
        //19, 9letters, we we want 8*2, 1*3
        let real_num_terms = std::cmp::min(num_letters/2, num_keys);
        let mut most_balanced = vec![2; real_num_terms];
        let mut remainder = num_letters - 2*real_num_terms;
        for i in 0..most_balanced.len() {
            if remainder > 0 {
                most_balanced[i]+=1;
                remainder-=1;
            }
        }
        most_balanced
    }
    
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
    pub fn calculate_scorecast_score(parent_score: f32, child_score: f32) -> OrderedFloat<f32> {
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
        OrderedFloat(child_score - parent_score)
    }

    pub fn update_scorecast_score(&self, layer: u8, key: Set32, scorecast_score: f32) {
        // let guard = self.scorecast_scores_by_layer.upgradable_read();
        let layer_map = self
            .scorecast_scores_by_layer
            .read()
            .get(layer as usize)
            .cloned();
        if layer_map.is_none() {
            let mut write_guard = self.scorecast_scores_by_layer.write();

            // Make sure no one else has changed it in the meantime
            if write_guard.get(layer as usize).is_none() {
                let new_map = Arc::new(RwLock::new(HashMap::new()));
                if layer as usize >= write_guard.len() + 1 {
                    write_guard.resize(layer as usize + 1, new_map.clone());
                } else {
                    write_guard.insert(layer as usize, new_map.clone());
                }
                drop(write_guard);
                new_map.write().insert(key, scorecast_score);
                return;
            }
        }
        // Need to redeclare scores_by_layer for safety
        // Important not to do "else if" here since this might still need to run if inner "if" above fails
        if let Some(layer_map) = self
            .scorecast_scores_by_layer
            .read()
            .get(layer as usize)
            .cloned()
        {
            if layer_map.read().get(&key).is_none() {
                layer_map.write().insert(key, scorecast_score);
            } else if *layer_map.read().get(&key).unwrap() > scorecast_score {
                layer_map.write().insert(key, scorecast_score);
            }
        }
    }

    fn calculate_best_groupings(
        &self,
        current: Arc<RwLock<ScorecastNode>>,
        parent: Option<Arc<RwLock<ScorecastNode>>>,
        path: &[usize],
    ) {
        
        /*
            E.g. abc def 3-3, should look layer above for abc, abd, abe, abf, acd, ace, acf, ... 15 total
            abc defghi 3-6

            3-3 
            (27 choose 3)(24 choose 3)/2 = 2,960,100
            6-3
            (27 choose 6)(21 choose 3) = 393,693,300
            9-3
            (27 choose 9)(18 choose 3) = 3,824,449,200. Will store a Set32, Score = 64 bits = 8 bytes = 8*3B = 24GB
            12-2
            (27 choose 12)(15 choose 2) = 1,825,305,300
            14-2
            16-2
            18-2
            20-2
         */        
        // Need to iterate through all combinations of our letters and for each iteration, iterate through all of parents letters
        // excluding our letters. Sum up our scorecast score with parents score and get the min

        // let best_groupings = HashMap::new();
    }
    fn get_node_at_path(&self, path: &[usize]) -> Arc<RwLock<ScorecastNode>> {
        let mut current_node = self.root.clone();
        for key in path {
            let find = current_node
                .read()
                .children
                .iter()
                .find(|child| child.read().num_letters == *key)
                .unwrap()
                .clone();
            current_node = find;
        }
        current_node
    }
    pub fn get_add_amount(&self, target_num_keys: usize, target_num_letters: usize, max_key_len: usize, threshold: f32) -> Option<f32> {
        // 25 more letters, 9 more keys 
        // 25 - 9 + 1
        if target_num_letters < target_num_keys {
            panic!("Should never be looking for fewer letters than keys");
        }
        if self.best_sums.read().contains_key(&(target_num_letters, target_num_keys)) {
            return self.best_sums.read().get(&(target_num_letters, target_num_keys)).unwrap().clone()
        }
        let fewer_letters = target_num_letters - target_num_keys + 1;
        let config = Self::generate_balanced_no_ones(fewer_letters, target_num_letters);
        for key_len in config {
            /*
                Steps: 
                Create a fn that finds the best grouping of letters (abcdef) by looking at all subsets of self and then recurse on that grouping. 
             */
        }
        None
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
                best_groupings: Arc::new(RwLock::new(HashMap::new())),
                children: vec![],
            })),
            best_sums: Arc::new(RwLock::new(HashMap::new())),
            scorecast_scores_by_layer: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

pub struct ScorecastNode {
    num_letters: usize,
    best_groupings: Arc<RwLock<HashMap<Set32, f32>>>,
    children: Vec<Arc<RwLock<ScorecastNode>>>,
}

impl ScorecastNode {
    pub fn new(
        num_letters: usize,
        best_groupings: Arc<RwLock<HashMap<Set32, f32>>>,
        children: Vec<Arc<RwLock<ScorecastNode>>>,
    ) -> ScorecastNode {
        ScorecastNode {
            num_letters,
            best_groupings,
            children,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use hashbrown::{HashMap, HashSet};
    use rand::Rng;

    use crate::scorecast::Scorecast;
    fn generate_hash_pairs_and_find_min_sum(num_pairs: usize) -> Option<u64> {
        let mut rng = rand::thread_rng();
        let mut min_sum: Option<u64> = None;
    
        for i in 0..num_pairs {
            let key1 = rng.gen::<u64>();
            let key2 = rng.gen::<u64>();
            let value1 = rng.gen::<u64>();
            let value2 = rng.gen::<u64>();
    
            let mut hash_table = HashMap::new();
            hash_table.insert(key1, value1);
            hash_table.insert(key2, value2);
    
            if let (Some(&val1), Some(&val2)) = (hash_table.get(&key1), hash_table.get(&key2)) {
                if i % 100000 == 0 {
                    // println!("Evaluating {}", i);
                }
                
                let sum = val1 + val2;
                min_sum = match min_sum {
                    Some(current_min) => Some(current_min.min(sum)),
                    None => Some(sum),
                };
            }
        }
    
        min_sum
    }
    #[test]
    fn scorecast_balance_sums(){
        let most_balanced = Scorecast::generate_balanced_no_ones(15, 10);
        let result = vec![3,2,2,2,2,2,2];
        assert_eq!(most_balanced, result);
        println!("Most balanced: {:?}", most_balanced);
    }
    #[test]
    fn scorecast_find_min_sum(){

        let start = Instant::now();
        let min_sum = generate_hash_pairs_and_find_min_sum(100_000_000);
        let duration = start.elapsed();
    
        match min_sum {
            Some(min) => println!("Minimum sum: {}", min),
            None => println!("No pairs found"),
        }
    
        println!("Time taken: {:?}", duration);
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

    /*
        25 letters, 9 keys, later 
                         1*17, 8*1.      Eval, now can use 18 letters, 9k
                         1*16, 1*2, 7*1. 
                         1*15, 1*3, 7*1
                         1*14, 2*2, 7*1. Eval, now can use 19 letters, 9k
                         1*13, 5*1, 7*1. 
                         ...
                         3*8, 1*1
                         Calc 1*17 and 14. We get 1.2 and 1.3, as long as our score < those, we can call it valid

                         3,3,3,3,3,2,2



                         If max == 6
                            3*6, 1*2, 5*1, can use 20 keys

        Later on: 24 letters, 9 keys
                        1*16

        
     */ 

    #[test]
    fn count_num_nodes() {
        let n = 27;
        let k = 10;
        let mut results = Vec::new();
        let mut current = Vec::new();

        Scorecast::sum_to_n_k_terms_helper(n, k, 1, &mut current, &mut results);
        // for result in results.clone() {
        //     println!("{:?}", result.into_iter().rev().collect::<Vec<usize>>());
        // }
        let mut unique_path_count: HashSet<Vec<usize>> = HashSet::new();
        for vec in results {
            let mut path = Vec::new();
            for node in vec.into_iter().rev() {
                if node == 1 || node > 6 {
                    break;
                }
                path.push(node);
                unique_path_count.insert(path.clone());
            }
        }
        println!("Unique paths:");
        for node in unique_path_count.iter() {
            println!("{:?}", node);   
        }
        println!("Num unique nodes: {}", unique_path_count.len());

    }
}
