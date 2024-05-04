use std::{
    cmp::Ordering,
    sync::{Arc, RwLock},
};

use itertools::Itertools;

use crate::{node::Node, set32::Set32, util::compare_letters};
#[derive(Clone)]
pub struct NodeIterator {
    pub root: Arc<RwLock<Node>>,
    pub current_node: Arc<RwLock<Node>>,
    pub path: Vec<Set32>,
}

impl NodeIterator {
    pub fn create_empty() -> NodeIterator {
        let root = Arc::new(RwLock::new(Node::new(0.0, Set32::EMPTY, vec![])));
        NodeIterator {
            current_node: root.clone(),
            path: vec![],
            root,
        }
    }
    pub fn new(root: Arc<RwLock<Node>>, current_node: Arc<RwLock<Node>>, path: Vec<Set32>) -> Self {
        NodeIterator {
            root,
            current_node,
            path,
        }
    }

    pub fn get_keys_to_evaluate(
        num_keys: usize,
        num_letters: usize,
        max_key_len: usize,
    ) -> Vec<Set32> {
        let mut combinations: Vec<Vec<u32>> = Vec::new();
        let theoretical_max_key_len = num_letters - num_keys + 1; // e.g. 18 is max len for 10 key, 27 letters. 18*1 + 1*9
        for i in 2..=std::cmp::min(theoretical_max_key_len, max_key_len) {
            let numbers: Vec<u32> = (0..num_letters as u32).collect();
            let mut combinations_i = numbers.into_iter().combinations(i).collect::<Vec<_>>();
            combinations_i.reverse();
            combinations.append(&mut combinations_i);
        }
        combinations.reverse();
        let mut result = vec![];
        for vec in combinations.iter() {
            let mut s = Set32::EMPTY;
            for val in vec {
                s = s.add(num_letters as u32 - 1 - *val);
            }
            result.push(s);
        }
        result.reverse();
        result
    }
    pub fn insert_node_from_here(&self, node: Node) {
        let mut write_guard = self.current_node.write().unwrap();
        let node_idx = write_guard
            .children
            .binary_search_by(|n| compare_letters(n.read().unwrap().letters, node.letters));

        write_guard
            .children
            .insert(node_idx.unwrap_or_else(|e| e), Arc::new(RwLock::new(node)));
    }

    // TODO, if use anywhere, make safe like insert_from_here
    pub fn insert_node_from_root(&self, path: &[Set32], node: Node) {
        let mut current_node: Arc<RwLock<Node>> = self.root.clone();
        for key in path.iter() {
            let node_idx = current_node
                .read()
                .unwrap()
                .children
                .binary_search_by(|n| compare_letters(n.read().unwrap().letters, *key));

            // Handle the result of the binary search
            current_node = match node_idx {
                Ok(idx) => {
                    // If the key exists, move deeper into the tree
                    Arc::clone(&current_node.read().unwrap().children[idx]) // Clone the Rc of the next node in the path
                }
                Err(_) => {
                    panic!("Node does not exist"); // Handling as per your original logic
                }
            };
        }

        // Now insert the final node at the current location
        current_node
            .write()
            .unwrap()
            .children
            .push(Arc::new(RwLock::new(node)));
    }

    pub fn calculate_predecessors_from_path(path: &[Set32]) -> Vec<Vec<Set32>> {
        if path.is_empty() {
            panic!("Shouldn't call on empty path");
        }
        let mut result: Vec<Vec<Set32>> = Vec::new();
        for index in 0..path.len() {
            let mut predecessor = path.to_vec();
            predecessor.remove(index);
            result.push(predecessor);
        }
        result
    }

    pub fn get_predecessors_from_path(&self, path: &[Set32]) -> Option<Vec<NodeIterator>> {
        let mut iterators: Vec<NodeIterator> = vec![];
        for predecessor in Self::calculate_predecessors_from_path(&path) {
            let predecessor_iter = self.find_node_iter_from_root(&predecessor);
            if predecessor_iter.is_none() {
                return None;
            }
            iterators.push(predecessor_iter.unwrap());
        }
        Some(iterators)
    }

    fn intersect_sorted_vectors_2(vec_a: &[Set32], vec_b: &[Set32]) -> Vec<Set32> {
        let mut result = Vec::new();
        let mut it_a = vec_a.iter();
        let mut it_b = vec_b.iter();

        let mut value_a = it_a.next();
        let mut value_b = it_b.next();

        while value_a.is_some() && value_b.is_some() {
            match (value_a, value_b) {
                (Some(a), Some(b)) if a == b => {
                    result.push(*a);
                    value_a = it_a.next();
                    value_b = it_b.next();
                }
                (Some(a), Some(b)) if compare_letters(*a, *b) == Ordering::Less => {
                    value_a = it_a.next()
                }
                _ => value_b = it_b.next(),
            }
        }

        result
    }
    pub fn node_to_iter(&self, node: Arc<RwLock<Node>>, path: &[Set32]) -> NodeIterator {
        NodeIterator {
            root: self.root.clone(),
            current_node: node.clone(),
            path: path.to_vec(),
        }
    }
    pub fn find_node_iter_from_root(&self, path: &[Set32]) -> Option<NodeIterator> {
        let node = self.find_node_from_root(path);
        if node.is_none() {
            return None;
        }
        Some(self.node_to_iter(node.unwrap(), path))
    }
    pub fn find_node_iter_from_here(&self, path: &[Set32]) -> Option<NodeIterator> {
        let node = self.find_node_from_here(path);
        if node.is_none() {
            return None;
        }
        Some(self.node_to_iter(node.unwrap(), path))
    }
    pub fn find_node_from_node(
        node: Arc<RwLock<Node>>,
        path: &[Set32],
    ) -> Option<Arc<RwLock<Node>>> {
        if path.is_empty() {
            return Some(node.clone());
        }

        let mut current: Arc<RwLock<Node>> = node.clone();
        for key in path {
            let search = &current
                .read()
                .unwrap()
                .children
                .binary_search_by(|node| compare_letters(node.read().unwrap().letters, *key));
            match search {
                Ok(index) => {
                    let clone = current.read().unwrap().children[*index].clone();
                    current = clone;
                }
                Err(_) => return None, // Return None if any key in the path does not match.
            }
        }
        Some(current)
    }
    pub fn find_node_from_root(&self, path: &[Set32]) -> Option<Arc<RwLock<Node>>> {
        Self::find_node_from_node(self.root.clone(), path)
    }
    pub fn find_node_from_here(&self, path: &[Set32]) -> Option<Arc<RwLock<Node>>> {
        Self::find_node_from_node(self.current_node.clone(), path)
    }
    fn get_end_idx(keys: &[Set32], target: Set32) -> usize {
        let start = keys.binary_search_by(|node| compare_letters(*node, target));
        if start.is_ok() {
            start.unwrap() + 1
        } else {
            start.unwrap_err()
        }
    }
    pub fn new_get_children_to_evaluate(
        &self,
        num_keys: usize,
        num_letters: usize,
        max_key_len: usize,
    ) -> Option<Vec<Set32>> {
        let mut letters_to_exclude = Set32::EMPTY;
        for key in &self.path {
            letters_to_exclude = letters_to_exclude.union(*key);
        }
        return match &self.path.is_empty() {
            false => {
                let predecessor_paths = Self::calculate_predecessors_from_path(&self.path);
                let mut valid_predecessor_children: Option<Vec<Set32>> = None;
                for path in predecessor_paths {
                    let node = self.find_node_from_root(&path);
                    if node.is_none() {
                        return None;
                    }
                    let node = node.unwrap();
                    let node_children = &node.read().unwrap().children;
                    if node_children.is_empty() {
                        return None;
                    }
                    let end_idx: usize = Self::get_end_idx(
                        &node_children
                            .iter()
                            .map(|n| n.read().unwrap().letters)
                            .collect::<Vec<Set32>>(),
                        *self.path.last().unwrap(),
                    );
                    let valid_children = node_children
                        .iter()
                        .take(end_idx)
                        .filter(|node| {
                            node.read()
                                .unwrap()
                                .letters
                                .intersect(letters_to_exclude)
                                .is_empty()
                        })
                        .map(|node| node.read().unwrap().letters)
                        .collect();
                    if valid_predecessor_children.is_none() {
                        valid_predecessor_children = Some(valid_children);
                    } else {
                        valid_predecessor_children = Some(Self::intersect_sorted_vectors_2(
                            &valid_predecessor_children.unwrap(),
                            &valid_children,
                        ));
                    }
                }
                valid_predecessor_children
            }
            true => Some(Self::get_keys_to_evaluate(
                num_keys,
                num_letters,
                max_key_len,
            )),
        };
    }

    pub fn get_children_to_evaluate(&self, key: Set32) -> Option<Vec<Set32>> {
        let mut letters_to_exclude = key;
        for key in &self.path {
            letters_to_exclude = letters_to_exclude.union(*key);
        }
        return match &self.path.is_empty() {
            false => {
                let mut full_path = self.path.clone();
                full_path.push(key);
                let predecessor_paths = Self::calculate_predecessors_from_path(&full_path);
                let mut valid_predecessor_children: Option<Vec<Set32>> = None;
                for path in predecessor_paths {
                    let node = self.find_node_from_root(&path);
                    if node.is_none() {
                        return None;
                    }
                    let node = node.unwrap();
                    let node_children = &node.read().unwrap().children;
                    if node_children.is_empty() {
                        return None;
                    }
                    let start_idx: usize = Self::get_end_idx(
                        &node_children
                            .iter()
                            .map(|n| n.read().unwrap().letters)
                            .collect::<Vec<Set32>>(),
                        key,
                    );
                    let valid_children = node_children
                        .iter()
                        .take(start_idx)
                        .filter(|node| {
                            node.read()
                                .unwrap()
                                .letters
                                .intersect(letters_to_exclude)
                                .is_empty()
                        })
                        .map(|node| node.read().unwrap().letters)
                        .collect();
                    if valid_predecessor_children.is_none() {
                        valid_predecessor_children = Some(valid_children);
                    } else {
                        valid_predecessor_children = Some(Self::intersect_sorted_vectors_2(
                            &valid_predecessor_children.unwrap(),
                            &valid_children,
                        ));
                    }
                }
                valid_predecessor_children
            }
            true => {
                let start_idx: usize = Self::get_end_idx(
                    &self
                        .root
                        .read()
                        .unwrap()
                        .children
                        .iter()
                        .map(|n| n.read().unwrap().letters)
                        .collect::<Vec<Set32>>(),
                    key,
                );
                Some(
                    self.root
                        .read()
                        .unwrap()
                        .children
                        .iter()
                        .take(start_idx)
                        .filter(|node| {
                            node.read()
                                .unwrap()
                                .letters
                                .intersect(letters_to_exclude)
                                .is_empty()
                        })
                        .map(|node| node.read().unwrap().letters)
                        .collect::<Vec<Set32>>(),
                )
            }
        };
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    use std::sync::{Arc, RwLock};

    use crate::{
        create_set32_str, create_set32s_vec,
        globals::{CHAR_TO_INDEX, NUM_LETTERS},
        node::Node,
        set32::Set32,
    };

    use super::NodeIterator;

    fn create_empty_node_iterator() -> NodeIterator {
        let root = Arc::new(RwLock::new(Node::new(0.0, Set32::EMPTY, vec![])));
        NodeIterator {
            current_node: root.clone(),
            path: vec![],
            root,
        }
    }
    #[test]
    fn node_iterator_new_get_children_to_evaluate() {
        let root = create_empty_node_iterator();
        let base_keys = NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 2);
        let children = root.new_get_children_to_evaluate(10, NUM_LETTERS, 2);
        assert!(children.is_some());
        assert_eq!(children.unwrap().len(), 351);
        for key in base_keys {
            let new_node = Node::new(0.0, key, vec![]);
            root.insert_node_from_root(&[], new_node);
        }
        let wx = root.find_node_iter_from_root(&create_set32s_vec!("wx"));
        assert!(wx.is_some());
        let children = wx.unwrap().new_get_children_to_evaluate(10, NUM_LETTERS, 2);
        assert!(children.is_some());
        assert_eq!(children.unwrap().len(), 3);
    }

    #[test]
    fn node_iterator_insert_node() {
        let root = create_empty_node_iterator();
        let base_keys = NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 2);
        for key in base_keys {
            let new_node = Node::new(0.0, key, vec![]);
            root.insert_node_from_root(&[], new_node);
        }
        assert_eq!(root.root.read().unwrap().children.len(), 351);
    }
    #[test]
    fn node_iterator_node_find() {
        let iter = create_empty_node_iterator();
        let a = Node {
            score: 2.0,
            letters: create_set32_str!("a"),
            children: vec![],
        };
        let path = vec![a.letters];
        iter.root
            .write()
            .unwrap()
            .children
            .push(Arc::new(RwLock::new(a)));
        let found_node = iter.find_node_from_root(&path);
        assert!(found_node.is_some());
        assert_eq!(found_node.unwrap().read().unwrap().score, 2.0);

        let b = Node {
            score: 2.1,
            letters: create_set32_str!("b"),
            children: vec![],
        };
        iter.root.read().unwrap().children[0]
            .write()
            .unwrap()
            .children
            .push(Arc::new(RwLock::new(b)));
        let path = vec![create_set32_str!("a"), create_set32_str!("b")];
        assert!(iter.find_node_from_root(&path).is_some());
        assert_eq!(
            iter.find_node_from_root(&path)
                .unwrap()
                .read()
                .unwrap()
                .score,
            2.1
        );

        let path = vec![
            create_set32_str!("a"),
            create_set32_str!("b"),
            create_set32_str!("c"),
        ];
        assert!(iter.find_node_from_root(&path).is_none());

        let yz = Arc::new(RwLock::new(Node::new(1.3, create_set32_str!("yz"), vec![])));
        let ef = Arc::new(RwLock::new(Node::new(
            1.01,
            create_set32_str!("ef"),
            vec![],
        )));
        let ab = Arc::new(RwLock::new(Node::new(
            1.01,
            create_set32_str!("ab"),
            vec![],
        )));
        iter.root.write().unwrap().children.extend([yz, ef, ab]);
        assert!(iter
            .find_node_from_root(&vec![create_set32_str!("yz")])
            .is_some());
        assert!(
            iter.find_node_from_root(&vec![create_set32_str!("yz")])
                .unwrap()
                .read()
                .unwrap()
                .score
                == 1.3
        );
        assert!(iter
            .find_node_from_root(&vec![create_set32_str!("ef")])
            .is_some());
        assert!(iter
            .find_node_from_root(&vec![create_set32_str!("ab")])
            .is_some());
    }

    #[test]
    fn node_iterator_get_children_to_evaluate() {
        let root = create_empty_node_iterator();
        let base_keys = NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 2);
        for key in base_keys {
            let new_node = Node::new(0.0, key, vec![]);
            root.insert_node_from_root(&[], new_node);
        }
        let children = root.get_children_to_evaluate(create_set32_str!("wx"));
        assert!(children.is_some());
        assert_eq!(children.unwrap().len(), 3);
        let children = root.get_children_to_evaluate(create_set32_str!("ab"));
        assert!(children.is_some());
        assert_eq!(children.unwrap().len(), 300); // 351 - (26 + 25), all A's and B's
    }
    #[test]
    fn node_iterator_vector_intersect_2() {
        let a1 = create_set32s_vec!("xyz,abc");
        let a2 = create_set32s_vec!("xyz,abc, ab, yz");
        let a3 = create_set32s_vec!("xyz,qrs,def,abc");
        assert_eq!(NodeIterator::intersect_sorted_vectors_2(&a1, &a2).len(), 2);
        assert_eq!(NodeIterator::intersect_sorted_vectors_2(&a1, &a3).len(), 2);
        assert_eq!(
            NodeIterator::intersect_sorted_vectors_2(&a1, &vec![]).len(),
            0
        );
        let c1 = create_set32s_vec!("xyz,ef,cd,ab");
        let c2 = create_set32s_vec!("xyz,em,en,ef,cd,ab");
        assert_eq!(NodeIterator::intersect_sorted_vectors_2(&c1, &c2).len(), 4);
    }
    #[test]
    fn node_iterator_get_keys_to_evaluate() {
        assert_eq!(
            NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 2).len(),
            351
        );
        // 27 choose 2 + 27 choose 3
        assert_eq!(
            NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 3).len(),
            3276
        );
        assert_eq!(
            NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 3)[0],
            create_set32_str!("z\'")
        );
        assert_eq!(
            NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 3)[3275],
            create_set32_str!("abc")
        );
    }
}
