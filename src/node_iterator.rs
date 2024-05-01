use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use itertools::Itertools;

use crate::{node::Node, set32::Set32, util::compare_letters};

pub struct NodeIterator {
    pub root: Rc<RefCell<Node>>,
    pub current_node: Rc<RefCell<Node>>,
    pub path: Vec<Set32>,
}


impl NodeIterator {
    pub fn create_empty() -> NodeIterator {
        let root = Rc::new(RefCell::new(Node::new(0.0, Set32::EMPTY, vec![])));
        NodeIterator {
            current_node: root.clone(),
            path: vec![],
            root,
        }
    }    
    pub fn new(
        root: Rc<RefCell<Node>>,
        current_node: Rc<RefCell<Node>>,
        path: Vec<Set32>,
    ) -> Self {
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
    pub fn insert_node_from_here(&mut self, node: Node) {
        self.current_node
            .borrow_mut()
            .children
            .push(Rc::new(RefCell::new(node)));
    }    
    pub fn insert_node_from_root(&mut self, path: &[Set32], node: Node) {
        let mut current_node: Rc<RefCell<Node>> = self.root.clone();
        for key in path.iter() {
            let node_idx = current_node
                .borrow_mut()
                .children
                .binary_search_by(|n| compare_letters(n.as_ref().borrow().letters, *key));

            // Handle the result of the binary search
            current_node = match node_idx {
                Ok(idx) => {
                    // If the key exists, move deeper into the tree
                    Rc::clone(&current_node.as_ref().borrow().children[idx]) // Clone the Rc of the next node in the path
                }
                Err(_) => {
                    panic!("Node does not exist"); // Handling as per your original logic
                }
            };
        }

        // Now insert the final node at the current location
        current_node
            .borrow_mut()
            .children
            .push(Rc::new(RefCell::new(node)));
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

    pub fn get_predecessors_from_path(&self, path: &[Set32]) -> Option<Vec<NodeIterator>>{
        let mut iterators:Vec<NodeIterator> = vec![];
        for predecessor in Self::calculate_predecessors_from_path(&path) {
            let predecessor_iter = self.find_node_iter(&predecessor);
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
    pub fn node_to_iter(&self, node: Rc<RefCell<Node>>, path:&[Set32]) -> NodeIterator {
        NodeIterator {
            root: self.root.clone(),
            current_node: node.clone(),
            path: path.to_vec()
        }
    }
    pub fn find_node_iter(&self, path: &[Set32]) -> Option<NodeIterator> {
        let node = self.find_node(path);
        if node.is_none() {
            return None;
        }
        Some(self.node_to_iter(node.unwrap(), path))
    }    
    pub fn find_node(&self, path: &[Set32]) -> Option<Rc<RefCell<Node>>> {
        if path.is_empty() {
            return Some(self.root.clone());
        }

        let mut current: Rc<RefCell<Node>> = self.root.clone();
        for key in path {
            let search = &current
                .as_ref()
                .borrow()
                .children
                .binary_search_by(|node| compare_letters(node.as_ref().borrow().letters, *key));
            match search {
                Ok(index) => {
                    let clone = current.as_ref().borrow().children[*index].clone();
                    current = clone;
                }
                Err(_) => return None, // Return None if any key in the path does not match.
            }
        }
        Some(current)
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
                    let node = self.find_node(&path);
                    if node.is_none() {
                        return None;
                    }
                    let node = node.unwrap();
                    let node_children = &node.as_ref().borrow().children;
                    if node_children.is_empty() {
                        return None;
                    }
                    let end_idx: usize = Self::get_end_idx(
                        &node_children
                            .iter()
                            .map(|n| n.as_ref().borrow().letters)
                            .collect::<Vec<Set32>>(),
                        *self.path.last().unwrap(),
                    );
                    let valid_children = node_children
                        .iter()
                        .take(end_idx)
                        .filter(|node| {
                            node.as_ref()
                                .borrow()
                                .letters
                                .intersect(letters_to_exclude)
                                .is_empty()
                        })
                        .map(|node| node.as_ref().borrow().letters)
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
                    let node = self.find_node(&path);
                    if node.is_none() {
                        return None;
                    }
                    let node = node.unwrap();
                    let node_children = &node.as_ref().borrow().children;
                    if node_children.is_empty() {
                        return None;
                    }
                    let start_idx: usize = Self::get_end_idx(
                        &node_children
                            .iter()
                            .map(|n| n.as_ref().borrow().letters)
                            .collect::<Vec<Set32>>(),
                        key,
                    );
                    let valid_children = node_children
                        .iter()
                        .take(start_idx)
                        .filter(|node| {
                            node.as_ref()
                                .borrow()
                                .letters
                                .intersect(letters_to_exclude)
                                .is_empty()
                        })
                        .map(|node| node.as_ref().borrow().letters)
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
                        .as_ref()
                        .borrow()
                        .children
                        .iter()
                        .map(|n| n.as_ref().borrow().letters)
                        .collect::<Vec<Set32>>(),
                    key,
                );
                Some(
                    self.root
                        .as_ref()
                        .borrow()
                        .children
                        .iter()
                        .take(start_idx)
                        .filter(|node| {
                            node.as_ref()
                                .borrow()
                                .letters
                                .intersect(letters_to_exclude)
                                .is_empty()
                        })
                        .map(|node| node.as_ref().borrow().letters)
                        .collect::<Vec<Set32>>(),
                )
            }
        };
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    use std::{cell::RefCell, rc::Rc};

    use crate::{
        create_set32_str, create_set32s_vec,
        globals::{CHAR_TO_INDEX, NUM_LETTERS},
        node::Node,
        set32::Set32,
    };

    use super::NodeIterator;

    fn create_empty_node_iterator() -> NodeIterator {
        let root = Rc::new(RefCell::new(Node::new(0.0, Set32::EMPTY, vec![])));
        NodeIterator {
            current_node: root.clone(),
            path: vec![],
            root,
        }
    }
    #[test]
    fn node_iterator_new_get_children_to_evaluate() {
        let mut root = create_empty_node_iterator();
        let base_keys = NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 2);
        let children = root.new_get_children_to_evaluate(10, NUM_LETTERS, 2);
        assert!(children.is_some());
        assert_eq!(children.unwrap().len(), 351);
        for key in base_keys {
            let new_node = Node::new(0.0, key, vec![]);
            root.insert_node_from_root(&[], new_node);
        }
        let wx = root.find_node_iter(&create_set32s_vec!("wx"));
        assert!(wx.is_some());
        let children = wx.unwrap().new_get_children_to_evaluate(10, NUM_LETTERS, 2);
        assert!(children.is_some());
        assert_eq!(children.unwrap().len(), 3);
        // let children = root.get_children_to_evaluate(create_set32_str!("ab"));
        // assert!(children.is_some());
        // assert_eq!(children.unwrap().len(), 300); // 351 - (26 + 25), all A's and B's
    }

    #[test]
    fn node_iterator_insert_node() {
        let mut root = create_empty_node_iterator();
        let base_keys = NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS, 2);
        for key in base_keys {
            let new_node = Node::new(0.0, key, vec![]);
            root.insert_node_from_root(&[], new_node);
        }
        assert_eq!(root.root.as_ref().borrow().children.len(), 351);
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
            .as_ref()
            .borrow_mut()
            .children
            .push(Rc::new(RefCell::new(a)));
        let found_node = iter.find_node(&path);
        assert!(found_node.is_some());
        assert_eq!(found_node.unwrap().as_ref().borrow().score, 2.0);

        let b = Node {
            score: 2.1,
            letters: create_set32_str!("b"),
            children: vec![],
        };
        iter.root.as_ref().borrow().children[0]
            .as_ref()
            .borrow_mut()
            .children
            .push(Rc::new(RefCell::new(b)));
        let path = vec![create_set32_str!("a"), create_set32_str!("b")];
        assert!(iter.find_node(&path).is_some());
        assert_eq!(iter.find_node(&path).unwrap().as_ref().borrow().score, 2.1);

        let path = vec![
            create_set32_str!("a"),
            create_set32_str!("b"),
            create_set32_str!("c"),
        ];
        assert!(iter.find_node(&path).is_none());

        let yz = Rc::new(RefCell::new(Node::new(
            1.3,
            create_set32_str!("yz"),
            vec![],
        )));
        let ef = Rc::new(RefCell::new(Node::new(
            1.01,
            create_set32_str!("ef"),
            vec![],
        )));
        let ab = Rc::new(RefCell::new(Node::new(
            1.01,
            create_set32_str!("ab"),
            vec![],
        )));
        iter.root
            .as_ref()
            .borrow_mut()
            .children
            .extend([yz, ef, ab]);
        assert!(iter.find_node(&vec![create_set32_str!("yz")]).is_some());
        assert!(
            iter.find_node(&vec![create_set32_str!("yz")])
                .unwrap()
                .as_ref()
                .borrow()
                .score
                == 1.3
        );
        assert!(iter.find_node(&vec![create_set32_str!("ef")]).is_some());
        assert!(iter.find_node(&vec![create_set32_str!("ab")]).is_some());
    }

    #[test]
    fn node_iterator_get_children_to_evaluate() {
        let mut root = create_empty_node_iterator();
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

        // let children =
        //     root.get_children_to_evaluate(&create_set32s_vec!("ab"), create_set32_str!("ef"));
        // assert!(children.is_none());
        // root.insert_node(
        //     &create_set32s_vec!("ef"),
        //     Node::new(1.2, create_set32_str!("yz"), vec![]),
        // );
        // root.insert_node(
        //     &create_set32s_vec!("ab"),
        //     Node::new(1.8, create_set32_str!("yz"), vec![]),
        // );
        // let children =
        //     root.get_children_to_evaluate(&create_set32s_vec!("ab"), create_set32_str!("ef"));
        // assert!(children.is_some());
        // assert_eq!(children.unwrap().len(), 1);
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
        // // 27 choose 2 + 27 choose 3 + 27 choose 4
        // assert!(NodeIterator::get_keys_to_evaluate(10, NUM_LETTERS).len() > 100000 );
    }

    // #[test]
    // fn test_with_exclusions() {
    //     let letter_start = 23;
    //     let letters_to_exclude = vec![24, 26];
    //     let expected = vec![[23, 25].iter().cloned().collect()];
    //     let result = get_keys_to_evaluate(2, letter_start, letters_to_exclude);
    //     assert_eq!(result, expected);
    // }

    // #[test]
    // fn test_letter_start_at_boundary() {
    //     let letter_start = 25;
    //     let letters_to_exclude = vec![];
    //     let expected = vec![[25, 26].iter().cloned().collect()];
    //     let result = get_keys_to_evaluate(2, letter_start, letters_to_exclude);
    //     assert_eq!(result, expected);
    // }

    // #[test]
    // fn test_all_excluded_except_one() {
    //     let letter_start = 23;
    //     let letters_to_exclude = vec![24, 25];
    //     let expected = vec![[23, 26].iter().cloned().collect()];
    //     let result = get_keys_to_evaluate(2, letter_start, letters_to_exclude);
    //     assert_eq!(result, expected);
    // }
}

// impl TreeIterator {
//     fn is_child_valid(&self, pot_child_index: usize) -> bool {
//         let byte_index = bits_to_byte_index(pot_child_index);
//         let bit_index = 7 - pot_child_index % 8;
//         let bit_check = 1 << bit_index;
//         return self.tree.borrow_mut().layers[self.layer_index].nodes
//             [self.ptr + U16_SIZE + byte_index]
//             & bit_check
//             == bit_check;
//     }

//     pub fn get_all_valid_children(&self) -> Vec<Set32> {
//         let mut res: Vec<Set32> = Vec::new();
//         for i in 0..self.potential_children.len() {
//             if self.is_child_valid(i) {
//                 res.push(self.potential_children[i]);
//             }
//         }
//         res
//     }

//     /// Child must exist otherwise this will return all valid children
//     pub fn get_subset_valid_children(&self, child: Set32) -> Vec<Set32> {
//         let mut res: Vec<Set32> = Vec::new();
//         let tree = self.tree.borrow_mut();
//         for i in 0..self.potential_children.len() {
//             if self.potential_children[i] == child {
//                 break;
//             }
//             let byte = bits_to_byte_index(i);
//             let bit_check = 1 << (7 - i % 8);
//             if (tree.layers[self.layer_index].nodes[self.ptr + U16_SIZE + byte] & bit_check)
//                 == bit_check
//                 && self.potential_children[i].intersect(child) == Set32::EMPTY
//             {
//                 res.push(self.potential_children[i]);
//             }
//         }
//         res
//     }

//     // Used to get aunt for child
//     pub fn find_sibling(&self, sibling: Set32) -> Option<TreeIterator> {
//         if self.parent.is_none() {
//             panic!("TreeIterator has no sibling.");
//         }
//         self.parent
//             .as_ref()
//             .unwrap()
//             .as_ref()
//             .borrow()
//             .find_child(sibling)
//     }

//     pub fn find_child(&self, child: Set32) -> Option<TreeIterator> {
//         let mut num_bytes = 0;
//         let mut child_children: Vec<Set32> = vec![];
//         let all_valid_children = self.get_all_valid_children();
//         for (index, ch) in all_valid_children.iter().enumerate() {
//             child_children =
//                 Self::remove_children_sharing_letters(&all_valid_children[0..=index], *ch);
//             if ch == &child {
//                 break;
//             }
//             if child_children.len() > 0 {
//                 num_bytes += util::bits_to_num_bytes(child_children.len()) + U16_SIZE;
//             }
//         }
//         if child_children.is_empty() {
//             return None;
//         }
//         let byte_index = self.get_ptr_to_next_layer() + num_bytes;
//         let mut path = self.path.clone();
//         path.push(child);
//         Some(TreeIterator {
//             layer_index: self.layer_index + 1,
//             parent: Some(Rc::new(RefCell::new(self.clone()))),
//             path,
//             ptr: byte_index,
//             tree: Rc::clone(&self.tree),
//             potential_children: child_children,
//         })
//     }

//     pub fn find_node(&self, path: &Vec<Set32>) -> Option<TreeIterator> {
//         let mut iter = self.clone();
//         for node in path {
//             let mut num_bytes = 0;
//             let mut child_children: Vec<Set32> = vec![];
//             let all_valid_children = iter.get_all_valid_children();
//             if all_valid_children.is_empty() {
//                 return None;
//             }
//             let mut child_found = false;
//             for (index, child) in all_valid_children.iter().enumerate() {
//                 child_children =
//                     Self::remove_children_sharing_letters(&all_valid_children[0..=index], *child);
//                 if child == node {
//                     child_found = true;
//                     break;
//                 }
//                 if child_children.len() > 0 {
//                     num_bytes += util::bits_to_num_bytes(child_children.len()) + U16_SIZE;
//                 }
//             }
//             if !child_found {
//                 return None;
//             }
//             let mut path = iter.path.clone();
//             path.push(*node);
//             iter = TreeIterator {
//                 layer_index: iter.layer_index + 1,
//                 parent: Some(Rc::new(RefCell::new(iter.clone()))),
//                 path,
//                 ptr: iter.get_ptr_to_next_layer() + num_bytes,
//                 tree: Rc::clone(&iter.tree),
//                 potential_children: child_children,
//             }
//         }
//         Some(iter)
//     }

//     pub fn children_cmp(probe: &Set32, target: &Set32) -> Ordering {
//         if probe == target {
//             return Ordering::Equal;
//         }
//         let ones_probe = probe.ones_indices();
//         let ones_target = target.ones_indices();
//         if ones_probe.len() > ones_target.len() {
//             return Ordering::Greater;
//         } else if ones_probe.len() < ones_target.len() {
//             return Ordering::Less;
//         }
//         if probe > target {
//             return Ordering::Greater;
//         }
//         return Ordering::Less;
//     }

//     // Once we determined something is valid, we flip bit
//     pub fn set_child_valid(&mut self, child: Set32) {
//         let num_bits = self
//             .potential_children
//             .binary_search_by(|probe: &Set32| TreeIterator::children_cmp(&probe, &child))
//             .expect("child to be found");
//         let ptr = self.ptr;
//         let mut tree = self.tree.borrow_mut();
//         let index = util::bits_to_byte_index(num_bits);
//         let bit_index = 7 - num_bits % 8;
//         tree.layers[self.layer_index].nodes[index + U16_SIZE + ptr] |= 1 << bit_index;
//     }
//     pub fn set_ptr_to_next_layer_ptr(&mut self, next_layer_ptr: usize) {
//         let mut tree: std::cell::RefMut<'_, BinAryTree> = self.tree.borrow_mut();
//         let mut last_bp = *tree.layers[self.layer_index].base_ptrs.last().unwrap();
//         let sub = next_layer_ptr - last_bp.1;
//         if sub > u16::MAX as usize {
//             last_bp = (self.ptr, next_layer_ptr);
//             tree.layers[self.layer_index].base_ptrs.push(last_bp);
//         }
//         let u16_ptr: u16 = (next_layer_ptr - last_bp.1) as u16;
//         let bytes = u16_ptr.to_ne_bytes();
//         tree.layers[self.layer_index].nodes[self.ptr] = bytes[0];
//         tree.layers[self.layer_index].nodes[self.ptr + 1] = bytes[1];
//     }
//     fn get_ptr_to_next_layer(&self) -> usize {
//         let tree = self.tree.as_ref().borrow();
//         let base_ptr = tree.layers[self.layer_index]
//             .base_ptrs
//             .binary_search_by(|probe: &(usize, usize)| probe.0.cmp(&self.ptr));
//         let bp_index = base_ptr.unwrap_or_else(|op| op - 1);
//         let bp = &tree.layers[self.layer_index].base_ptrs[bp_index];
//         let bytes = [
//             tree.layers[self.layer_index].nodes[self.ptr],
//             tree.layers[self.layer_index].nodes[self.ptr + 1],
//         ];
//         return bp.1 + u16::from_ne_bytes(bytes) as usize;
//     }

//     fn get_all_key_combinations(k: usize) -> Vec<Vec<u32>> {
//         let mut res: Vec<Vec<u32>> = Vec::new();
//         for i in 2..=k {
//             let numbers: Vec<u32> = (0..=26).collect();
//             let mut combinations = numbers.into_iter().combinations(i).collect::<Vec<_>>();
//             combinations.reverse();
//             res.append(&mut combinations);
//         }
//         res
//     }

//     fn combinations_to_set32s(combinations: &Vec<Vec<u32>>) -> Vec<Set32> {
//         let mut res: Vec<Set32> = Vec::new();
//         for vec in combinations.iter() {
//             let mut s = Set32::EMPTY;
//             for val in vec {
//                 s = s.add(26 - *val);
//             }
//             res.push(s);
//         }
//         res
//     }
//     pub fn set_base(&mut self, num_keys: usize) {
//         if num_keys < 2 {
//             panic!("Creating a base under 2 keys has no effect");
//         }
//         let mut tree_mut = self.tree.borrow_mut(); // Directly borrow mutably from self.tree
//         if tree_mut.borrow().layers.len() > 0 {
//             panic!("Base layer already set");
//         }
//         let mut base_layer = TreeLayer::default();
//         let values = Self::get_all_key_combinations(num_keys);
//         self.potential_children = Self::combinations_to_set32s(&values);
//         base_layer.nodes = vec![0; self.potential_children.len() + U16_SIZE];
//         tree_mut.layers.push(base_layer);
//     }
//     // pub fn find_child(&self, key: Set32) {}
//     fn remove_children_sharing_letters(
//         potential_valid_children: &[Set32],
//         child: Set32,
//     ) -> Vec<Set32> {
//         let mut result: Vec<Set32> = Vec::new();
//         for ch in potential_valid_children {
//             if ch.intersect(child) == Set32::EMPTY {
//                 result.push(*ch);
//             }
//         }
//         result
//     }

//     /// Assumes being called in "sequential" order. No valid children > child
//     pub fn create_children_block(&self, child: Set32) -> Option<TreeIterator> {
//         let subset_valid_children = self.get_subset_valid_children(child);
//         if subset_valid_children.is_empty() {
//             return None;
//         }
//         let mut tree_mut = self.tree.borrow_mut();
//         if tree_mut.layers.len() == self.layer_index + 1 {
//             tree_mut.layers.push(TreeLayer::default());
//         }
//         let num_bytes = util::bits_to_num_bytes(subset_valid_children.len());
//         let next_layer_len = tree_mut.layers[self.layer_index + 1].nodes.len();
//         tree_mut.layers[self.layer_index + 1]
//             .nodes
//             .resize(next_layer_len + U16_SIZE + num_bytes, 0);
//         let mut new_path: Vec<Set32> = self.path.clone();
//         new_path.push(child);
//         let result = TreeIterator {
//             parent: Some(Rc::new(RefCell::new(self.clone()))),
//             layer_index: self.layer_index + 1,
//             ptr: next_layer_len,
//             tree: Rc::clone(&self.tree.borrow()),
//             path: new_path,
//             potential_children: subset_valid_children,
//         };
//         Some(result)
//     }
// }

// #[cfg(test)]
// pub mod tests {
//     use crate::{
//         bin_ary_tree::BinAryTree, create_set32, create_set32_str, globals::CHAR_TO_INDEX,
//         set32::Set32, tree_iterator::TreeIterator,
//     };
//     use std::{cell::RefCell, cmp::Ordering, rc::Rc};

//     #[test]
//     fn tree_iterator_find_node() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(2);
//         let z_apos = create_set32_str!("z\'");
//         let xy = create_set32_str!("xy");
//         it.set_child_valid(z_apos);
//         assert!(it.find_node(&vec![xy.clone()]).is_none());
//         assert!(it.find_node(&vec![z_apos.clone()]).is_some());
//         it.set_child_valid(xy);
//         let mut next = it.create_children_block(xy).unwrap();
//         assert!(it.find_node(&vec![xy.clone(), z_apos.clone()]).is_none());
//         next.set_child_valid(z_apos);
//         assert!(it.find_node(&vec![xy.clone(), z_apos.clone()]).is_some());
//     }

//     #[test]
//     fn tree_iterator_set_ptr_to_next_layer2() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(2);
//         let z_apostrophe = create_set32!('z', '\'');
//         it.set_child_valid(z_apostrophe);
//         let valid_children = it.get_all_valid_children();
//         assert_eq!(valid_children.len(), 1);
//         let uz = create_set32!('u', 'z');
//         let uv = create_set32!('u', 'v');
//         it.set_child_valid(uz);
//         it.set_child_valid(uv);
//         let valid_children = it.get_all_valid_children();
//         assert_eq!(valid_children.len(), 3);
//     }

//     #[test]
//     fn tree_iterator_get_valid_children() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(2);
//         let z_apostrophe = create_set32!('z', '\'');
//         it.set_child_valid(z_apostrophe);
//         let valid_children = it.get_all_valid_children();
//         assert_eq!(valid_children.len(), 1);
//         let uz = create_set32!('u', 'z');
//         let uv = create_set32!('u', 'v');
//         it.set_child_valid(uz);
//         it.set_child_valid(uv);
//         let valid_children = it.get_all_valid_children();
//         assert_eq!(valid_children.len(), 3);
//     }

//     #[test]
//     fn tree_iterator_get_potential_children_for_child() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(2);
//         let z_apostrophe = create_set32!('\'', 'z');
//         let xy = create_set32!('x', 'y');
//         let uz = create_set32!('u', 'z');
//         let uv = create_set32!('u', 'v');
//         it.set_child_valid(z_apostrophe);
//         it.set_child_valid(xy);
//         it.set_child_valid(uz);
//         it.set_child_valid(uv);
//         assert_eq!(it.get_subset_valid_children(uv).len(), 2);
//         assert_eq!(it.get_subset_valid_children(xy).len(), 1);
//         assert_eq!(it.get_subset_valid_children(z_apostrophe).len(), 0);
//     }

//     #[test]
//     fn tree_iterator_find_sibling() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(2);
//         let z_apostrophe = create_set32!('\'', 'z');
//         let xy = create_set32!('x', 'y');
//         it.create_children_block(z_apostrophe);
//         it.set_child_valid(z_apostrophe);
//         let xy_it = it.create_children_block(xy).unwrap();
//         it.set_child_valid(xy);
//         let xy_sibling = xy_it.find_sibling(z_apostrophe);
//         // No z' child should be create since it has no children
//         assert!(xy_sibling.is_none());
//         let uv = create_set32!('u', 'v');
//         let uv_it = it.create_children_block(uv).unwrap();
//         let uv_sibling = uv_it.find_sibling(xy).unwrap();
//         assert_eq!(uv_sibling.path[0], xy);
//         assert_eq!(uv_sibling.ptr, 0);
//     }
//     #[test]
//     fn tree_iterator_set_get_ptr() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(2);
//         it.set_ptr_to_next_layer_ptr(2097);
//         assert_eq!(it.get_ptr_to_next_layer(), 2097);
//     }

//     #[test]
//     fn tree_iterator_set_child_valid() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };

//         // | Z' Y' YZ X' XZ XY W' WZ | WY
//         let z_apostrophe = create_set32_str!("z\'");
//         let w_z = create_set32_str!("wz");
//         let w_y = create_set32_str!("wy");
//         let y_z_apostrophe = create_set32_str!("yz\'");
//         it.set_base(5);
//         assert_eq!(it.tree.borrow().layers[0].nodes[2], 0);
//         it.set_child_valid(z_apostrophe);
//         assert_eq!(it.tree.borrow().layers[0].nodes[2], 128);
//         it.set_child_valid(w_z);
//         assert_eq!(it.tree.borrow().layers[0].nodes[2], 129); // 2^7 + 1
//         it.set_child_valid(w_y);
//         assert_eq!(it.tree.borrow().layers[0].nodes[3], 128); // 2^7
//                                                               // Bit number: 27 choose 2 = 351. 351/8 = 43 bytes, bit # 7
//         it.set_child_valid(y_z_apostrophe);
//         assert_eq!(it.tree.borrow().layers[0].nodes[45], 1);
//     }

//     #[test]
//     fn tree_iterator_children_cmp() {
//         let z_apostrophe = create_set32!('z', '\'');
//         let x_apostrophe = create_set32!('x', '\'');
//         let y_z_apostrophe = create_set32!('y', 'z', '\'');

//         assert_eq!(
//             TreeIterator::children_cmp(&z_apostrophe, &x_apostrophe),
//             Ordering::Less
//         );
//         assert_eq!(
//             TreeIterator::children_cmp(&z_apostrophe, &y_z_apostrophe),
//             Ordering::Less
//         );
//         assert_eq!(
//             TreeIterator::children_cmp(&y_z_apostrophe, &z_apostrophe),
//             Ordering::Greater
//         );
//         assert_eq!(
//             TreeIterator::children_cmp(&y_z_apostrophe, &y_z_apostrophe),
//             Ordering::Equal
//         );
//     }

//     #[test]
//     fn tree_iterator_create_children_block_none() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: vec![],
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(5);
//         let z_apostrophe = Set32::singleton(26).add(25);
//         let res = it.create_children_block(z_apostrophe);
//         assert!(res.is_none());
//     }
//     #[test]
//     fn tree_iterator_create_children_block_one_child() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(5);
//         let x_apostrophe = create_set32!('x', '\'');
//         let yz = create_set32!('y', 'z');
//         it.set_child_valid(yz);

//         // let y_apostrophe =
//         // Set32::singleton(CHAR_TO_INDEX[&'y'] as u32).add(CHAR_TO_INDEX[&'\''] as u32);

//         // let all_children: Vec<Set32> = vec![y_apostrophe, y_z, x_apostrophe];
//         // let child = x_apostrophe;
//         // let vc = TreeIterator::remove_children_sharing_letters(&all_children, child);
//         let res = it.create_children_block(x_apostrophe);
//         let res_it = res.unwrap();
//         assert_eq!(res_it.layer_index, 1);
//         assert_eq!(res_it.potential_children.len(), 1);
//         assert_eq!(res_it.potential_children[0], yz);
//         assert!(res_it.parent.is_some());
//         assert_eq!(it.tree.borrow().layers.len(), 2);
//     }
//     #[test]
//     fn tree_iterator_create_children_block_updates_bp() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(5);
//         let xy = create_set32!('x', 'y');

//         // We need one valid child so that there will be an xy children block
//         let z_apos = create_set32!('z', '\'');
//         it.set_child_valid(z_apos);

//         // creates 3 bytes an iteration, u16 max is 65535.
//         // 21845 iterations (u16 max / 3) will create 65535 bytes (index 65534, one less than u16 max)
//         // to create a new base pointer, we must go over u16 max, so we + 1
//         let mut next_iter: Option<TreeIterator> = None;

//         for i in 0..u16::MAX / 3 + 1 {
//             next_iter = it.create_children_block(xy);
//             println!("{}", i);
//         }
//         assert_eq!(
//             next_iter.unwrap().tree.borrow().layers[0].base_ptrs.len(),
//             1
//         );
//         next_iter = it.create_children_block(xy);
//         let next_iter = next_iter.unwrap();
//         it.set_ptr_to_next_layer_ptr(next_iter.ptr);
//         assert_eq!(next_iter.tree.borrow().layers[0].base_ptrs.len(), 2);
//         assert_eq!(next_iter.tree.borrow().layers[0].base_ptrs[1].1, 65535 + 3);
//     }
//     #[test]
//     fn tree_iterator_get_all_key_combinations_5_keys() {
//         let result = TreeIterator::get_all_key_combinations(5);
//         // 27 choose 5 + 27 choose 4 + 27 choose 3 + 27 choose 2
//         assert_eq!(result.len(), 101556);
//     }
//     #[test]
//     fn tree_iterator_get_all_key_combinations_is_0_when_1() {
//         let result = TreeIterator::get_all_key_combinations(1);
//         assert_eq!(result.len(), 0);
//     }
//     #[test]
//     fn tree_iterator_combinations_to_sets() {
//         let result =
//             TreeIterator::combinations_to_set32s(&TreeIterator::get_all_key_combinations(3));
//         let z_apostrophe = create_set32_str!("z\'");
//         assert_eq!(result.len(), 3276);
//         assert_eq!(result[0], z_apostrophe);
//     }
//     #[test]
//     fn tree_iterator_set_base_correct_values() {
//         let mut it = TreeIterator {
//             tree: Rc::new(RefCell::new(BinAryTree::default())),
//             potential_children: Vec::new(),
//             layer_index: 0,
//             parent: None,
//             path: Vec::new(),
//             ptr: 0,
//         };
//         it.set_base(5);
//         assert_eq!(
//             it.tree.borrow().layers[0].nodes.len(),
//             TreeIterator::get_all_key_combinations(5).len() + 2
//         );
//         assert_eq!(
//             it.potential_children.len(),
//             TreeIterator::get_all_key_combinations(5).len()
//         );
//         let z_apostrophe = create_set32_str!("z\'");
//         let y_apostrophe = create_set32_str!("y\'");
//         assert_eq!(it.potential_children[0], z_apostrophe);
//         assert_eq!(it.potential_children[1], y_apostrophe);
//     }
//     #[test]
//     fn tree_iterator_remove_same_letter_children_is_zero_when_invalid_children() {
//         let pot_valid_children: Vec<Set32> = vec![Set32::singleton(3)];
//         let child = Set32::singleton(3);
//         assert_eq!(
//             TreeIterator::remove_children_sharing_letters(&pot_valid_children, child).len(),
//             0
//         );
//     }
//     #[test]
//     fn tree_iterator_remove_same_letter_children_is_zero_when_valid_children() {
//         let pot_valid_children: Vec<Set32> = vec![
//             Set32::fill_from_left(2),
//             Set32::singleton(4),
//             Set32::fill(2),
//         ];
//         let child = Set32::singleton(3);
//         assert_eq!(
//             TreeIterator::remove_children_sharing_letters(&pot_valid_children, child).len(),
//             3
//         );
//     }
// }
/*
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
     */
// pub fn get_keys_to_evaluate2(
//     path: &Vec<Set32>,
//     solution_num_keys: usize,
// ) -> Vec<BTreeSet<u32>> {
//     let (max_key_len, letters_to_exclude, letter_start) = match path.is_empty() {
//         true => (NUM_LETTERS - solution_num_keys + 1, Vec::new(), 0),
//         false => {
//             let mut letters_to_exclude = Vec::new();
//             for key in path {
//                 for one in key.ones_indices() {
//                     letters_to_exclude.push(one);
//                 }
//             }
//             let last_key_ones = path.last().unwrap().ones_indices();
//             (last_key_ones.len(), letters_to_exclude, last_key_ones[0])
//         }
//     };
//     let mut combinations: Vec<BTreeSet<u32>> = Vec::new();
//     // Define the valid range of letters, excluding any in `letters_to_exclude`.
//     let valid_letters: Vec<u32> = (0..=NUM_LETTERS as u32)
//         .filter(|&k| !letters_to_exclude.contains(&k))
//         .collect();

//     // Generate combinations starting from size 2 up to `num_keys`
//     for i in 2..max_key_len {
//         let mut combos = valid_letters
//             .iter()
//             .combinations(i)
//             .map(|combo| {
//                 let mut set = BTreeSet::new();
//                 set.insert(letter_start);
//                 for &item in combo.iter() {
//                     set.insert(*item);
//                 }
//                 set
//             })
//             .collect::<Vec<BTreeSet<u32>>>();
//         combinations.append(&mut combos);
//     }
//     let mut combos = valid_letters
//         .iter()
//         .combinations(max_key_len)
//         .map(|combo| {
//             let mut set = BTreeSet::new();
//             set.insert(letter_start);
//             for &item in combo.iter() {
//                 set.insert(*item);
//             }
//             set
//         })
//         .collect::<Vec<BTreeSet<u32>>>();
//     combinations.append(&mut combos);
//     //Special rule at max_key_len,

//     combinations
// }
