use std::{borrow::Borrow, cell::RefCell, cmp::Ordering, rc::Rc};

use itertools::Itertools;

use crate::{
    bin_ary_tree::BinAryTree,
    globals::U16_SIZE,
    set32::Set32,
    tree_layer::TreeLayer,
    util::{self, bits_to_byte_index},
};

#[derive(Clone)]
pub struct TreeIterator {
    pub ptr: usize,
    pub path: Vec<Set32>,
    pub layer_index: usize,
    pub tree: Rc<RefCell<BinAryTree>>,
    pub potential_children: Vec<Set32>,
    // pub valid_children: Vec<Set32>,
    pub parent: Option<Rc<RefCell<TreeIterator>>>,
}

impl TreeIterator {
    fn is_child_valid(&self, pot_child_index: usize) -> bool {
        let byte_index = bits_to_byte_index(pot_child_index);
        let bit_index = 7 - pot_child_index % 8;
        let bit_check = 1 << bit_index;
        return self.tree.borrow_mut().layers[self.layer_index].nodes
            [self.ptr + U16_SIZE + byte_index]
            & bit_check
            == bit_check;
    }

    pub fn get_all_valid_children(&self) -> Vec<Set32> {
        let mut res: Vec<Set32> = Vec::new();
        for i in 0..self.potential_children.len() {
            if self.is_child_valid(i) {
                res.push(self.potential_children[i]);
            }
        }
        res
    }

    /// Child must exist otherwise this will return all valid children
    pub fn get_subset_valid_children(&self, child: Set32) -> Vec<Set32> {
        let mut res: Vec<Set32> = Vec::new();
        let tree = self.tree.borrow_mut();
        for i in 0..self.potential_children.len() {
            if self.potential_children[i] == child {
                break;
            }
            let byte = bits_to_byte_index(i);
            let bit_check = 1 << (7 - i % 8);
            if (tree.layers[self.layer_index].nodes[self.ptr + U16_SIZE + byte] & bit_check)
                == bit_check
                && self.potential_children[i].intersect(child) == Set32::EMPTY
            {
                res.push(self.potential_children[i]);
            }
        }
        res
    }

    // Used to get aunt for child
    pub fn find_sibling(&self, sibling: Set32) -> Option<TreeIterator> {
        if self.parent.is_none() {
            panic!("TreeIterator has no sibling.");
        }
        self.parent
            .as_ref()
            .unwrap()
            .as_ref()
            .borrow()
            .find_child(sibling)
    }

    pub fn find_child(&self, child: Set32) -> Option<TreeIterator> {
        let mut num_bytes = 0;
        let mut child_children: Vec<Set32> = vec![];
        let all_valid_children = self.get_all_valid_children();
        for (index, ch) in all_valid_children.iter().enumerate() {
            child_children =
                Self::remove_children_sharing_letters(&all_valid_children[0..=index], *ch);
            if ch == &child {
                break;
            }
            if child_children.len() > 0 {
                num_bytes += util::bits_to_num_bytes(child_children.len()) + U16_SIZE;
            }
        }
        if child_children.is_empty() {
            return None;
        }
        let byte_index = self.get_ptr_to_next_layer() + num_bytes;
        let mut path = self.path.clone();
        path.push(child);
        Some(TreeIterator {
            layer_index: self.layer_index + 1,
            parent: Some(Rc::new(RefCell::new(self.clone()))),
            path,
            ptr: byte_index,
            tree: Rc::clone(&self.tree),
            potential_children: child_children,
        })
    }

    pub fn find_node(&self, path: &Vec<Set32>) -> Option<TreeIterator> {
        let mut iter = self.clone();
        for node in path {
            let mut num_bytes = 0;
            let mut child_children: Vec<Set32> = vec![];
            let all_valid_children = iter.get_all_valid_children();
            if all_valid_children.is_empty() {
                return None;
            }
            let mut child_found = false;
            for (index, child) in all_valid_children.iter().enumerate() {
                child_children =
                    Self::remove_children_sharing_letters(&all_valid_children[0..=index], *child);
                if child == node {
                    child_found = true;
                    break;
                }
                if child_children.len() > 0 {
                    num_bytes += util::bits_to_num_bytes(child_children.len()) + U16_SIZE;
                }
            }
            if !child_found {
                return None;
            }
            let mut path = iter.path.clone();
            path.push(*node);
            iter = TreeIterator {
                layer_index: iter.layer_index + 1,
                parent: Some(Rc::new(RefCell::new(iter.clone()))),
                path,
                ptr: iter.get_ptr_to_next_layer() + num_bytes,
                tree: Rc::clone(&iter.tree),
                potential_children: child_children,
            }
        }
        Some(iter)
    }

    pub fn children_cmp(probe: &Set32, target: &Set32) -> Ordering {
        if probe == target {
            return Ordering::Equal;
        }
        let ones_probe = probe.ones_indices();
        let ones_target = target.ones_indices();
        if ones_probe.len() > ones_target.len() {
            return Ordering::Greater;
        } else if ones_probe.len() < ones_target.len() {
            return Ordering::Less;
        }
        if probe > target {
            return Ordering::Greater;
        }
        return Ordering::Less;
    }

    // Once we determined something is valid, we flip bit
    pub fn set_child_valid(&mut self, child: Set32) {
        let num_bits = self
            .potential_children
            .binary_search_by(|probe: &Set32| TreeIterator::children_cmp(&probe, &child))
            .expect("child to be found");
        let ptr = self.ptr;
        let mut tree = self.tree.borrow_mut();
        let index = util::bits_to_byte_index(num_bits);
        let bit_index = 7 - num_bits % 8;
        tree.layers[self.layer_index].nodes[index + U16_SIZE + ptr] |= 1 << bit_index;
    }
    pub fn set_ptr_to_next_layer_ptr(&mut self, next_layer_ptr: usize) {
        let mut tree: std::cell::RefMut<'_, BinAryTree> = self.tree.borrow_mut();
        let mut last_bp = *tree.layers[self.layer_index].base_ptrs.last().unwrap();
        let sub = next_layer_ptr - last_bp.1;
        if sub > u16::MAX as usize {
            last_bp = (self.ptr, next_layer_ptr);
            tree.layers[self.layer_index].base_ptrs.push(last_bp);
        }
        let u16_ptr: u16 = (next_layer_ptr - last_bp.1) as u16;
        let bytes = u16_ptr.to_ne_bytes();
        tree.layers[self.layer_index].nodes[self.ptr] = bytes[0];
        tree.layers[self.layer_index].nodes[self.ptr + 1] = bytes[1];
    }
    fn get_ptr_to_next_layer(&self) -> usize {
        let tree = self.tree.as_ref().borrow();
        let base_ptr = tree.layers[self.layer_index]
            .base_ptrs
            .binary_search_by(|probe: &(usize, usize)| probe.0.cmp(&self.ptr));
        let bp_index = base_ptr.unwrap_or_else(|op| op - 1);
        let bp = &tree.layers[self.layer_index].base_ptrs[bp_index];
        let bytes = [
            tree.layers[self.layer_index].nodes[self.ptr],
            tree.layers[self.layer_index].nodes[self.ptr + 1],
        ];
        return bp.1 + u16::from_ne_bytes(bytes) as usize;
    }

    fn get_all_key_combinations(k: usize) -> Vec<Vec<u32>> {
        let mut res: Vec<Vec<u32>> = Vec::new();
        for i in 2..=k {
            let numbers: Vec<u32> = (0..=26).collect();
            let mut combinations = numbers.into_iter().combinations(i).collect::<Vec<_>>();
            combinations.reverse();
            res.append(&mut combinations);
        }
        res
    }

    fn combinations_to_set32s(combinations: &Vec<Vec<u32>>) -> Vec<Set32> {
        let mut res: Vec<Set32> = Vec::new();
        for vec in combinations.iter() {
            let mut s = Set32::EMPTY;
            for val in vec {
                s = s.add(26 - *val);
            }
            res.push(s);
        }
        res
    }
    pub fn set_base(&mut self, num_keys: usize) {
        if num_keys < 2 {
            panic!("Creating a base under 2 keys has no effect");
        }
        let mut tree_mut = self.tree.borrow_mut(); // Directly borrow mutably from self.tree
        if tree_mut.borrow().layers.len() > 0 {
            panic!("Base layer already set");
        }
        let mut base_layer = TreeLayer::default();
        let values = Self::get_all_key_combinations(num_keys);
        self.potential_children = Self::combinations_to_set32s(&values);
        base_layer.nodes = vec![0; self.potential_children.len() + U16_SIZE];
        tree_mut.layers.push(base_layer);
    }
    // pub fn find_child(&self, key: Set32) {}
    fn remove_children_sharing_letters(
        potential_valid_children: &[Set32],
        child: Set32,
    ) -> Vec<Set32> {
        let mut result: Vec<Set32> = Vec::new();
        for ch in potential_valid_children {
            if ch.intersect(child) == Set32::EMPTY {
                result.push(*ch);
            }
        }
        result
    }

    /// Assumes being called in "sequential" order. No valid children > child
    pub fn create_children_block(&self, child: Set32) -> Option<TreeIterator> {
        let subset_valid_children = self.get_subset_valid_children(child);
        if subset_valid_children.is_empty() {
            return None;
        }
        let mut tree_mut = self.tree.borrow_mut();
        if tree_mut.layers.len() == self.layer_index + 1 {
            tree_mut.layers.push(TreeLayer::default());
        }
        let num_bytes = util::bits_to_num_bytes(subset_valid_children.len());
        let next_layer_len = tree_mut.layers[self.layer_index + 1].nodes.len();
        tree_mut.layers[self.layer_index + 1]
            .nodes
            .resize(next_layer_len + U16_SIZE + num_bytes, 0);
        let mut new_path: Vec<Set32> = self.path.clone();
        new_path.push(child);
        let result = TreeIterator {
            parent: Some(Rc::new(RefCell::new(self.clone()))),
            layer_index: self.layer_index + 1,
            ptr: next_layer_len,
            tree: Rc::clone(&self.tree.borrow()),
            path: new_path,
            potential_children: subset_valid_children,
        };
        Some(result)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{
        bin_ary_tree::BinAryTree, create_set32, create_set32_str, globals::CHAR_TO_INDEX,
        set32::Set32, tree_iterator::TreeIterator,
    };
    use std::{cell::RefCell, cmp::Ordering, rc::Rc};

    #[test]
    fn tree_iterator_find_node() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(2);
        let z_apos = create_set32_str!("z\'");
        let xy = create_set32_str!("xy");
        it.set_child_valid(z_apos);
        assert!(it.find_node(&vec![xy.clone()]).is_none());
        assert!(it.find_node(&vec![z_apos.clone()]).is_some());
        it.set_child_valid(xy);
        let mut next = it.create_children_block(xy).unwrap();
        assert!(it.find_node(&vec![xy.clone(), z_apos.clone()]).is_none());
        next.set_child_valid(z_apos);
        assert!(it.find_node(&vec![xy.clone(), z_apos.clone()]).is_some());
    }

    #[test]
    fn tree_iterator_set_ptr_to_next_layer2() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(2);
        let z_apostrophe = create_set32!('z', '\'');
        it.set_child_valid(z_apostrophe);
        let valid_children = it.get_all_valid_children();
        assert_eq!(valid_children.len(), 1);
        let uz = create_set32!('u', 'z');
        let uv = create_set32!('u', 'v');
        it.set_child_valid(uz);
        it.set_child_valid(uv);
        let valid_children = it.get_all_valid_children();
        assert_eq!(valid_children.len(), 3);
    }

    #[test]
    fn tree_iterator_get_valid_children() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(2);
        let z_apostrophe = create_set32!('z', '\'');
        it.set_child_valid(z_apostrophe);
        let valid_children = it.get_all_valid_children();
        assert_eq!(valid_children.len(), 1);
        let uz = create_set32!('u', 'z');
        let uv = create_set32!('u', 'v');
        it.set_child_valid(uz);
        it.set_child_valid(uv);
        let valid_children = it.get_all_valid_children();
        assert_eq!(valid_children.len(), 3);
    }

    #[test]
    fn tree_iterator_get_potential_children_for_child() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(2);
        let z_apostrophe = create_set32!('\'', 'z');
        let xy = create_set32!('x', 'y');
        let uz = create_set32!('u', 'z');
        let uv = create_set32!('u', 'v');
        it.set_child_valid(z_apostrophe);
        it.set_child_valid(xy);
        it.set_child_valid(uz);
        it.set_child_valid(uv);
        assert_eq!(it.get_subset_valid_children(uv).len(), 2);
        assert_eq!(it.get_subset_valid_children(xy).len(), 1);
        assert_eq!(it.get_subset_valid_children(z_apostrophe).len(), 0);
    }

    #[test]
    fn tree_iterator_find_sibling() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(2);
        let z_apostrophe = create_set32!('\'', 'z');
        let xy = create_set32!('x', 'y');
        it.create_children_block(z_apostrophe);
        it.set_child_valid(z_apostrophe);
        let xy_it = it.create_children_block(xy).unwrap();
        it.set_child_valid(xy);
        let xy_sibling = xy_it.find_sibling(z_apostrophe);
        // No z' child should be create since it has no children
        assert!(xy_sibling.is_none());
        let uv = create_set32!('u', 'v');
        let uv_it = it.create_children_block(uv).unwrap();
        let uv_sibling = uv_it.find_sibling(xy).unwrap();
        assert_eq!(uv_sibling.path[0], xy);
        assert_eq!(uv_sibling.ptr, 0);
    }
    #[test]
    fn tree_iterator_set_get_ptr() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(2);
        it.set_ptr_to_next_layer_ptr(2097);
        assert_eq!(it.get_ptr_to_next_layer(), 2097);
    }

    #[test]
    fn tree_iterator_set_child_valid() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };

        // | Z' Y' YZ X' XZ XY W' WZ | WY
        let z_apostrophe = create_set32_str!("z\'");
        let w_z = create_set32_str!("wz");
        let w_y = create_set32_str!("wy");
        let y_z_apostrophe = create_set32_str!("yz\'");
        it.set_base(5);
        assert_eq!(it.tree.borrow().layers[0].nodes[2], 0);
        it.set_child_valid(z_apostrophe);
        assert_eq!(it.tree.borrow().layers[0].nodes[2], 128);
        it.set_child_valid(w_z);
        assert_eq!(it.tree.borrow().layers[0].nodes[2], 129); // 2^7 + 1
        it.set_child_valid(w_y);
        assert_eq!(it.tree.borrow().layers[0].nodes[3], 128); // 2^7
                                                              // Bit number: 27 choose 2 = 351. 351/8 = 43 bytes, bit # 7
        it.set_child_valid(y_z_apostrophe);
        assert_eq!(it.tree.borrow().layers[0].nodes[45], 1);
    }

    #[test]
    fn tree_iterator_children_cmp() {
        let z_apostrophe = create_set32!('z', '\'');
        let x_apostrophe = create_set32!('x', '\'');
        let y_z_apostrophe = create_set32!('y', 'z', '\'');

        assert_eq!(
            TreeIterator::children_cmp(&z_apostrophe, &x_apostrophe),
            Ordering::Less
        );
        assert_eq!(
            TreeIterator::children_cmp(&z_apostrophe, &y_z_apostrophe),
            Ordering::Less
        );
        assert_eq!(
            TreeIterator::children_cmp(&y_z_apostrophe, &z_apostrophe),
            Ordering::Greater
        );
        assert_eq!(
            TreeIterator::children_cmp(&y_z_apostrophe, &y_z_apostrophe),
            Ordering::Equal
        );
    }

    #[test]
    fn tree_iterator_create_children_block_none() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: vec![],
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(5);
        let z_apostrophe = Set32::singleton(26).add(25);
        let res = it.create_children_block(z_apostrophe);
        assert!(res.is_none());
    }
    #[test]
    fn tree_iterator_create_children_block_one_child() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(5);
        let x_apostrophe = create_set32!('x', '\'');
        let yz = create_set32!('y', 'z');
        it.set_child_valid(yz);

        // let y_apostrophe =
        // Set32::singleton(CHAR_TO_INDEX[&'y'] as u32).add(CHAR_TO_INDEX[&'\''] as u32);

        // let all_children: Vec<Set32> = vec![y_apostrophe, y_z, x_apostrophe];
        // let child = x_apostrophe;
        // let vc = TreeIterator::remove_children_sharing_letters(&all_children, child);
        let res = it.create_children_block(x_apostrophe);
        let res_it = res.unwrap();
        assert_eq!(res_it.layer_index, 1);
        assert_eq!(res_it.potential_children.len(), 1);
        assert_eq!(res_it.potential_children[0], yz);
        assert!(res_it.parent.is_some());
        assert_eq!(it.tree.borrow().layers.len(), 2);
    }
    #[test]
    fn tree_iterator_create_children_block_updates_bp() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(5);
        let xy = create_set32!('x', 'y');

        // We need one valid child so that there will be an xy children block
        let z_apos = create_set32!('z', '\'');
        it.set_child_valid(z_apos);

        // creates 3 bytes an iteration, u16 max is 65535.
        // 21845 iterations (u16 max / 3) will create 65535 bytes (index 65534, one less than u16 max)
        // to create a new base pointer, we must go over u16 max, so we + 1
        let mut next_iter: Option<TreeIterator> = None;

        for i in 0..u16::MAX / 3 + 1 {
            next_iter = it.create_children_block(xy);
            println!("{}", i);
        }
        assert_eq!(
            next_iter.unwrap().tree.borrow().layers[0].base_ptrs.len(),
            1
        );
        next_iter = it.create_children_block(xy);
        let next_iter = next_iter.unwrap();
        it.set_ptr_to_next_layer_ptr(next_iter.ptr);
        assert_eq!(next_iter.tree.borrow().layers[0].base_ptrs.len(), 2);
        assert_eq!(next_iter.tree.borrow().layers[0].base_ptrs[1].1, 65535 + 3);
    }
    #[test]
    fn tree_iterator_get_all_key_combinations_5_keys() {
        let result = TreeIterator::get_all_key_combinations(5);
        // 27 choose 5 + 27 choose 4 + 27 choose 3 + 27 choose 2
        assert_eq!(result.len(), 101556);
    }
    #[test]
    fn tree_iterator_get_all_key_combinations_is_0_when_1() {
        let result = TreeIterator::get_all_key_combinations(1);
        assert_eq!(result.len(), 0);
    }
    #[test]
    fn tree_iterator_combinations_to_sets() {
        let result =
            TreeIterator::combinations_to_set32s(&TreeIterator::get_all_key_combinations(3));
        let z_apostrophe = create_set32_str!("z\'");
        assert_eq!(result.len(), 3276);
        assert_eq!(result[0], z_apostrophe);
    }
    #[test]
    fn tree_iterator_set_base_correct_values() {
        let mut it = TreeIterator {
            tree: Rc::new(RefCell::new(BinAryTree::default())),
            potential_children: Vec::new(),
            layer_index: 0,
            parent: None,
            path: Vec::new(),
            ptr: 0,
        };
        it.set_base(5);
        assert_eq!(
            it.tree.borrow().layers[0].nodes.len(),
            TreeIterator::get_all_key_combinations(5).len() + 2
        );
        assert_eq!(
            it.potential_children.len(),
            TreeIterator::get_all_key_combinations(5).len()
        );
        let z_apostrophe = create_set32_str!("z\'");
        let y_apostrophe = create_set32_str!("y\'");
        assert_eq!(it.potential_children[0], z_apostrophe);
        assert_eq!(it.potential_children[1], y_apostrophe);
    }
    #[test]
    fn tree_iterator_remove_same_letter_children_is_zero_when_invalid_children() {
        let pot_valid_children: Vec<Set32> = vec![Set32::singleton(3)];
        let child = Set32::singleton(3);
        assert_eq!(
            TreeIterator::remove_children_sharing_letters(&pot_valid_children, child).len(),
            0
        );
    }
    #[test]
    fn tree_iterator_remove_same_letter_children_is_zero_when_valid_children() {
        let pot_valid_children: Vec<Set32> = vec![
            Set32::fill_from_left(2),
            Set32::singleton(4),
            Set32::fill_from_right(2),
        ];
        let child = Set32::singleton(3);
        assert_eq!(
            TreeIterator::remove_children_sharing_letters(&pot_valid_children, child).len(),
            3
        );
    }
}
