use std::{cell::RefCell, rc::Rc};

/*

Start at root with target layer = 0 or w/e:
- if target layer found, 

DFS through root node children, if at target layer. e.g. AB + CD, we need to look at children of AB and children of CD 
and iterate through those children to evaluate

If at target layer == 0, children to evaluate is all
If at target layer == 1, e.g. AB, children to evaluate are all predecessors's children that are left of us (just root's children) 
If at target layer == 2, e.g. AB CDchildren to evaluate are all predecessors's children that are left of us (AB's children and CD's children)

So we find all parents and intersect

1. DFS through all nodes at current layer 
    If current layer == 0, save scores of all nodess < threshold, binary search through them to make sure 
    that all n-1 sized nodes are there
2. Create scorecasting trees
3. Loop through all nodes at all layers and see if any break new score casting limits, remove any nodes/branches 
    Every branch has an expected height, anything that doesn't meet that expected height should be removed
4. 

*/


use crate::set32::Set32;

pub trait ChildrenContainer {
    fn children_mut(&mut self) -> &mut Vec<Rc<RefCell<Node>>>;
    fn children(&self) -> &Vec<Rc<RefCell<Node>>>;
}

pub struct Node {
    pub score: f32,
    pub letters: Set32,
    pub children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    pub fn new(score: f32, letters: Set32, children: Vec<Rc<RefCell<Node>>>) -> Self {
        Node {
            score,
            letters,
            children,
        }
    }
}
