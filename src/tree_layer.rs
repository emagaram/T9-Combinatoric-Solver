pub struct TreeLayer {
    pub nodes:Vec<u8>,
    pub base_ptrs: Vec<(usize,usize)>
}

impl Default for TreeLayer {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            base_ptrs: vec![(0,0)]
        }
    }
}