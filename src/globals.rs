use std::collections::HashMap;
use lazy_static::lazy_static;

use crate::set32::Set32;

pub const SET32_SIZE: usize = std::mem::size_of::<Set32>();
pub const U16_SIZE: usize = std::mem::size_of::<u16>();
pub const NUM_LETTERS: usize = 27;
pub const TEST_MODE: bool = false;
lazy_static! {
    pub static ref ASCII_CODE_TO_INDEX: HashMap<u8, u8> = {
        let mut m = HashMap::new();
        m.insert(39, 0); // Apostrophe
        for i in 1..=26 {
            m.insert(97u8 + 26 - i, i);
        }
        m
    };

    pub static ref INDEX_TO_CHAR: HashMap<u8, String> = {
        let mut m = HashMap::new();
        m.insert(0, "\'".to_string()); // Apostrophe
        for i in 1..=26 {
            // 1, 'a' + 26 - 1 = 'z'
            let char_value = (97u8 + 26 - i) as char;
            m.insert(i, char_value.to_string());
        }
        m
    };

    pub static ref CHAR_TO_INDEX: HashMap<char, u8> = {
        let mut m = HashMap::new();
        m.insert('\'', 0); // Apostrophe
        for i in 1..=26 { 
            let char_value = (97u8 + 26 - i as u8) as char;
            m.insert(char_value, i);
        }
        m
    };    
}