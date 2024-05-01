use std::cmp::Ordering;

use crate::{
    globals::{INDEX_TO_CHAR, SET32_SIZE},
    set32::Set32,
};

#[macro_export]
macro_rules! create_set32 {
    ($($ch:expr),*) => {{
            let mut res = Set32::EMPTY;
            for ch in [$($ch),*] {
                res = res.add(CHAR_TO_INDEX[&ch] as u32);
            }
            res
    }};
}

#[macro_export]
macro_rules! create_set32_str {
    ($chars:expr) => {{
        let mut res = Set32::EMPTY;
        for ch in $chars.chars() {
            res = res.add(CHAR_TO_INDEX[&ch] as u32);
        }
        res
    }};
}
#[macro_export]
macro_rules! create_set32s_vec {
    ($chars:expr) => {{
        let sets = $chars.split(',')
                         .map(|ch| {
                             let mut set = Set32::EMPTY;
                             for c in ch.trim().chars() {
                                 set = set.add(CHAR_TO_INDEX[&c] as u32);
                             }
                             set
                         })
                         .collect::<Vec<Set32>>();
        sets
    }};
}


// pub(crate) use create_set32;
pub fn bits_to_num_bytes(num_bits: usize) -> usize {
    (num_bits + 7) / 8
}

pub fn bits_to_byte_index(num_bits: usize) -> usize {
    num_bits / 8
}

pub fn u8slice_to_set(slice: &[u8]) -> Set32 {
    Set32(u32::from_ne_bytes(slice.try_into().unwrap()))
}

pub fn set_to_u8slice(set: &Set32) -> [u8; 4] {
    set.value().to_ne_bytes()
}

pub fn set32_to_string(s: Set32) -> String {
    let mut res: String = String::new();
    for index in s.ones_indices() {
        res.push_str(INDEX_TO_CHAR[&index.try_into().unwrap()].as_str());
    }
    res.chars().rev().collect()
}

pub fn set32s_to_string(sets: &[Set32]) -> String {
    let mut res: String = String::new();
    for s in sets {
        for index in s.ones_indices().iter().rev() {
            res.push_str(INDEX_TO_CHAR[&(*index as u8)].as_str());
        }
        res.push(' ');
    }
    res.pop();
    res
}

pub fn u8s_to_sets(u8s: &Vec<u8>) -> Vec<Set32> {
    let mut ptr = 0;
    let mut res: Vec<Set32> = Vec::new();
    while ptr < u8s.len() {
        let slice = &u8s[ptr..ptr + 4];
        let bytes: [u8; SET32_SIZE] = slice.try_into().expect("Slice with incorrect size");
        res.push(Set32(u32::from_ne_bytes(bytes)));
        ptr += SET32_SIZE;
    }
    res
}

pub fn compare_letters(probe: Set32, other: Set32) -> Ordering {
    if probe == other {
        return Ordering::Equal;
    }
    let ones_probe = probe.ones_indices();
    let ones_target = other.ones_indices();
    if ones_probe.len() > ones_target.len() {
        return Ordering::Greater;
    } else if ones_probe.len() < ones_target.len() {
        return Ordering::Less;
    }
    if probe > other {
        return Ordering::Greater;
    }
    return Ordering::Less;
}


#[cfg(test)]
mod tests {
    use crate::util::{bits_to_byte_index, bits_to_num_bytes, u8s_to_sets};
    #[test]
    fn util_bits_to_bytes() {
        assert_eq!(bits_to_num_bytes(16), 2);
        assert_eq!(bits_to_num_bytes(15), 2);
        assert_eq!(bits_to_num_bytes(8), 1);
        assert_eq!(bits_to_num_bytes(7), 1);
    }

    #[test]
    fn util_bits_to_byte_index() {
        assert_eq!(bits_to_byte_index(16), 2);
        assert_eq!(bits_to_byte_index(15), 1);
        assert_eq!(bits_to_byte_index(8), 1);
        assert_eq!(bits_to_byte_index(7), 0);
    }
    #[test]
    fn util_u8s_to_sets_has_4_items_when_u8s_has_16() {
        let u8s = &vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let res = u8s_to_sets(&u8s);
        assert_eq!(res.len(), 4);
    }
    #[test]
    fn util_u8s_to_sets_has_right_values_when_not_empty() {
        let u8s = &vec![1, 0, 0, 0, 2, 0, 0, 0];
        let res = u8s_to_sets(&u8s);
        assert_eq!(res[0].value(), 1);
        assert_eq!(res[1].value(), 2);
    }
}
