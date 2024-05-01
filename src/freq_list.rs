use std::{collections::HashMap, fs::File, io::BufReader};

use crate::globals::ASCII_CODE_TO_INDEX;

pub type FreqList = Vec<(Vec<u8>, f32)>;

pub fn create_freq_list(dict_path: &str) -> FreqList {
    let file = File::open(dict_path).expect("file not found");
    let reader = BufReader::new(file);
    let word_frequencies: HashMap<String, f32> =
        serde_json::from_reader(reader).expect("read json properly");
    let mut freq_list: FreqList = word_frequencies
        .into_iter() // Use `.into_iter()` to consume the hashmap and avoid cloning the keys
        .map(|(key, value)| {
            (
                key.into_bytes()
                    .iter()
                    .map(|b| ASCII_CODE_TO_INDEX[b])
                    .collect(),
                value,
            )
        }) // Convert each key into a Vec<u8>
        .collect();
    freq_list.sort_by(|a, b| {
        b.1.partial_cmp(&a.1) // First, try to compare the values
            .unwrap_or(std::cmp::Ordering::Equal) // In case of NaNs or partial comparison failure
            .then_with(|| a.0.cmp(&b.0)) // If values are equal, sort by keys in ascending order
    });
    freq_list
}

#[cfg(test)]
mod tests {
    use super::create_freq_list;
    #[test]
    fn freq_list_has_right_num_bytes() {
        let list = create_freq_list("./small_list.json");
        assert_eq!(list[0].0.len(), 3);
        assert_eq!(list[1].0.len(), 4);
        assert_eq!(list[2].0.len(), 10);
    }
    #[test]
    fn freq_list_init_same() {
        let list1 = create_freq_list("./word_freq.json");
        let list2 = create_freq_list("./word_freq.json");
        assert_eq!(list1[1000].0, list2[1000].0);
        assert_eq!(list1[5000].0, list2[5000].0);
        assert_eq!(list1[50000].0, list2[50000].0);
    }
}
