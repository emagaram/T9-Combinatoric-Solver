use crate::{globals::NUM_LETTERS, set32::Set32, util::set32s_to_string};
use rayon::prelude::*;
use std::{cmp::min, collections::HashMap};
pub fn get_alphabet_set(a: &Set32) -> Set32 {
    let mask = Set32::fill_from_right(NUM_LETTERS.try_into().unwrap());
    a.intersect(mask)
}
fn config_to_t10_config_full(config: &Vec<Set32>, config_hashmap: &mut HashMap<u8, u8>) {
    for (config_index, set) in config.iter().enumerate() {
        let key_indices = set.ones_indices();
        for key_index in key_indices {
            config_hashmap.insert(
                key_index.try_into().unwrap(),
                config_index.try_into().unwrap(),
            );
        }
    }
}

fn word_to_t10(word: &Vec<u8>, t10_config: &HashMap<u8, u8>) -> Vec<u8> {
    word.iter()
        .map(|b| *t10_config.get(&b).expect("Should have a key"))
        .collect()
}

pub fn new_evaluate(freq_list: &Vec<(Vec<u8>, f32)>, config: &Vec<Set32>, stop_at: f32) -> (f32, usize) {
    let unused_letters = get_missing_letters(config);
    let num_keys = config.len() + unused_letters.len();

    // println!("At {}, adding {} to score", set32s_to_string(&config), score_to_add);
    let mut full_config = config.clone();
    full_config.extend(& unused_letters);
    // Threshold 0.1, add 0.05, we should stop checking once we hit 0.05, because we're adding 0.05
    let base_score = evaluate(freq_list, &full_config, stop_at);
    return (base_score, num_keys);
}

pub fn evaluate(freq_list: &Vec<(Vec<u8>, f32)>, config: &Vec<Set32>, stop_at: f32) -> f32 {
    if get_missing_letters(&config).len() > 0 {
        panic!("Config missing letters!");
    }
    let mut config_hm: HashMap<u8, u8> = HashMap::new();
    let mut count_hm: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut score = 0.0;
    config_to_t10_config_full(&config, &mut config_hm);
    for (word, value) in freq_list {
        let t10_word = word_to_t10(word, &config_hm);
        let count = count_hm.get(&t10_word).unwrap_or(&0);
        if *count > 0 {
            score += value * (min(4, *count) as f32);
            if score > stop_at {
                break;
            }
        }
        count_hm.insert(t10_word, count + 1);
    }
    return score;
}

pub fn evaluate_using_previous_solutions(
    freq_list: &Vec<(Vec<u8>, f32)>,
    config: &Vec<Set32>,
    stop_at: f32,
    target_num_keys: usize,
    best_scores: &Vec<f32>,
) -> (f32, usize) {
    let unused_letters = get_missing_letters(config);
    let num_keys = config.len() + unused_letters.len();

    // If target is 26, and we're at 26 then we'll do (26 - 26) - 1 == -1.
    // If target is 25, and we're at 26 then we'll do (26 - 25) - 1 = 0.
    let best_score_index_to_add = num_keys - target_num_keys; // a b c d.. ab c d... abc d.. ab ac d
    let score_to_add = if best_score_index_to_add == 0 {
        0.0
    } else {
        best_scores[best_score_index_to_add - 1]
    };
    // println!("At {}, adding {} to score", set32s_to_string(&config), score_to_add);
    let mut full_config = config.clone();
    full_config.extend(& unused_letters);
    // Threshold 0.1, add 0.05, we should stop checking once we hit 0.05, because we're adding 0.05
    let mut base_score = evaluate(freq_list, &full_config, stop_at - score_to_add);
    base_score += score_to_add;
    return (base_score, num_keys);
}

pub fn evaluate_full(
    freq_list: &Vec<(Vec<u8>, f32)>,
    config: &Vec<Set32>,
    stop_at: f32,
    target_num_keys: usize,
) -> (f32, usize, usize) {
    let unused_letters = get_missing_letters(config);
    let num_keys = config.len() + unused_letters.len();
    // println!("Config num keys: {}", num_keys);
    let variations = merge_slices(&config, &unused_letters);
    // for variation in variations {
    //     let score = evaluate(freq_list, &variation, stop_at);
    //     scores.push(score);
    // }
    let mut scores: Vec<_> = variations
        .par_iter()
        .map(|variation| evaluate(freq_list, variation, stop_at))
        .collect();
    let num_merges_needed = num_keys - target_num_keys;
    // println!("Num merges needed: {}", num_keys);
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut full_config = config.clone();
    full_config.extend(&unused_letters);
    let base_score = evaluate(freq_list, &full_config, stop_at);

    let mut score_sum = base_score;
    for i in 0..num_merges_needed {
        let add = scores[i] - base_score;
        println!("\tAdding {} to {}", add, set32s_to_string(config));
        score_sum += add;
    }
    return (score_sum, num_keys, NUM_LETTERS - unused_letters.len());
}

fn get_missing_letters(config: &Vec<Set32>) -> Vec<Set32> {
    let mut missing_letters_set = Set32::EMPTY;
    let mut res = Vec::new();
    for key in config {
        missing_letters_set = missing_letters_set.union(*key);
    }
    missing_letters_set = get_alphabet_set(&Set32(!missing_letters_set.value()));
    for one in missing_letters_set.ones_indices() {
        res.push(Set32::singleton(one));
    }
    res
}

// fn generate_config_variations(config: &Vec<Set32>)->Vec<Vec<Set32>>{
//     let missing_letters = get_missing_letters(config);
//     let mut full_config = config.clone();
//     let res:Vec<Vec<Set32>> = vec![];
//     for letter in missing_letters.ones_indices() {
//         full_config.push(Set32::singleton(letter));
//     }
//     res
// }
// fn merge_adjacent(items: &[Set32]) -> Vec<Vec<Set32>> {
//     let mut results = Vec::new();

//     for i in 0..items.len() {
//         for j in i + 1..items.len() {
//             let mut merged = Vec::new();

//             // Add all elements before the first element to be merged
//             if i > 0 {
//                 merged.extend(items[..i].iter().cloned());
//             }

//             // Merge the i-th and j-th items
//             let merged_item = items[i].union(items[j]);
//             merged.push(merged_item);

//             // Add elements between the merged items
//             if i + 1 < j {
//                 merged.extend(items[i + 1..j].iter().cloned());
//             }

//             // Add all elements after the merged pair
//             if j + 1 < items.len() {
//                 merged.extend(items[j + 1..].iter().cloned());
//             }

//             results.push(merged);
//         }
//     }

//     results
// }

fn merge_slices(keys: &Vec<Set32>, unused_letters: &Vec<Set32>) -> Vec<Vec<Set32>> {
    let mut results = Vec::new();

    // Pair each item in `unused_letters` with each item in `keys`
    for (key_index, key) in keys.iter().enumerate() {
        for (let_index, unused) in unused_letters.iter().enumerate() {
            let mut merged = Vec::new();
            let mut new_keys = keys.clone();
            new_keys.remove(key_index);
            let merged_key = key.union(*unused);
            let mut new_unused = unused_letters.clone();
            new_unused.remove(let_index);
            merged.push(merged_key);
            merged.append(&mut new_unused);
            merged.append(&mut new_keys);
            results.push(merged);
        }
    }

    // Pair each item in `unused_letters` with every other item in `unused_letters`
    for i in 0..unused_letters.len() {
        for j in i + 1..unused_letters.len() {
            let mut merged = Vec::new();
            let merged_unused = unused_letters[i].union(unused_letters[j]);
            let mut new_unused = unused_letters.clone();
            new_unused.remove(i);
            new_unused.remove(j - 1);
            merged.push(merged_unused);
            merged.append(&mut keys.to_vec());
            merged.append(&mut new_unused);
            results.push(merged);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{evaluate, evaluate_full, get_missing_letters};
    use crate::{
        create_set32,
        evaluate::{config_to_t10_config_full, merge_slices},
        freq_list::create_freq_list,
        globals::CHAR_TO_INDEX,
        set32::Set32,
        util::set32s_to_string,
    };

    #[test]
    fn evaluate_full_best_10_key() {
        let freq_list: Vec<(Vec<u8>, f32)> = create_freq_list("./word_freq.json");
        let target = 0.024770784;
        // Everything except rp
        let config = vec![
            create_set32!('g', 'm', 'x'),
            create_set32!('z', 'e', 'f'),
            create_set32!('q', '\'', 'h', 'd'),
            create_set32!('l', 'j', 'i'),
            create_set32!('o', 't'),
            create_set32!('s', 'u'),
            create_set32!('a', 'k', 'w'),
            create_set32!('n', 'b'),
            create_set32!('v', 'y', 'c'),
        ];
        let (score1, _, _) = evaluate_full(&freq_list, &config, target, 10);
        assert!(score1 <= target);

        // Everything except rp, vyc
        let config = vec![
            create_set32!('g', 'm', 'x'),
            create_set32!('z', 'e', 'f'),
            create_set32!('q', '\'', 'h', 'd'),
            create_set32!('l', 'j', 'i'),
            create_set32!('o', 't'),
            create_set32!('s', 'u'),
            create_set32!('a', 'k', 'w'),
            create_set32!('n', 'b'),
        ];
        let (score2, _, _) = evaluate_full(&freq_list, &config, target, 10);
        assert!(score2 <= target);

        // Everything except rp, vyc, nb
        let config = vec![
            create_set32!('g', 'm', 'x'),
            create_set32!('z', 'e', 'f'),
            create_set32!('q', '\'', 'h', 'd'),
            create_set32!('l', 'j', 'i'),
            create_set32!('o', 't'),
            create_set32!('s', 'u'),
            create_set32!('a', 'k', 'w'),
        ];
        let (score3, _, _) = evaluate_full(&freq_list, &config, target, 10);
        assert!(score3 <= target);

        assert!(score1 >= score2);
        assert!(score2 >= score3);
    }
    #[test]
    fn evaluate_get_missing_letters() {
        let config = vec![
            create_set32!('a'),
            create_set32!('b'),
            create_set32!('c'),
            create_set32!('d', 'e', 'f'),
        ];
        let unused = get_missing_letters(&config);
        assert_eq!(unused.len(), 21);
    }
    #[test]
    fn evaluate_merge_adjacent_complex() {
        let keys = vec![create_set32!('a', 'b'), create_set32!('c', 'd')];
        let unused_letters = get_missing_letters(&keys);
        let merged_sets = merge_slices(&keys, &unused_letters);

        for s in &merged_sets {
            println!("{}", set32s_to_string(&s));
        }
        // assert_eq!(merged_sets.len(), 6);
    }
    #[test]
    fn evaluate_merge_adjacent_simple() {
        let unused_letters = vec![create_set32!('a'), create_set32!('b'), create_set32!('c')];
        let keys = vec![create_set32!('d', 'e')];
        let merged_sets = merge_slices(&keys, &unused_letters);

        for s in &merged_sets {
            println!("{}", set32s_to_string(&s));
        }
        assert_eq!(merged_sets.len(), 6);
    }

    fn str_to_set(s: &str) -> Set32 {
        let mut res = Set32::EMPTY;
        for c in s.chars() {
            res = res.add(CHAR_TO_INDEX[&c].try_into().unwrap());
        }
        res
    }
    #[test]
    fn evaluate_one_char_per_letter_0() {
        let fl = create_freq_list("./word_freq.json");
        let mut config = Vec::<Set32>::new();
        config.push(Set32::fill_from_right(1));
        config.append(&mut get_missing_letters(&config));
        let score = evaluate(&fl, &config, 10000.9);
        assert_eq!(score, 0.0);
    }
    #[test]
    fn evaluate_abc_same_as_python_result() {
        let fl = create_freq_list("./word_freq.json");
        let mut config: Vec<Set32> = vec![
            str_to_set("abc"),
            str_to_set("def"),
            str_to_set("ghi"),
            str_to_set("jkl"),
            str_to_set("mno"),
            str_to_set("pqr"),
            str_to_set("st"),
            str_to_set("uv"),
            str_to_set("wx"),
            str_to_set("yz'"),
        ];

        let mut config_hm: HashMap<u8, u8> = HashMap::new();
        config.extend(&get_missing_letters(&config));
        config_to_t10_config_full(&config, &mut config_hm);
        let score = evaluate(&fl, &config, 10000.9);
        let python_score: f32 = 0.080390571;
        let epsilon = 0.0001;
        assert!((score - python_score).abs() < epsilon);
    }
}
// fn merge_slices2(keys: &[Set32], unused_letters: &[Set32]) -> Vec<Vec<Set32>> {
//     let mut results = Vec::new();

//     // Pre-allocate to avoid frequent resizing
//     results.reserve(
//         keys.len() * unused_letters.len() + unused_letters.len() * (unused_letters.len() - 1) / 2,
//     );

//     // Pair each item in `unused_letters` with each item in `keys`
//     for (key_index, key) in keys.iter().enumerate() {
//         for (let_index, unused) in unused_letters.iter().enumerate() {
//             let merged_key = key.union(*unused);

//             // Use iterators and avoid cloning when possible
//             let mut merged = Vec::with_capacity(keys.len() + unused_letters.len() - 1);
//             merged.push(merged_key);
//             merged.extend(unused_letters.iter().enumerate().filter_map(|(i, x)| {
//                 if i != let_index {
//                     Some(*x)
//                 } else {
//                     None
//                 }
//             }));
//             merged.extend(keys.iter().enumerate().filter_map(|(i, x)| {
//                 if i != key_index {
//                     Some(*x)
//                 } else {
//                     None
//                 }
//             }));

//             results.push(merged);
//         }
//     }

//     // Pair each item in `unused_letters` with every other item in `unused_letters`
//     for i in 0..unused_letters.len() {
//         for j in i + 1..unused_letters.len() {
//             let merged_unused = unused_letters[i].union(unused_letters[j]);
//             let mut merged = Vec::with_capacity(keys.len() + unused_letters.len() - 1);
//             merged.push(merged_unused);
//             merged.extend(keys.iter().copied());
//             merged.extend(unused_letters.iter().enumerate().filter_map(|(index, x)| {
//                 if index != i && index != j {
//                     Some(*x)
//                 } else {
//                     None
//                 }
//             }));

//             results.push(merged);
//         }
//     }
//     results
// }
// fn config_to_t10_config(config: &Vec<Set32>, config_hashmap: &mut HashMap<u8, u8>) {
//     let mut chars_seen = Set32::EMPTY;
//     for (config_index, set) in config.iter().enumerate() {
//         let key_indices = set.ones_indices();
//         chars_seen = chars_seen.union(*set);
//         for key_index in key_indices {
//             config_hashmap.insert(
//                 key_index.try_into().unwrap(),
//                 config_index.try_into().unwrap(),
//             );
//         }
//     }
//     let starting_index = config_hashmap.len() as u8;
//     let chars_not_seen =
//         Set32(get_alphabet_set(&Set32(!chars_seen.value())).value()).ones_indices();
//     for (index, ch) in chars_not_seen.iter().enumerate() {
//         let index: u8 = index.try_into().unwrap();
//         config_hashmap.insert((*ch).try_into().unwrap(), index + starting_index);
//     }
// }
