use std::ops::Deref;

// Nothing is optimized here, this is just an experiment to see how the code behaves.

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct TernaryNode {
    pub index: usize,
    pub descendants: Vec<usize>,
}

impl Deref for TernaryNode {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.descendants
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct TernaryMapping {
    pub nodes: Vec<TernaryNode>,
}

impl Deref for TernaryMapping {
    type Target = Vec<TernaryNode>;

    fn deref(&self) -> &Self::Target {
        &self.nodes
    }
}

impl TernaryMapping {
    pub fn new(size: usize) -> Self {
        let mut tree = Self { nodes: Vec::new() };

        for _ in 0..size {
            tree.push();
        }

        tree
    }

    pub fn push(&mut self) {
        let index = self.nodes.len();

        let descendants = Self::generate_descendants(index);

        self.nodes.push(TernaryNode { index, descendants });
    }

    fn generate_descendants(index: usize) -> Vec<usize> {
        let mut descendants = vec![];
        let mut stack = vec![index];

        while let Some(current) = stack.pop() {
            let mut n = current;
            let mut factor = 1;
            let mut found_two = false;

            while n > 0 {
                if n % 3 == 2 {
                    found_two = true;
                    for replacement in 0..=1 {
                        stack.push(current - 2 * factor + replacement * factor);
                    }
                    break;
                }
                n /= 3;
                factor *= 3;
            }

            if !found_two {
                descendants.push(2 * Self::to_binary(current));
            }
        }

        descendants.sort_unstable();
        descendants
    }

    const fn to_binary(mut n: usize) -> usize {
        let mut result = 0;
        let mut factor = 1;

        while n > 0 {
            let digit = n % 3;
            n /= 3;

            result += digit * factor;
            factor *= 2;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree() {
        let mut tree = TernaryMapping { nodes: Vec::new() };

        for _ in 0..27 {
            tree.push();
        }

        let expected_tree = TernaryMapping {
            nodes: vec![
                TernaryNode { index: 0, descendants: vec![0] },
                TernaryNode { index: 1, descendants: vec![2] },
                TernaryNode { index: 2, descendants: vec![0, 2] },
                TernaryNode { index: 3, descendants: vec![4] },
                TernaryNode { index: 4, descendants: vec![6] },
                TernaryNode { index: 5, descendants: vec![4, 6] },
                TernaryNode { index: 6, descendants: vec![0, 4] },
                TernaryNode { index: 7, descendants: vec![2, 6] },
                TernaryNode { index: 8, descendants: vec![0, 2, 4, 6] },
                TernaryNode { index: 9, descendants: vec![8] },
                TernaryNode { index: 10, descendants: vec![10] },
                TernaryNode { index: 11, descendants: vec![8, 10] },
                TernaryNode { index: 12, descendants: vec![12] },
                TernaryNode { index: 13, descendants: vec![14] },
                TernaryNode { index: 14, descendants: vec![12, 14] },
                TernaryNode { index: 15, descendants: vec![8, 12] },
                TernaryNode { index: 16, descendants: vec![10, 14] },
                TernaryNode { index: 17, descendants: vec![8, 10, 12, 14] },
                TernaryNode { index: 18, descendants: vec![0, 8] },
                TernaryNode { index: 19, descendants: vec![2, 10] },
                TernaryNode { index: 20, descendants: vec![0, 2, 8, 10] },
                TernaryNode { index: 21, descendants: vec![4, 12] },
                TernaryNode { index: 22, descendants: vec![6, 14] },
                TernaryNode { index: 23, descendants: vec![4, 6, 12, 14] },
                TernaryNode { index: 24, descendants: vec![0, 4, 8, 12] },
                TernaryNode { index: 25, descendants: vec![2, 6, 10, 14] },
                TernaryNode { index: 26, descendants: vec![0, 2, 4, 6, 8, 10, 12, 14] },
            ],
        };

        assert_eq!(tree, expected_tree);
    }
}
