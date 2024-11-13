use std::iter::Iterator;

pub struct DatasetIterator<'a, I> {
    current: usize,
    dataset: &'a dyn Dataset<I>,
}

impl<'a, I> DatasetIterator<'a, I> {
    pub fn new<D>(dataset: &'a D) -> Self
    where 
        D: Dataset<I>
    {
        DatasetIterator {
            current: 0,
            dataset
        }
    }
}

impl<'a, I> Iterator for DatasetIterator<'a, I> {
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.dataset.get(self.current);
        self.current += 1;
        item
    }
}

pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;

    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized
    {
        DatasetIterator::new(self)
    }
}
