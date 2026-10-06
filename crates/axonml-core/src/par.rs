//! Data-parallel iteration that degrades to serial iteration without `std`.
//!
//! With the `std` feature this re-exports rayon's prelude and thread count, so
//! the CPU backend and tensor ops run in parallel exactly as before. Without
//! `std` (bare-metal / `no_std` targets) rayon is not available, so the same
//! `par_iter` / `par_iter_mut` / `par_chunks` / `par_chunks_mut` call sites
//! resolve to `serial::Seq`, a serial iterator that mirrors the rayon signatures the
//! crates use (`fold` and `reduce` take an identity closure, as in rayon).
//! One code path, two execution strategies.

/// Parallel-iterator traits, imported as `use axonml_core::par::prelude::*`
/// (rayon's prelude with `std`, the serial stand-ins without).
pub mod prelude {
    #[cfg(feature = "std")]
    pub use rayon::prelude::*;

    #[cfg(not(feature = "std"))]
    pub use super::serial::{IntoParallelIterator, ParallelSlice, ParallelSliceMut, Seq};
}

/// Number of worker threads data-parallel loops fan out across (1 without `std`).
#[cfg(feature = "std")]
#[must_use]
pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "std"))]
pub use serial::current_num_threads;

// Compiled for tests on std hosts too, so its rayon parity is checked in CI.
#[cfg(any(not(feature = "std"), test))]
#[allow(dead_code)]
mod serial {
    /// Number of worker threads data-parallel loops fan out across (always 1).
    #[must_use]
    pub fn current_num_threads() -> usize {
        1
    }

    /// Serial stand-in for a rayon parallel iterator.
    ///
    /// Adapters are inherent methods so that rayon-shaped calls such as
    /// `.fold(|| init, op).reduce(|| init, op)` resolve here rather than to
    /// `Iterator::fold` / `Iterator::reduce`, whose signatures differ.
    #[derive(Clone, Debug)]
    pub struct Seq<I>(pub I);

    impl<I: Iterator> Iterator for Seq<I> {
        type Item = I::Item;
        #[inline]
        fn next(&mut self) -> Option<I::Item> {
            self.0.next()
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    impl<I: Iterator> Seq<I> {
        /// rayon `zip`.
        pub fn zip<J: IntoIterator>(self, other: J) -> Seq<core::iter::Zip<I, J::IntoIter>> {
            Seq(self.0.zip(other))
        }
        /// rayon `map`.
        pub fn map<B, F: FnMut(I::Item) -> B>(self, f: F) -> Seq<core::iter::Map<I, F>> {
            Seq(self.0.map(f))
        }
        /// rayon `enumerate`.
        pub fn enumerate(self) -> Seq<core::iter::Enumerate<I>> {
            Seq(self.0.enumerate())
        }
        /// rayon `filter`.
        pub fn filter<P: FnMut(&I::Item) -> bool>(self, p: P) -> Seq<core::iter::Filter<I, P>> {
            Seq(self.0.filter(p))
        }
        /// rayon `filter_map`.
        pub fn filter_map<B, F: FnMut(I::Item) -> Option<B>>(
            self,
            f: F,
        ) -> Seq<core::iter::FilterMap<I, F>> {
            Seq(self.0.filter_map(f))
        }
        /// rayon `with_min_len` (a scheduling hint; no-op when serial).
        #[must_use]
        pub fn with_min_len(self, _min: usize) -> Self {
            self
        }
        /// rayon `with_max_len` (a scheduling hint; no-op when serial).
        #[must_use]
        pub fn with_max_len(self, _max: usize) -> Self {
            self
        }
        /// rayon `fold`: `identity` seeds the (single) partial result.
        pub fn fold<T, ID: Fn() -> T, F: FnMut(T, I::Item) -> T>(
            self,
            identity: ID,
            op: F,
        ) -> Seq<core::iter::Once<T>> {
            Seq(core::iter::once(self.0.fold(identity(), op)))
        }
        /// rayon `reduce`: combine every item, starting from `identity()`.
        pub fn reduce<ID: Fn() -> I::Item, F: FnMut(I::Item, I::Item) -> I::Item>(
            self,
            identity: ID,
            op: F,
        ) -> I::Item {
            self.0.fold(identity(), op)
        }
    }

    impl<'a, I: Iterator<Item = &'a T>, T: 'a + Clone> Seq<I> {
        /// rayon `cloned`.
        pub fn cloned(self) -> Seq<core::iter::Cloned<I>> {
            Seq(self.0.cloned())
        }
    }

    /// `par_iter` / `par_chunks` on slices (and anything that derefs to one).
    pub trait ParallelSlice<T> {
        /// Serial stand-in for rayon `par_iter`.
        fn par_iter(&self) -> Seq<core::slice::Iter<'_, T>>;
        /// Serial stand-in for rayon `par_chunks`.
        fn par_chunks(&self, size: usize) -> Seq<core::slice::Chunks<'_, T>>;
    }

    impl<T> ParallelSlice<T> for [T] {
        fn par_iter(&self) -> Seq<core::slice::Iter<'_, T>> {
            Seq(self.iter())
        }
        fn par_chunks(&self, size: usize) -> Seq<core::slice::Chunks<'_, T>> {
            Seq(self.chunks(size))
        }
    }

    /// `par_iter_mut` / `par_chunks_mut` on mutable slices.
    pub trait ParallelSliceMut<T> {
        /// Serial stand-in for rayon `par_iter_mut`.
        fn par_iter_mut(&mut self) -> Seq<core::slice::IterMut<'_, T>>;
        /// Serial stand-in for rayon `par_chunks_mut`.
        fn par_chunks_mut(&mut self, size: usize) -> Seq<core::slice::ChunksMut<'_, T>>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        fn par_iter_mut(&mut self) -> Seq<core::slice::IterMut<'_, T>> {
            Seq(self.iter_mut())
        }
        fn par_chunks_mut(&mut self, size: usize) -> Seq<core::slice::ChunksMut<'_, T>> {
            Seq(self.chunks_mut(size))
        }
    }

    /// `into_par_iter` on ranges and other owned iterables.
    pub trait IntoParallelIterator: IntoIterator + Sized {
        /// Serial stand-in for rayon `into_par_iter`.
        fn into_par_iter(self) -> Seq<Self::IntoIter> {
            Seq(self.into_iter())
        }
    }

    impl<I: IntoIterator> IntoParallelIterator for I {}
}

/// The serial stand-ins must produce exactly what rayon does for the shapes the
/// kernels use (identity-seeded fold/reduce, zip, chunks, enumerate). Each
/// side lives in its own module so only one set of traits is in scope.
#[cfg(all(test, feature = "std"))]
mod tests {
    /// One result per kernel shape: sum, max, argmax, zip-add, chunk ids,
    /// range map-sum, chunk lengths.
    type Results = (f32, f32, usize, Vec<f32>, Vec<usize>, usize, Vec<usize>);

    fn data() -> Vec<f32> {
        (0..1000).map(|i| ((i * 37) % 101) as f32 - 50.0).collect()
    }

    macro_rules! shapes {
        ($a:expr) => {{
            let a: &[f32] = $a;
            let b: Vec<f32> = a.iter().map(|v| v * 0.5).collect();
            let sum = a.par_iter().cloned().reduce(|| 0.0, |x, y| x + y);
            let max = a
                .par_iter()
                .cloned()
                .fold(|| a[0], |acc, x| if x > acc { x } else { acc })
                .reduce(|| a[0], |x, y| if y > x { y } else { x });
            let argmax = a
                .par_iter()
                .enumerate()
                .map(|(i, v)| (v, i))
                .reduce(|| (&a[0], 0), |x, y| if y.0 > x.0 { y } else { x })
                .1;
            let mut zipped = vec![0.0f32; a.len()];
            zipped
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (x, y))| *d = x + y);
            let mut ids = vec![0usize; 64];
            ids.par_chunks_mut(10)
                .enumerate()
                .for_each(|(i, c)| c.iter_mut().for_each(|v| *v = i));
            let range_sum: usize = (0..100usize).into_par_iter().map(|i| i * i).sum();
            let lens: Vec<usize> = a.par_chunks(7).map(<[f32]>::len).collect();
            (sum, max, argmax, zipped, ids, range_sum, lens)
        }};
    }

    mod with_rayon {
        use rayon::prelude::*;
        pub(super) fn run(a: &[f32]) -> super::Results {
            shapes!(a)
        }
    }

    mod with_serial {
        use super::super::serial::{IntoParallelIterator, ParallelSlice, ParallelSliceMut};
        pub(super) fn run(a: &[f32]) -> super::Results {
            shapes!(a)
        }
    }

    #[test]
    fn serial_stand_ins_match_rayon() {
        let a = data();
        let (s, p) = (with_serial::run(&a), with_rayon::run(&a));
        assert!((s.0 - p.0).abs() < 1e-3, "sum {} vs {}", s.0, p.0);
        assert_eq!(s.1.to_bits(), p.1.to_bits(), "fold/reduce max (an element, so exact)");
        assert_eq!(s.2, p.2, "argmax");
        assert_eq!(s.3, p.3, "zip for_each");
        assert_eq!(s.4, p.4, "par_chunks_mut enumerate");
        assert_eq!(s.5, p.5, "into_par_iter map sum");
        assert_eq!(s.6, p.6, "par_chunks lengths");
    }
}
