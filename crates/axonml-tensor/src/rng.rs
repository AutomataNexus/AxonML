//! Optional deterministic RNG for tensor creation and weight initialisation.
//!
//! Every random path (`randn`, `rand`, `uniform_range`, `orthogonal`, `sparse`, and hence every
//! initialiser built on them) reached for `rand::thread_rng()` directly, so a model's weights could
//! not be made reproducible. That makes convergence-style assertions inherently flaky: an unlucky
//! draw diverges and the test fails for reasons unrelated to the code under test.
//!
//! Installing a seed is opt-in and THREAD-LOCAL — an un-seeded thread keeps using `thread_rng()`,
//! so nothing changes for normal training runs. Scope it with [`with_seed`] rather than leaving a
//! seed installed, so one test cannot silently determine another's draws.

use std::cell::RefCell;

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

thread_local! {
    static SEEDED: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Installs a deterministic RNG for this thread. Subsequent random tensor creation and weight
/// initialisation draw from it instead of `thread_rng()`.
pub fn set_seed(seed: u64) {
    SEEDED.with(|s| *s.borrow_mut() = Some(StdRng::seed_from_u64(seed)));
}

/// Removes any deterministic RNG installed on this thread, restoring `thread_rng()`.
pub fn clear_seed() {
    SEEDED.with(|s| *s.borrow_mut() = None);
}

/// Whether this thread currently has a deterministic RNG installed.
#[must_use]
pub fn is_seeded() -> bool {
    SEEDED.with(|s| s.borrow().is_some())
}

/// Runs `f` with `seed` installed, restoring the previous state afterwards (including on unwind).
pub fn with_seed<R>(seed: u64, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<StdRng>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SEEDED.with(|s| *s.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(SEEDED.with(|s| s.borrow_mut().take()));
    set_seed(seed);
    f()
}

/// Runs `f` with whichever RNG is active on this thread — the installed deterministic one if any,
/// otherwise `thread_rng()`. `Rng` is blanket-implemented for `RngCore + ?Sized`, so the closure can
/// call `gen`, `gen_range` and `Distribution::sample` on the argument as usual.
pub fn with_rng<R>(f: impl FnOnce(&mut dyn RngCore) -> R) -> R {
    SEEDED.with(|s| {
        let mut slot = s.borrow_mut();
        match slot.as_mut() {
            Some(seeded) => f(seeded),
            None => f(&mut rand::thread_rng()),
        }
    })
}

// ── tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_draws_are_reproducible() {
        let a = with_seed(1234, || crate::randn::<f32>(&[64]).to_vec());
        let b = with_seed(1234, || crate::randn::<f32>(&[64]).to_vec());
        assert_eq!(a, b, "same seed must give identical draws");
    }

    #[test]
    fn different_seeds_differ() {
        let a = with_seed(1, || crate::randn::<f32>(&[64]).to_vec());
        let b = with_seed(2, || crate::randn::<f32>(&[64]).to_vec());
        assert_ne!(a, b, "different seeds should give different draws");
    }

    #[test]
    fn seed_is_scoped_and_restored() {
        assert!(!is_seeded());
        with_seed(7, || assert!(is_seeded()));
        assert!(!is_seeded(), "with_seed must restore the previous state");
    }
}
