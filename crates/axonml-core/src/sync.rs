//! Reader-writer lock used by tensor storage, chosen per target.
//!
//! With `std`: `parking_lot` (fast, OS-parked). Without `std`: `spin`, a
//! `no_std` spinlock with the same `read()` / `write()` guard API, so storage
//! code is identical on both.

#[cfg(feature = "std")]
pub use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(not(feature = "std"))]
pub use spin::{RwLock, RwLockReadGuard, RwLockWriteGuard};
