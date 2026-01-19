//! Data Types - Axonml Type System
//!
//! Defines the data types supported by Axonml tensors and provides traits
//! for type-safe operations. Supports floating point (f16, f32, f64),
//! integer (i8, i16, i32, i64, u8), and boolean types.
//!
//! # Key Features
//! - Type-safe numeric operations via traits
//! - Runtime dtype information via `DType` enum
//! - Half-precision (f16) support
//! - Automatic type promotion rules
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use bytemuck::{Pod, Zeroable};
use half::f16;
use num_traits::{Float as NumFloat, Num, NumCast, One, Zero};

use core::fmt::Debug;

// =============================================================================
// DType Enum
// =============================================================================

/// Runtime representation of tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 16-bit floating point (half precision).
    F16,
    /// 32-bit floating point (single precision).
    F32,
    /// 64-bit floating point (double precision).
    F64,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// Boolean type.
    Bool,
}

impl DType {
    /// Returns the size in bytes of this data type.
    #[must_use]
    pub const fn size_of(self) -> usize {
        match self {
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::F16 | Self::I16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// Returns true if this is a floating point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Returns true if this is a signed integer type.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::F32 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    /// Returns true if this is an integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U8 | Self::U32 | Self::U64
        )
    }

    /// Returns the name of this data type as a string.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        }
    }

    /// Returns the default floating point type (f32).
    #[must_use]
    pub const fn default_float() -> Self {
        Self::F32
    }

    /// Returns the default integer type (i64).
    #[must_use]
    pub const fn default_int() -> Self {
        Self::I64
    }
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// =============================================================================
// Scalar Trait
// =============================================================================

/// Trait for all scalar types that can be stored in a tensor.
///
/// This is the base trait that all tensor element types must implement.
pub trait Scalar: Copy + Clone + Debug + Default + Send + Sync + Pod + Zeroable + 'static {
    /// The runtime dtype for this scalar type.
    const DTYPE: DType;

    /// Returns the dtype for this type.
    #[must_use] fn dtype() -> DType {
        Self::DTYPE
    }
}

// =============================================================================
// Numeric Trait
// =============================================================================

/// Trait for numeric types that support arithmetic operations.
pub trait Numeric: Scalar + Num + NumCast + PartialOrd + Zero + One {
    /// The zero value for this type.
    const ZERO: Self;

    /// The one value for this type.
    const ONE: Self;

    /// Returns the minimum value for this type.
    fn min_value() -> Self;

    /// Returns the maximum value for this type.
    fn max_value() -> Self;
}

// =============================================================================
// Float Trait
// =============================================================================

/// Trait for floating point types.
pub trait Float: Numeric + NumFloat {
    /// Not a Number value.
    const NAN: Self;

    /// Positive infinity.
    const INFINITY: Self;

    /// Negative infinity.
    const NEG_INFINITY: Self;

    /// Machine epsilon.
    const EPSILON: Self;

    /// Returns true if this value is NaN.
    fn is_nan_value(self) -> bool;

    /// Returns true if this value is infinite.
    fn is_infinite_value(self) -> bool;

    /// Returns the exponential of this value.
    fn exp_value(self) -> Self;

    /// Returns the natural logarithm of this value.
    fn ln_value(self) -> Self;

    /// Returns this value raised to the power of `exp`.
    fn pow_value(self, exp: Self) -> Self;

    /// Returns the square root of this value.
    fn sqrt_value(self) -> Self;

    /// Returns the sine of this value.
    fn sin_value(self) -> Self;

    /// Returns the cosine of this value.
    fn cos_value(self) -> Self;

    /// Returns the hyperbolic tangent of this value.
    fn tanh_value(self) -> Self;
}

// =============================================================================
// Scalar Implementations
// =============================================================================

macro_rules! impl_scalar {
    ($ty:ty, $dtype:expr) => {
        impl Scalar for $ty {
            const DTYPE: DType = $dtype;
        }
    };
}

impl_scalar!(f32, DType::F32);
impl_scalar!(f64, DType::F64);
impl_scalar!(i8, DType::I8);
impl_scalar!(i16, DType::I16);
impl_scalar!(i32, DType::I32);
impl_scalar!(i64, DType::I64);
impl_scalar!(u8, DType::U8);
impl_scalar!(u32, DType::U32);
impl_scalar!(u64, DType::U64);

// f16 needs special handling because bytemuck doesn't impl Pod for half::f16 by default
unsafe impl Zeroable for F16Wrapper {}
unsafe impl Pod for F16Wrapper {}

/// Wrapper type for f16 to implement Pod.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct F16Wrapper(pub f16);

impl Scalar for F16Wrapper {
    const DTYPE: DType = DType::F16;
}

// Bool wrapper for Pod compatibility
/// Wrapper type for bool to implement Pod.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct BoolWrapper(pub u8);

unsafe impl Zeroable for BoolWrapper {}
unsafe impl Pod for BoolWrapper {}

impl Scalar for BoolWrapper {
    const DTYPE: DType = DType::Bool;
}

impl From<bool> for BoolWrapper {
    #[allow(clippy::bool_to_int_with_if)]
    fn from(b: bool) -> Self {
        Self(if b { 1 } else { 0 })
    }
}

impl From<BoolWrapper> for bool {
    fn from(b: BoolWrapper) -> Self {
        b.0 != 0
    }
}

// =============================================================================
// Numeric Implementations
// =============================================================================

macro_rules! impl_numeric {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Numeric for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;

            fn min_value() -> Self {
                <$ty>::MIN
            }

            fn max_value() -> Self {
                <$ty>::MAX
            }
        }
    };
}

impl_numeric!(f32, 0.0, 1.0);
impl_numeric!(f64, 0.0, 1.0);
impl_numeric!(i8, 0, 1);
impl_numeric!(i16, 0, 1);
impl_numeric!(i32, 0, 1);
impl_numeric!(i64, 0, 1);
impl_numeric!(u8, 0, 1);

// =============================================================================
// Float Implementations
// =============================================================================

macro_rules! impl_float {
    ($ty:ty) => {
        impl Float for $ty {
            const NAN: Self = <$ty>::NAN;
            const INFINITY: Self = <$ty>::INFINITY;
            const NEG_INFINITY: Self = <$ty>::NEG_INFINITY;
            const EPSILON: Self = <$ty>::EPSILON;

            fn is_nan_value(self) -> bool {
                self.is_nan()
            }

            fn is_infinite_value(self) -> bool {
                self.is_infinite()
            }

            fn exp_value(self) -> Self {
                self.exp()
            }

            fn ln_value(self) -> Self {
                self.ln()
            }

            fn pow_value(self, exp: Self) -> Self {
                self.powf(exp)
            }

            fn sqrt_value(self) -> Self {
                self.sqrt()
            }

            fn sin_value(self) -> Self {
                self.sin()
            }

            fn cos_value(self) -> Self {
                self.cos()
            }

            fn tanh_value(self) -> Self {
                self.tanh()
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::Bool.size_of(), 1);
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
    }

    #[test]
    fn test_scalar_dtype() {
        assert_eq!(f32::dtype(), DType::F32);
        assert_eq!(f64::dtype(), DType::F64);
        assert_eq!(i32::dtype(), DType::I32);
    }

    #[test]
    fn test_numeric_constants() {
        assert_eq!(f32::ZERO, 0.0);
        assert_eq!(f32::ONE, 1.0);
        assert_eq!(i32::ZERO, 0);
        assert_eq!(i32::ONE, 1);
    }

    #[test]
    fn test_float_operations() {
        let x: f32 = 2.0;
        assert!((x.exp_value() - 2.0_f32.exp()).abs() < f32::EPSILON);
        assert!((x.sqrt_value() - 2.0_f32.sqrt()).abs() < f32::EPSILON);
    }
}
